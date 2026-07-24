# أضف هذا السطر في أعلى الملف بعد استدعاء المكتبات
SECRET_API_KEY = "isubborah_secret_12345"
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def draw_perfect_circle(img, contour, color, thickness):
    (x, y, w, h) = cv2.boundingRect(contour)
    cx = x + w // 2
    cy = y + h // 2
    radius = max(w, h) // 2 + 2
    cv2.circle(img, (cx, cy), int(radius), color, thickness)

@app.post("/process-paper")
async def process_paper(
    file: UploadFile = File(...),
    answers: str = Form(...) 
):
    try:
        parsed_answers = json.loads(answers)
        DYNAMIC_ANSWER_KEY = {int(k): int(v) for k, v in parsed_answers.items()}

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 75, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        doc_cnt = None
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    doc_cnt = approx
                    break
        
        if doc_cnt is None:
            return {"success": False, "message": "لم يتم العثور على الإطار الأسود المكتمل."}

        warped_paper = four_point_transform(img, doc_cnt.reshape(4, 2))
        warped_gray = cv2.cvtColor(warped_paper, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.dilate(thresh, None, iterations=1)

        bubble_contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_bubbles = []
        for c in bubble_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            if 0.8 <= ar <= 1.2 and 20 <= w <= 75 and 20 <= h <= 75:
                raw_bubbles.append(c)

        bubbles = []
        for c in raw_bubbles:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w//2, y + h//2
            is_duplicate = False
            for b in bubbles:
                bx, by, bw, bh = cv2.boundingRect(b)
                bcx, bcy = bx + bw//2, by + bh//2
                if abs(cx - bcx) < 10 and abs(cy - bcy) < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                bubbles.append(c)

        height, width = warped_paper.shape[:2]
        col_left, col_mid, col_right = [], [], []
        
        for c in bubbles:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + (w / 2)
            if cx < width * 0.33:
                col_left.append(c)
            elif cx < width * 0.66:
                col_mid.append(c)
            else:
                col_right.append(c)

        if len(col_mid) != 80 or len(col_left) != 80:
            return {"success": False, "message": "لم يتم العثور على 80 دائرة في أعمدة الأسئلة."}
            
        col_right = sorted(col_right, key=lambda c: cv2.boundingRect(c)[1])
        if len(col_right) >= 30:
            col_right = col_right[:30]
        else:
            return {"success": False, "message": f"عمود رقم الطالب غير مكتمل."}

        output_img = warped_paper.copy()
        
        col_right = sorted(col_right, key=lambda c: cv2.boundingRect(c)[0])
        student_id = ""
        
        for i in range(3):
            digit_bubbles = col_right[i*10 : (i+1)*10]
            digit_bubbles = sorted(digit_bubbles, key=lambda c: cv2.boundingRect(c)[1])
            
            digit_val = "?" 
            for val, bubble in enumerate(digit_bubbles):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                
                mask_pixels = cv2.countNonZero(mask)
                ink_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                
                if mask_pixels > 0 and (ink_pixels / mask_pixels) > 0.75:
                    digit_val = str(val)
                    draw_perfect_circle(output_img, bubble, (255, 0, 255), 3)
                    break
            
            student_id += digit_val

        col_mid = sorted(col_mid, key=lambda c: cv2.boundingRect(c)[1])
        col_left = sorted(col_left, key=lambda c: cv2.boundingRect(c)[1])
        
        all_questions = []
        for q in range(20):
            row = sorted(col_mid[q*4 : (q+1)*4], key=lambda c: cv2.boundingRect(c)[0], reverse=True)
            all_questions.append(row)
        for q in range(20):
            row = sorted(col_left[q*4 : (q+1)*4], key=lambda c: cv2.boundingRect(c)[0], reverse=True)
            all_questions.append(row)

        score = 0
        wrong_answers = [] # المصفوفة الجديدة لجمع الأخطاء

        for q_idx, row_bubbles in enumerate(all_questions):
            student_answer = None
            for opt_idx, bubble in enumerate(row_bubbles):
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [bubble], -1, 255, -1)
                mask_pixels = cv2.countNonZero(mask)
                ink_pixels = cv2.countNonZero(cv2.bitwise_and(thresh, thresh, mask=mask))
                
                if mask_pixels > 0 and (ink_pixels / mask_pixels) > 0.75:
                    student_answer = opt_idx
                    break
            
            correct_answer = DYNAMIC_ANSWER_KEY.get(q_idx)
            
            if correct_answer is not None:
                if student_answer == correct_answer:
                    score += 1
                    draw_perfect_circle(output_img, row_bubbles[student_answer], (0, 255, 0), 3)
                else:
                    # إضافة رقم السؤال (نضيف 1 لأن البرمجة تبدأ من الصفر)
                    wrong_answers.append(q_idx + 1)
                    if student_answer is not None:
                        draw_perfect_circle(output_img, row_bubbles[student_answer], (0, 0, 255), 3)
                        draw_perfect_circle(output_img, row_bubbles[correct_answer], (255, 0, 0), 2)
                    else:
                        draw_perfect_circle(output_img, row_bubbles[correct_answer], (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', output_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "message": "تم التصحيح والتحليل بنجاح!",
            "data": {
                "student_id": student_id,
                "score": f"{score}",
                "total_questions": len(DYNAMIC_ANSWER_KEY),
                "wrong_answers": wrong_answers, # إرسال الأسئلة الخاطئة للمتصفح
                "image_base64": img_base64 
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
