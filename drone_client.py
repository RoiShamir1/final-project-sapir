import cv2
import json
import datetime
import torch
from ultralytics import YOLO

# --- הגדרות ---
# טוען את מודל YOLO11 בגרסת ה-Nano (הקלה ביותר)
# בפעם הראשונה זה יוריד את הקובץ אוטומטית מהאינטרנט
model_path = 'yolo11n.pt' 

# הגדרת סף ביטחון - רק זיהויים מעל 50% יתקבלו
CONFIDENCE_THRESHOLD = 0.5

# בחירת התקן ריצה (הכרטיס מסך שלך)
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running inference on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

# אתחול המודל
model = YOLO(model_path)

# פתיחת מצלמה (0 = מצלמת רשת מובנית)
# אם תרצה לבדוק על סרטון וידאו, החלף את 0 בשם הקובץ, למשל: "test_video.mp4"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

print("🎥 Starting Drone View Simulation... Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- שלב הזיהוי (Inference) ---
    # אנו שולחים את הפריים למודל ומבקשים שירוץ על ה-GPU (device=0)
    results = model(frame, device=device, verbose=False)

    detected_objects = []

    # --- עיבוד התוצאות ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            
            if conf > CONFIDENCE_THRESHOLD:
                # המרת קוד המחלקה (מספר) לשם (טקסט)
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                
                # קואורדינטות הריבוע (Bounding Box)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # הוספה לרשימת האובייקטים שנמצאו
                detected_objects.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

                # --- ויזואליזציה (ציור על המסך) ---
                # צבע אדום לאנשים, ירוק לשאר
                color = (0, 0, 255) if label == 'person' else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- הדמיית שליחה לענן ---
    # אם זוהה אובייקט רלוונטי (כרגע כל אובייקט, בהמשך נסנן רק נשק/אדם)
    if detected_objects:
        alert_payload = {
            "drone_id": "alpha_01",
            "timestamp": datetime.datetime.now().isoformat(),
            "location": {"lat": 32.0853, "lon": 34.7818, "alt": 50}, # מיקום פיקטיבי
            "objects_count": len(detected_objects),
            "detections": detected_objects
        }
        
        # כרגע רק מדפיסים את ה-JSON לטרמינל כדי לראות שזה עובד
        # בשבוע הבא נחליף את השורה הזו בשליחה אמיתית לשרת
        print(f"📡 Sending Data to Cloud: Found {len(detected_objects)} objects")
        # print(json.dumps(alert_payload, indent=2)) # תבטל הערה זו אם תרצה לראות את כל המידע

    # הצגת הוידאו על המסך
    cv2.imshow("Drone View (YOLO11n + RTX 3050)", frame)

    # יציאה בלחיצה על 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()