import cv2
import json
import datetime
from ultralytics import YOLO

# במקום yolov8n.pt, פשוט תכתוב:
model = YOLO('yolo11n.pt')

# 2. מקור הוידאו - יכול להיות קובץ או מצלמת רשת (0)
video_source = 0  # שנה ל-'battlefield_simulation.mp4' אם יש לך סרטון
cap = cv2.VideoCapture(video_source)

# רשימת מחלקות שמעניינות אותנו (לפי COCO dataset)
INTERESTING_CLASSES = [0, 24, 26, 67] # 0=person, 24=backpack, 26=handbag, etc.

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ביצוע זיהוי
    results = model(frame, verbose=False)
    
    detected_objects = []
    
    # עיבוד התוצאות
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # סינון: רק אם זה אובייקט חשוד ובביטחון גבוה מ-0.5
            if cls_id in INTERESTING_CLASSES and conf > 0.5:
                label = model.names[cls_id]
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                
                detected_objects.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [int(c) for c in coords]
                })

    # אם זוהה משהו חשוד - יצירת ההתראה (הדמיית שליחה לענן)
    if len(detected_objects) > 0:
        alert_json = {
            "drone_id": "simulated_drone_01",
            "timestamp": datetime.datetime.now().isoformat(),
            "location": {"lat": 32.0853, "lon": 34.7818}, # סתם מיקום לתל אביב כרגע
            "objects": detected_objects
        }
        
        # כרגע רק נדפיס, בשבוע הבא נשלח לשרת אמיתי
        print("🚨 ALERT SENT TO CLOUD:")
        print(json.dumps(alert_json, indent=2))
        
        # אופציונלי: ציור הריבועים על המסך לצורך דיבוג
        annotated_frame = results[0].plot()
        cv2.imshow("Drone View Simulation", annotated_frame)

    # יציאה בלחיצה על 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()