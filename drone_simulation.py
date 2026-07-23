import cv2
import json
import datetime
from ultralytics import YOLO

model = YOLO('yolo11n.pt')


video_source = 0 
cap = cv2.VideoCapture(video_source)

INTERESTING_CLASSES = [0, 24, 26, 67] 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, verbose=False)
    
    detected_objects = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            if cls_id in INTERESTING_CLASSES and conf > 0.5:
                label = model.names[cls_id]
                coords = box.xyxy[0].tolist()
                
                detected_objects.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [int(c) for c in coords]
                })

    if len(detected_objects) > 0:
        alert_json = {
            "drone_id": "simulated_drone_01",
            "timestamp": datetime.datetime.now().isoformat(),
            "location": {"lat": 32.0853, "lon": 34.7818},
            "objects": detected_objects
        }
        
        print("🚨 ALERT SENT TO CLOUD:")
        print(json.dumps(alert_json, indent=2))
        
        annotated_frame = results[0].plot()
        cv2.imshow("Drone View Simulation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()