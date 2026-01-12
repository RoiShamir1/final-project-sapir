import cv2
import datetime
import torch
from ultralytics import YOLO

# --- הגדרות ---
# טוען שני מודלים במקביל!
# 1. המודל החדש שאימנת (נשקים)
path_custom = 'runs/detect/drone_final_v1/weights/best.pt' 
# 2. המודל הכללי (אנשים)
path_general = 'yolo11n.pt'                                  

# ספי ביטחון
CONF_WEAPON = 0.4  # לנשקים (נהיה רגישים)
CONF_PERSON = 0.5  # לאנשים (נדרוש וודאות)

# בדיקת חומרה
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"🚀 Running dual-inference on: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

# אתחול המודלים
print("⏳ Loading Models... (This might take a moment)")
try:
    model_weapons = YOLO(path_custom) # המומחה לנשק
    model_general = YOLO(path_general) # המומחה לאנשים
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print(f"Verify that this file exists: {path_custom}")
    exit()

cap = cv2.VideoCapture(0) # למצלמה

# רשימת צבעים לזיהוי מהיר
COLOR_THREAT = (0, 0, 255)   # אדום
COLOR_SAFE = (0, 255, 0)     # ירוק
COLOR_WARN = (0, 165, 255)   # כתום

print("🎥 Starting Surveillance System... Press 'q' to exit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    detected_objects = []

    # --- שלב 1: זיהוי נשקים (המודל שלך) ---
    results_weapons = model_weapons(frame, device=device, verbose=False, conf=CONF_WEAPON)
    
    for r in results_weapons:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model_weapons.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            # הוספה לרשימה כ"איום"
            detected_objects.append({
                "label": label, 
                "conf": conf, 
                "bbox": [x1, y1, x2, y2],
                "type": "threat" 
            })

    # --- שלב 2: זיהוי אנשים (המודל הכללי) ---
    # classes=[0] -> מחפש רק Person
    results_people = model_general(frame, device=device, verbose=False, conf=CONF_PERSON, classes=[0])

    for r in results_people:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # הוספה לרשימה כ"אזרח"
            detected_objects.append({
                "label": "person", 
                "conf": float(box.conf[0]), 
                "bbox": [x1, y1, x2, y2],
                "type": "civilian" 
            })

    # --- שלב 3: ציור על המסך ---
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        conf = obj["conf"]
        
        # בחירת צבע לפי סוג האיום
        color = COLOR_THREAT if obj["type"] == "threat" else COLOR_SAFE
        
        # מסגרת
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # כיתוב רקע (כדי שיהיה קריא)
        label_text = f"{label} {conf:.0%}"
        t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + t_size[0], y1), color, -1)
        
        # טקסט
        cv2.putText(frame, label_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # הצגת הוידאו
    cv2.imshow("Sapir College Final Project - Threat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()