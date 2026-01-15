import cv2
import datetime
from detection_engine import DetectionEngine
from models import Event

# --- הגדרות ---
WEAPON_MODEL_PATH = 'runs/detect/drone_final_v1/weights/best.pt'
DRONE_ID = "Alpha_01" 
CURRENT_LOCATION = {"lat": 32.0853, "lon": 34.7818, "alt": 50}

# --- הגדרות סינון רעשים ---
# נעלה את הסף ל-60% כדי להימנע מזיהויים שגויים
CONF_THRESHOLD_WEAPON = 0.6 

# כמה פריימים רצופים צריך לזהות איום לפני שמדווחים?
# זה המסנן הכי חשוב! מונע התראות שווא על "גליצ'ים" של שבריר שנייה
REQUIRED_CONSECUTIVE_FRAMES = 5 

def draw_detections(frame, detections):
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox
        color = (0, 0, 255) if obj.type == "threat" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # מציג את השם והאחוזים על המסך
        label_text = f"{obj.label} {obj.confidence:.0%}"
        cv2.putText(frame, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    # 1. אתחול המנוע עם סף ביטחון גבוה יותר
    try:
        engine = DetectionEngine(
            weapon_model_path=WEAPON_MODEL_PATH,
            conf_weapon=CONF_THRESHOLD_WEAPON, # כאן השינוי ל-0.6
            conf_person=0.5
        )
    except Exception:
        print("❌ Failed to start engine.")
        return

    cap = cv2.VideoCapture(0)
    
    last_report_time = datetime.datetime.now()
    
    # --- משתנה חדש: מונה פריימים רצופים לאיום ---
    consecutive_threat_frames = 0

    print(f"🎥 Drone Active. Waiting for {REQUIRED_CONSECUTIVE_FRAMES} consecutive frames to alert...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. זיהוי
        detections = engine.detect(frame)

        # 2. לוגיקת יציבות (Stability Logic)
        # בודקים אם יש איום בפריים הנוכחי
        threats_in_frame = [d for d in detections if d.type == "threat"]

        if threats_in_frame:
            # אם יש איום, מגדילים את המונה
            consecutive_threat_frames += 1
            # מצייר טקסט על המסך כדי שתראה את הסטטוס
            cv2.putText(frame, f"Verifying Threat: {consecutive_threat_frames}/{REQUIRED_CONSECUTIVE_FRAMES}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            # אם אין איום, מאפסים את המונה מיד!
            consecutive_threat_frames = 0

        # 3. דיווח - רק אם המונה הגיע ליעד (5 פריימים רצופים)
        current_time = datetime.datetime.now()
        
        # התנאי: גם עברנו את מספר הפריימים הרצופים וגם עברה שנייה מהדיווח האחרון
        if consecutive_threat_frames >= REQUIRED_CONSECUTIVE_FRAMES:
            if (current_time - last_report_time).total_seconds() > 1.0:
                
                event = Event(
                    drone_id=DRONE_ID,
                    timestamp=current_time.isoformat(),
                    location=CURRENT_LOCATION,
                    detections=threats_in_frame,
                    event_type="alert"
                )

                print("\n🚨 CONFIRMED EVENT GENERATED:")
                print(event.to_json())
                
                last_report_time = current_time
                # לא מאפסים את המונה כאן, כדי שאם האיום נמשך, ימשיך לדווח כל שנייה

        # 4. ויזואליזציה
        draw_detections(frame, detections)
        cv2.imshow("Drone View - Stable Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()