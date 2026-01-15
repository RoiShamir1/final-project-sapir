import cv2
import datetime
import json
import numpy as np
from detection_engine import DetectionEngine
from models import Event

# --- הגדרות ---
WEAPON_MODEL_PATH = 'runs/detect/drone_final_v1/weights/best.pt'
DRONE_ID = "Alpha_01"
CURRENT_LOCATION = {"lat": 32.0853, "lon": 34.7818, "alt": 50}
LOG_FILE = "events_log.json" # שם הקובץ שיישמר

# הגדרות רגישות
CONF_THRESHOLD_WEAPON = 0.6
REQUIRED_CONSECUTIVE_FRAMES = 5

def save_event_to_file(event: Event):
    """שומר את האירוע לקובץ JSON (מצב Append)"""
    try:
        # אנו שומרים כל אירוע כשורה נפרדת (JSON Lines) כדי לא לשבור את הקובץ בקריסה
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
        print(f"💾 Event saved to {LOG_FILE}")
    except Exception as e:
        print(f"❌ Failed to save event: {e}")

def draw_dashboard(frame, last_event, fps):
    """
    גרסה מתוקנת: ללא אימוג'ים כדי למנוע סימני שאלה, ועם עיגול התראה
    """
    h, w, _ = frame.shape
    panel_width = 400
    dashboard = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    
    # הדבקת הוידאו
    dashboard[0:h, 0:w] = frame
    # פאנל ימני
    dashboard[0:h, w:w+panel_width] = (50, 50, 50) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # כותרות
    cv2.putText(dashboard, "SYSTEM STATUS", (w + 20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (w + 20, 80), font, 0.6, (200, 200, 200), 1)
    cv2.line(dashboard, (w + 20, 95), (w + 380, 95), (200, 200, 200), 1)

    if last_event:
        # --- תיקון: במקום אימוג'י, נצייר עיגול אדום ---
        # ציור עיגול אדום מלא (נורה דולקת)
        cv2.circle(dashboard, (w + 40, 140), 15, (0, 0, 255), -1)
        # ציור מסגרת לעיגול
        cv2.circle(dashboard, (w + 40, 140), 17, (255, 255, 255), 2)
        
        # הטקסט ליד הנורה
        cv2.putText(dashboard, "ALERT ACTIVE", (w + 70, 150), font, 0.8, (0, 0, 255), 2)
        
        # פרטי האירוע
        y_pos = 190
        line_gap = 30
        
        # זמן (רק שעות ודקות)
        t_str = last_event.timestamp.split("T")[1].split(".")[0]
        cv2.putText(dashboard, f"Time: {t_str}", (w + 20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += line_gap
        
        cv2.putText(dashboard, f"Lat: {last_event.location['lat']}", (w + 20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += line_gap
        cv2.putText(dashboard, f"Lon: {last_event.location['lon']}", (w + 20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += line_gap * 2
        
        cv2.putText(dashboard, "Detections:", (w + 20, y_pos), font, 0.7, (0, 255, 255), 1)
        y_pos += line_gap
        
        for det in last_event.detections:
            # שינוי צבע לפי ביטחון (אדום חזק = ביטחון גבוה)
            conf_color = (0, 165, 255) # כתום
            if det.confidence > 0.8:
                conf_color = (0, 0, 255) # אדום
                
            det_text = f"> {det.label} ({det.confidence:.0%})"
            cv2.putText(dashboard, det_text, (w + 20, y_pos), font, 0.6, conf_color, 1)
            y_pos += line_gap

    else:
        # מצב שגרה - עיגול ירוק
        cv2.circle(dashboard, (w + 40, 140), 15, (0, 255, 0), -1)
        cv2.putText(dashboard, "NO THREATS", (w + 70, 150), font, 0.8, (0, 255, 0), 2)
        cv2.putText(dashboard, "Scanning...", (w + 20, 190), font, 0.6, (200, 200, 200), 1)

    return dashboard

def draw_detections_on_frame(frame, detections):
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox
        color = (0, 0, 255) if obj.type == "threat" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{obj.label}"
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    try:
        engine = DetectionEngine(
            weapon_model_path=WEAPON_MODEL_PATH,
            conf_weapon=CONF_THRESHOLD_WEAPON,
            conf_person=0.5
        )
    except Exception:
        print("❌ Failed to start engine.")
        return

    cap = cv2.VideoCapture(0)
    
    last_report_time = datetime.datetime.now()
    consecutive_threat_frames = 0
    
    # משתנה לשמירת האירוע האחרון לתצוגה בדשבורד
    latest_event_display = None 
    
    # לחישוב FPS
    prev_frame_time = 0
    new_frame_time = 0

    print(f"🎥 Dashboard Active. Logging to: {LOG_FILE}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # חישוב FPS
        new_frame_time = datetime.datetime.now().timestamp()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # 1. זיהוי
        detections = engine.detect(frame)

        # 2. לוגיקת יציבות
        threats_in_frame = [d for d in detections if d.type == "threat"]
        if threats_in_frame:
            consecutive_threat_frames += 1
        else:
            consecutive_threat_frames = 0
            # אופציונלי: אחרי כמה זמן לנקות את התצוגה?
            # כרגע נשאיר את ההתראה האחרונה על המסך שתהיה בולטת

        # 3. יצירת אירוע ושמירה
        current_time = datetime.datetime.now()
        if consecutive_threat_frames >= REQUIRED_CONSECUTIVE_FRAMES:
            if (current_time - last_report_time).total_seconds() > 1.0:
                
                # יצירת האירוע
                event = Event(
                    drone_id=DRONE_ID,
                    timestamp=current_time.isoformat(),
                    location=CURRENT_LOCATION,
                    detections=threats_in_frame,
                    event_type="alert"
                )
                
                # --- שמירה לקובץ ---
                save_event_to_file(event)
                
                # --- עדכון התצוגה ---
                latest_event_display = event
                last_report_time = current_time

        # 4. ציור על הוידאו המקורי
        draw_detections_on_frame(frame, detections)
        
        # 5. יצירת הדשבורד (חיבור וידאו + טקסט)
        final_display = draw_dashboard(frame, latest_event_display, fps)

        # הצגה
        cv2.imshow("Drone Commander Dashboard", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()