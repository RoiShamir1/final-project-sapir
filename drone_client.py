import cv2
import datetime
import json
import numpy as np
from detection_engine import DetectionEngine
from models import Event, ObjectDetection
from db_connector import MongoDBClient
import os
from dotenv import load_dotenv  # ייבוא הפונקציה

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("❌ ERROR: MONGO_URI not found in .env file!")
    exit() # עוצר את התוכנית כי אין טעם להמשיך בלי דאטה-בייס

# --- הגדרות ---
WEAPON_MODEL_PATH = 'runs/detect/drone_final_v1/weights/best.pt'
DRONE_ID = "Alpha_01"
CURRENT_LOCATION = {"lat": 32.0853, "lon": 34.7818, "alt": 50}
LOG_FILE = "events_log.json"

# הגדרות רגישות לזיהוי נשק
CONF_THRESHOLD_WEAPON = 0.6
REQUIRED_CONSECUTIVE_FRAMES = 5

# --- הגדרות חדשות: התנהגות חשודה ---
# יחס גובה/רוחב: מתחת לזה נחשוד שאדם מתכופף
# (1.0 אומר ריבוע, פחות מזה אומר מלבן שוכב)
CROUCHING_ASPECT_RATIO_THRESHOLD = 1.2 

# כמה זמן (פריימים) אדם צריך להיות מכופף כדי להקפיץ התראה?
# נניח 30 פריימים = בערך שתי שניות של שהייה במקום
REQUIRED_CROUCHING_FRAMES = 20 

def save_event_to_file(event: Event):
    """שומר את האירוע לקובץ JSON בשורה אחת"""
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
        print(f"💾 Event saved: {event.event_type}")
    except Exception as e:
        print(f"❌ Failed to save event: {e}")

def detect_suspicious_behavior(detections, frame_shape):
    """
    גרסה מתוקנת: מתעלמת מאנשים שחתוכים ע"י גבולות המסך
    """
    suspicious_people = []
    frame_h, frame_w, _ = frame_shape # גובה ורוחב התמונה
    
    for obj in detections:
        if obj.label == "person":
            x1, y1, x2, y2 = obj.bbox
            
            # --- תיקון: בדיקת קצוות ---
            # אם הריבוע נוגע בתחתית המסך (עם מרווח ביטחון של 10 פיקסלים)
            # זה אומר שאנחנו לא רואים את כל הגוף -> מדלגים על הבדיקה
            if y2 > frame_h - 10:
                continue 

            # חישוב גובה ורוחב
            height = y2 - y1
            width = x2 - x1
            
            if width == 0: continue
            
            ratio = height / width
            
            # הלוגיקה הרגילה
            if ratio < CROUCHING_ASPECT_RATIO_THRESHOLD:
                suspicious_obj = ObjectDetection(
                    label="Suspect (Crouching)",
                    confidence=obj.confidence,
                    bbox=obj.bbox,
                    type="behavior_threat"
                )
                suspicious_people.append(suspicious_obj)
                
    return suspicious_people

def draw_dashboard(frame, last_event, fps, behavior_counter):
    h, w, _ = frame.shape
    panel_width = 400
    dashboard = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    
    dashboard[0:h, 0:w] = frame
    dashboard[0:h, w:w+panel_width] = (50, 50, 50) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # סטטוס מערכת
    cv2.putText(dashboard, "SYSTEM STATUS", (w + 20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (w + 20, 80), font, 0.6, (200, 200, 200), 1)
    
    # חיווי על מונה התנהגות (כדי שתראה את המערכת "חושבת")
    if behavior_counter > 0:
        bar_len = int((behavior_counter / REQUIRED_CROUCHING_FRAMES) * 200)
        cv2.putText(dashboard, "Analyzing Behavior...", (w + 20, 110), font, 0.5, (0, 255, 255), 1)
        cv2.rectangle(dashboard, (w + 20, 120), (w + 20 + bar_len, 130), (0, 255, 255), -1)

    cv2.line(dashboard, (w + 20, 140), (w + 380, 140), (200, 200, 200), 1)

    if last_event:
        # נורה אדומה
        cv2.circle(dashboard, (w + 40, 180), 15, (0, 0, 255), -1)
        cv2.circle(dashboard, (w + 40, 180), 17, (255, 255, 255), 2)
        
        cv2.putText(dashboard, "ALERT ACTIVE", (w + 70, 190), font, 0.8, (0, 0, 255), 2)
        
        y_pos = 230
        line_gap = 30
        
        t_str = last_event.timestamp.split("T")[1].split(".")[0]
        cv2.putText(dashboard, f"Time: {t_str}", (w + 20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += line_gap
        
        cv2.putText(dashboard, "Detections:", (w + 20, y_pos), font, 0.7, (0, 255, 255), 1)
        y_pos += line_gap
        
        for det in last_event.detections:
            # צבע שונה לנשק ולחריגת התנהגות
            label_color = (0, 0, 255) # אדום לנשק
            if det.type == "behavior_threat":
                label_color = (255, 0, 255) # סגול להתנהגות
            
            det_text = f"> {det.label} ({det.confidence:.0%})"
            cv2.putText(dashboard, det_text, (w + 20, y_pos), font, 0.6, label_color, 1)
            y_pos += line_gap
    else:
        cv2.circle(dashboard, (w + 40, 180), 15, (0, 255, 0), -1)
        cv2.putText(dashboard, "NO THREATS", (w + 70, 190), font, 0.8, (0, 255, 0), 2)

    return dashboard

def draw_detections_on_frame(frame, detections, suspicious_people):
    # ציור זיהויים רגילים ונשקים
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox
        if obj.type == "threat":
            color = (0, 0, 255) # אדום לנשק
            label = obj.label
        else:
            color = (0, 255, 0) # ירוק לאדם רגיל
            label = "Person"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ציור (דריסה) של אנשים חשודים בצבע סגול
    for obj in suspicious_people:
        x1, y1, x2, y2 = obj.bbox
        color = (255, 0, 255) # סגול - התנהגות חשודה
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, "SUSPICIOUS BEHAVIOR", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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

    # התחברות למסד הנתונים בענן
    cloud_db = MongoDBClient(MONGO_URI)
    cloud_db.connect()
    
    cap = cv2.VideoCapture(0)
    
    last_report_time = datetime.datetime.now()
    
    # מונים נפרדים לנשק ולהתנהגות
    consecutive_weapon_frames = 0
    consecutive_behavior_frames = 0
    
    latest_event_display = None 
    prev_frame_time = 0

    print(f"🎥 Surveillance Active. Logging to: {LOG_FILE}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # FPS
        new_frame_time = datetime.datetime.now().timestamp()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # 1. זיהוי בסיסי (נשקים ואנשים)
        detections = engine.detect(frame)

        # 2. ניתוח התנהגות (על גבי הזיהויים)
        suspicious_behavior_list = detect_suspicious_behavior(detections, frame.shape)

        # 3. לוגיקת יציבות (מונים)
        
        # --- א. בדיקת נשקים ---
        weapons_in_frame = [d for d in detections if d.type == "threat"]
        if weapons_in_frame:
            consecutive_weapon_frames += 1
        else:
            consecutive_weapon_frames = 0

        # --- ב. בדיקת התנהגות ---
        if suspicious_behavior_list:
            consecutive_behavior_frames += 1
        else:
            consecutive_behavior_frames = 0

        # 4. יצירת אירוע (אם אחד מהתנאים התקיים)
        current_time = datetime.datetime.now()
        active_threats = []
        event_reason = ""

        # האם יש נשק יציב?
        if consecutive_weapon_frames >= REQUIRED_CONSECUTIVE_FRAMES:
            active_threats.extend(weapons_in_frame)
            event_reason = "weapon_detected"

        # האם יש התנהגות חשודה יציבה?
        if consecutive_behavior_frames >= REQUIRED_CROUCHING_FRAMES:
            active_threats.extend(suspicious_behavior_list)
            # אם כבר יש נשק, זה מוסיף לסיבה. אם לא, זו הסיבה הראשית
            if event_reason: 
                event_reason += "+suspicious_behavior"
            else:
                event_reason = "suspicious_behavior"

        # אם יש איומים מאושרים ועבר זמן מהדיווח האחרון -> דווח
        if active_threats and (current_time - last_report_time).total_seconds() > 1.0:
            
            event = Event(
                drone_id=DRONE_ID,
                timestamp=current_time.isoformat(),
                location=CURRENT_LOCATION,
                detections=active_threats,
                event_type="alert" # אפשר לשנות ל-event_reason אם רוצים פירוט ב-Type
            )
            
            save_event_to_file(event)
            cloud_db.insert_event(event.to_dict())
            latest_event_display = event
            last_report_time = current_time

        # 5. ציור
        # שולחים את הזיהויים הרגילים ואת החשודים כדי לצייר אותם בצבעים שונים
        draw_detections_on_frame(frame, detections, suspicious_behavior_list)
        
        # 6. דשבורד
        # מעבירים גם את מונה ההתנהגות כדי להציג את הבר "Analyzing Behavior"
        final_display = draw_dashboard(frame, latest_event_display, fps, consecutive_behavior_frames)

        cv2.imshow("Drone Commander Dashboard", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()