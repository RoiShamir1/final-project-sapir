import cv2
import datetime
import json
import numpy as np
from detection_engine import DetectionEngine
from models import Event, ObjectDetection
from db_connector import MongoDBClient
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    print("❌ ERROR: MONGO_URI not found in .env file!")
    exit()

# ─── הגדרות ───────────────────────────────────────────────────────────────────
WEAPON_MODEL_PATH = 'runs/detect/drone_v3/weights/best.pt'

# ⚠️  הרץ ipconfig ב-cmd → IPv4 של ה-Wi-Fi adapter
# הרחפן ישדר ל: rtmp://<IP_שלך>/live/drone
RTMP_URL = "rtmp://10.186.183.143/live/argus"   # ← שנה ל-IP האמיתי שלך

DRONE_ID = "Alpha_01"
CURRENT_LOCATION = {"lat": 32.0853, "lon": 34.7818, "alt": 50}
LOG_FILE = "events_log.json"

CONF_THRESHOLD_WEAPON = 0.5
REQUIRED_CONSECUTIVE_FRAMES = 5
CROUCHING_ASPECT_RATIO_THRESHOLD = 1.2
REQUIRED_CROUCHING_FRAMES = 20


def save_event_to_file(event: Event):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
        print(f"💾 Event saved: {event.event_type}")
    except Exception as e:
        print(f"❌ Failed to save event: {e}")


def detect_suspicious_behavior(detections, frame_shape):
    suspicious_people = []
    frame_h, frame_w, _ = frame_shape

    for obj in detections:
        if obj.label == "person":
            x1, y1, x2, y2 = obj.bbox
            if y2 > frame_h - 10:
                continue
            height = y2 - y1
            width = x2 - x1
            if width == 0:
                continue
            ratio = height / width
            if ratio < CROUCHING_ASPECT_RATIO_THRESHOLD:
                suspicious_people.append(ObjectDetection(
                    label="Suspect (Crouching)",
                    confidence=obj.confidence,
                    bbox=obj.bbox,
                    type="behavior_threat"
                ))
    return suspicious_people


def draw_dashboard(frame, last_event, fps, behavior_counter):
    h, w, _ = frame.shape
    panel_width = 400
    dashboard = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    dashboard[0:h, 0:w] = frame
    dashboard[0:h, w:w + panel_width] = (50, 50, 50)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(dashboard, "SYSTEM STATUS", (w + 20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(dashboard, f"FPS: {fps:.1f}", (w + 20, 80), font, 0.6, (200, 200, 200), 1)
    cv2.putText(dashboard, "Engine: YOLO v3", (w + 20, 100), font, 0.5, (0, 200, 255), 1)

    if behavior_counter > 0:
        bar_len = int((behavior_counter / REQUIRED_CROUCHING_FRAMES) * 200)
        cv2.putText(dashboard, "Analyzing Behavior...", (w + 20, 120), font, 0.5, (0, 255, 255), 1)
        cv2.rectangle(dashboard, (w + 20, 130), (w + 20 + bar_len, 140), (0, 255, 255), -1)

    cv2.line(dashboard, (w + 20, 150), (w + 380, 150), (200, 200, 200), 1)

    if last_event:
        cv2.circle(dashboard, (w + 40, 190), 15, (0, 0, 255), -1)
        cv2.circle(dashboard, (w + 40, 190), 17, (255, 255, 255), 2)
        cv2.putText(dashboard, "ALERT ACTIVE", (w + 70, 200), font, 0.8, (0, 0, 255), 2)

        y_pos = 240
        line_gap = 30
        t_str = last_event.timestamp.split("T")[1].split(".")[0]
        cv2.putText(dashboard, f"Time: {t_str}", (w + 20, y_pos), font, 0.6, (255, 255, 255), 1)
        y_pos += line_gap
        cv2.putText(dashboard, "Detections:", (w + 20, y_pos), font, 0.7, (0, 255, 255), 1)
        y_pos += line_gap
        for det in last_event.detections:
            label_color = (0, 0, 255) if det.type == "threat" else (255, 0, 255)
            cv2.putText(dashboard, f"> {det.label} ({det.confidence:.0%})",
                        (w + 20, y_pos), font, 0.6, label_color, 1)
            y_pos += line_gap
    else:
        cv2.circle(dashboard, (w + 40, 190), 15, (0, 255, 0), -1)
        cv2.putText(dashboard, "NO THREATS", (w + 70, 200), font, 0.8, (0, 255, 0), 2)

    return dashboard


def draw_detections_on_frame(frame, detections, suspicious_people):
    for obj in detections:
        x1, y1, x2, y2 = obj.bbox
        color = (0, 0, 255) if obj.type == "threat" else (0, 255, 0)
        label = obj.label if obj.type == "threat" else "Person"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for obj in suspicious_people:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, "SUSPICIOUS BEHAVIOR", (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


def read_latest_frame(cap):
    """משליך פריימים ישנים שנצברו בבאפר — לוקח רק את הכי עדכני"""
    cap.grab()
    cap.grab()
    cap.grab()
    success, frame = cap.retrieve()
    return success, frame


def open_stream():
    """פותח את ה-RTMP stream עם הגדרות latency מינימלי"""
    print(f"📡 Connecting to RTMP stream: {RTMP_URL}")

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "fflags;nobuffer|"
        "flags;low_delay|"
        "framedrop;1|"
        "bufsize;0"
    )

    cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ Could not open RTMP stream.")
        print("   בדוק:")
        print("   1. mediamtx.exe רץ?")
        print("   2. הרחפן מחובר ומשדר?")
        print(f"   3. ה-IP בהגדרות הרחפן תואם: {RTMP_URL}")
        return None

    print("✅ RTMP stream connected!")
    return cap


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

    cloud_db = MongoDBClient(MONGO_URI)
    cloud_db.connect()

    cap = open_stream()
    if cap is None:
        return

    last_report_time = datetime.datetime.now()
    consecutive_weapon_frames = 0
    consecutive_behavior_frames = 0
    latest_event_display = None
    prev_frame_time = datetime.datetime.now().timestamp()

    print(f"🎥 Surveillance Active. Logging to: {LOG_FILE}")

    while True:
        success, frame = read_latest_frame(cap)

        if not success:
            print("⚠️  Stream lost. Reconnecting in 3 seconds...")
            cap.release()
            cv2.waitKey(3000)
            cap = open_stream()
            if cap is None:
                break
            continue

        new_frame_time = datetime.datetime.now().timestamp()
        fps = 1 / max(new_frame_time - prev_frame_time, 1e-6)
        prev_frame_time = new_frame_time

        detections = engine.detect(frame)
        suspicious_behavior_list = detect_suspicious_behavior(detections, frame.shape)

        weapons_in_frame = [d for d in detections if d.type == "threat"]
        consecutive_weapon_frames = consecutive_weapon_frames + 1 if weapons_in_frame else 0
        consecutive_behavior_frames = consecutive_behavior_frames + 1 if suspicious_behavior_list else 0

        current_time = datetime.datetime.now()
        active_threats = []
        event_reason = ""

        if consecutive_weapon_frames >= REQUIRED_CONSECUTIVE_FRAMES:
            active_threats.extend(weapons_in_frame)
            event_reason = "weapon_detected"

        if consecutive_behavior_frames >= REQUIRED_CROUCHING_FRAMES:
            active_threats.extend(suspicious_behavior_list)
            event_reason = (event_reason + "+suspicious_behavior") if event_reason else "suspicious_behavior"

        if active_threats and (current_time - last_report_time).total_seconds() > 1.0:
            event = Event(
                drone_id=DRONE_ID,
                timestamp=current_time.isoformat(),
                location=CURRENT_LOCATION,
                detections=active_threats,
                event_type="alert"
            )
            save_event_to_file(event)
            cloud_db.insert_event(event.to_dict())
            latest_event_display = event
            last_report_time = current_time

        draw_detections_on_frame(frame, detections, suspicious_behavior_list)
        final_display = draw_dashboard(frame, latest_event_display, fps, consecutive_behavior_frames)
        cv2.imshow("Drone Commander Dashboard", final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()