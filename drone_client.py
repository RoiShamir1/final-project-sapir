import cv2
import datetime
import numpy as np
import threading
import time
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

# WEAPON_MODEL_PATH = 'runs/detect/drone_v4/weights/best.pt'
WEAPON_MODEL_PATH = 'runs/detect/drone_v5/weights/best.pt'
RTMP_URL = "rtmp://10.208.8.143/live/argus"   # IP

DRONE_ID = "Alpha_01"
CURRENT_LOCATION = {"lat": 32.0853, "lon": 34.7818, "alt": 50}
LOG_FILE = "events_log.json"

CONF_THRESHOLD_WEAPON = 0.5
CONF_THRESHOLD_PERSON = 0.5

DETECTION_WINDOW_SIZE = 8
DETECTION_WINDOW_HITS = 4

FRAMES_TO_CLEAR_ALERT = 15


def get_screen_size():
    
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


class StreamReader:
    
    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.success = False
        self.running = False
        self._lock = threading.Lock()
        self._cap = None

    def _open(self):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "fflags;nobuffer|flags;low_delay|framedrop;1|bufsize;0"
        )
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap if cap.isOpened() else None

    def start(self) -> bool:
        print(f"📡 Connecting to RTMP stream: {self.url}")
        self._cap = self._open()
        if self._cap is None:
            print("❌ Could not open RTMP stream.")
            return False
        print("✅ RTMP stream connected!")
        self.running = True
        threading.Thread(target=self._reader_loop, daemon=True).start()
        return True

    def _reader_loop(self):
        
        while self.running:
            if self._cap is None or not self._cap.isOpened():
                print("⚠️  Stream lost. Reconnecting...")
                time.sleep(2)
                self._cap = self._open()
                continue
            success, frame = self._cap.read()
            with self._lock:
                self.success = success
                self.frame = frame

    def read(self):
        
        with self._lock:
            return self.success, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self._cap:
            self._cap.release()


def save_event_to_file(event: Event):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
        print(f"💾 Event saved: {event.event_type}")
    except Exception as e:
        print(f"❌ Failed to save event: {e}")


def _text(img, txt, org, scale, color, thickness=1):
    
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_dashboard(frame, last_event, fps, person_count, weapon_count):
    h, w, _ = frame.shape
    panel_width = 400
    x = w + 25  
    dashboard = np.zeros((h, w + panel_width, 3), dtype=np.uint8)
    dashboard[0:h, 0:w] = frame
    dashboard[0:h, w:w + panel_width] = (38, 38, 38)

    
    _text(dashboard, "SYSTEM STATUS", (x, 50), 0.85, (255, 255, 255), 2)
    cv2.line(dashboard, (x, 65), (w + panel_width - 25, 65), (90, 90, 90), 1)

    
    _text(dashboard, f"FPS: {fps:.1f}", (x, 100), 0.6, (210, 210, 210), 1)
    _text(dashboard, "Model: V4 + YOLO26 Base", (x, 128), 0.55, (0, 200, 255), 1)

    _text(dashboard, f"Persons: {person_count}", (x, 165), 0.7, (0, 255, 0), 2)
    weapon_color = (0, 0, 255) if weapon_count > 0 else (180, 180, 180)
    _text(dashboard, f"Weapons: {weapon_count}", (x, 198), 0.7, weapon_color, 2)

    cv2.line(dashboard, (x, 220), (w + panel_width - 25, 220), (90, 90, 90), 1)

    
    if last_event:
        cv2.circle(dashboard, (x + 15, 262), 15, (0, 0, 255), -1)
        cv2.circle(dashboard, (x + 15, 262), 17, (255, 255, 255), 2)
        _text(dashboard, "WEAPON ALERT", (x + 45, 270), 0.85, (0, 0, 255), 2)

        y = 312
        t_str = last_event.timestamp.split("T")[1].split(".")[0]
        _text(dashboard, f"Time: {t_str}", (x, y), 0.65, (255, 255, 255), 1)
        y += 38
        _text(dashboard, "Detected weapons:", (x, y), 0.65, (0, 140, 255), 2)
        y += 38
        for det in last_event.detections:
            _text(dashboard, f"> {det.label}  ({det.confidence:.0%})",
                  (x + 10, y), 0.65, (0, 80, 255), 2)
            y += 34
    else:
        cv2.circle(dashboard, (x + 15, 262), 15, (0, 220, 0), -1)
        cv2.circle(dashboard, (x + 15, 262), 17, (255, 255, 255), 2)
        _text(dashboard, "NO THREATS", (x + 45, 270), 0.85, (0, 220, 0), 2)

    return dashboard


def draw_detections_on_frame(frame, weapons, persons):
    for obj in persons:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _text(frame, f"Person {obj.confidence:.0%}",
              (x1, y1 - 10), 0.6, (0, 255, 0), 2)
    for obj in weapons:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        _text(frame, f"{obj.label} {obj.confidence:.0%}",
              (x1, y1 - 10), 0.6, (0, 0, 255), 2)


def main():
    try:
        engine = DetectionEngine(
            weapon_model_path=WEAPON_MODEL_PATH,
            conf_weapon=CONF_THRESHOLD_WEAPON,
            conf_person=CONF_THRESHOLD_PERSON
        )
    except Exception:
        print("❌ Failed to start engine.")
        return

    cloud_db = MongoDBClient(MONGO_URI)
    cloud_db.connect()

    
    stream = StreamReader(RTMP_URL)
    if not stream.start():
        print("❌ Cannot start stream.")
        return

    
    time.sleep(1.0)

    from collections import deque
    weapon_history = deque(maxlen=DETECTION_WINDOW_SIZE) 
    no_weapon_frames = 0
    alert_active = False          
    latest_weapons = []           
    latest_event_display = None
    prev_frame_time = time.time()

    screen_w, screen_h = get_screen_size()
    window_name = "Drone Commander Dashboard"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    window_initialized = False

    print(f"🎥 Surveillance Active. Logging to: {LOG_FILE}")

    while True:
        success, frame = stream.read()

        if not success or frame is None:
            time.sleep(0.05)
            continue

        # FPS
        now = time.time()
        fps = 1 / max(now - prev_frame_time, 1e-6)
        prev_frame_time = now

        
        weapons, persons = engine.detect(frame)

        weapon_history.append(bool(weapons))
        if weapons:
            latest_weapons = weapons
            no_weapon_frames = 0
        else:
            no_weapon_frames += 1
            
            if no_weapon_frames >= FRAMES_TO_CLEAR_ALERT:
                alert_active = False

        hits_in_window = sum(weapon_history)

        if hits_in_window >= DETECTION_WINDOW_HITS and not alert_active:
            event = Event(
                drone_id=DRONE_ID,
                timestamp=datetime.datetime.now().isoformat(),
                location=CURRENT_LOCATION,
                detections=latest_weapons,
                event_type="alert"
            )
            save_event_to_file(event)
            cloud_db.insert_event(event.to_dict())
            latest_event_display = event
            alert_active = True

        
        draw_detections_on_frame(frame, weapons, persons)
        final_display = draw_dashboard(frame, latest_event_display, fps,
                                       len(persons), len(weapons))

        if not window_initialized:
            dh, dw = final_display.shape[:2]
            margin = 0.9  
            scale = min(screen_w * margin / dw, screen_h * margin / dh, 1.0)
            cv2.resizeWindow(window_name, int(dw * scale), int(dh * scale))
            window_initialized = True

        cv2.imshow(window_name, final_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()