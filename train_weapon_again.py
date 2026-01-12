from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ") 
project = rf.workspace("droneprojectsapir").project("drone-project-wvj7h")
version = project.version(1)
dataset = version.download("yolov11")

# הגדרת הנתיב לקובץ הנתונים שהורד
data_path = f"{dataset.location}/data.yaml"

def train():
    print(f"📂 Dataset location: {data_path}")
    print("🚀 Starting training on the Unified Drone Dataset...")

    # --- 2. טעינת מודל בסיס ---
    # אנחנו מתחילים מ-yolo11n.pt (נקי) ולא מהמודל הקודם,
    # כי עכשיו יש לנו סט חדש של מחלקות (Classes) וחשוב שהמודל ילמד את המבנה החדש מאפס.
    model = YOLO('yolo11n.pt') 

    # --- 3. הרצת האימון ---
    results = model.train(
        data=data_path,
        epochs=30,        # לאב-טיפוס זה מספיק. לפרויקט הסופי מומלץ 50-100.
        imgsz=640,
        device=0,         # שימוש ב-RTX 3050 שלך
        batch=8,          # שומרים על 8 כדי לא לחרוג מזיכרון ה-GPU
        name='drone_final_v1' # שם התיקייה החדשה שתיווצר ב-runs/detect
    )

if __name__ == '__main__':
    train()