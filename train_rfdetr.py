"""
train_rfdetr.py
---------------
אימון מודל RF-DETR על דאטה-סט הנשקים + ההתנהגות שלנו.

שינויים עיקריים לעומת train_weapon_again.py (YOLO):
  - הורדת הדאטה-סט בפורמט COCO (לא yolov11)
  - שימוש ב-rfdetr במקום ultralytics
  - המשקלים נשמרים בנתיב שונה
"""

from roboflow import Roboflow
from rfdetr import RFDETRBase          # אפשר גם RFDETRMedium / RFDETRSmall
import os

# ─── 1. הורדת הדאטה-סט מ-Roboflow ─────────────────────────────────────────
# ⚠️  שים לב: הפורמט הוא "coco" ולא "yolov11" כמו פעם!
rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ")
project = rf.workspace("droneprojectsapir").project("drone-project-wvj7h")
version = project.version(1)
dataset = version.download("coco")          # <--- פורמט COCO JSON

DATASET_DIR = dataset.location             # נתיב לתיקייה שמכילה train/ valid/ test/
print(f"📂 Dataset location: {DATASET_DIR}")

# ─── 2. בחירת גודל המודל ───────────────────────────────────────────────────
# RF-DETR-L מומלץ לפרויקט: דיוק גבוה, עדיין real-time ב-RTX 3050
# אם הזיכרון קצר — החלף ל-RFDETRMedium

def train():
    model = RFDETRBase()

# ─── 3. אימון ──────────────────────────────────────────────────────────────
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=50,
        batch_size=8,               # RTX 3050 (4GB) - אם נגמר זיכרון: batch_size=4
        grad_accum_steps=2,         # מדמה batch_size=16 בלי לטעון יותר לזיכרון
        lr=1e-4,
        output_dir="runs/rfdetr/drone_v1",
        checkpoint_interval=5,    # שמירה כל 5 epochs (אופציונלי)
    )


print("✅ Training complete!")
print("   Weights saved to: runs/rfdetr/drone_v1/")
print("   Use 'checkpoint_best_total.pth' in detection_engine.py")

if __name__ == '__main__':
    train()