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
from rfdetr import RFDETRBase         
import os

rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ")
project = rf.workspace("droneprojectsapir").project("drone-project-wvj7h")
version = project.version(1)
dataset = version.download("coco")          

DATASET_DIR = dataset.location             
print(f"📂 Dataset location: {DATASET_DIR}")



def train():
    model = RFDETRBase()


    model.train(
        dataset_dir=DATASET_DIR,
        epochs=50,
        batch_size=8,               
        grad_accum_steps=2,         
        lr=1e-4,
        output_dir="runs/rfdetr/drone_v1",
        checkpoint_interval=5,   
    )


print("✅ Training complete!")
print("   Weights saved to: runs/rfdetr/drone_v1/")
print("   Use 'checkpoint_best_total.pth' in detection_engine.py")

if __name__ == '__main__':
    train()