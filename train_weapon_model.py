from ultralytics import YOLO
import torch
from roboflow import Roboflow

rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ")
project = rf.workspace("wpns").project("weapons-s4k8n")
version = project.version(1)
dataset = version.download("yolov8")

# שים לב: המשתנה 'dataset' מחזיק עכשיו את המיקום של הקבצים במחשב שלך
# אם הורדת ידנית כקובץ ZIP וחילצת, פשוט תכתוב את הנתיב לקובץ data.yaml ידנית למטה
# דוגמה: data_path = "C:/Users/Roi/Datasets/Weapons/data.yaml"
data_path = f"{dataset.location}/data.yaml" # האם אני צריך לשנות את זה?

def train_model():
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Starting training on: {torch.cuda.get_device_name(0)}")

    model = YOLO('yolo11n.pt')  

    results = model.train(
        data=data_path,
        epochs=30,
        imgsz=640,
        device=device,
        batch=8,
        name='drone_weapon_model'
    )
    
    print("Training Complete!")

if __name__ == '__main__':
    train_model()
    
    
    
    
    