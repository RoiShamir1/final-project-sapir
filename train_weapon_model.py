from ultralytics import YOLO
import torch
from roboflow import Roboflow

rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ")
project = rf.workspace("wpns").project("weapons-s4k8n")
version = project.version(1)
dataset = version.download("yolov8")


data_path = f"{dataset.location}/data.yaml"

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
    
    
    
    
    