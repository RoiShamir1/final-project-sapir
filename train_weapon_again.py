from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ") 
project = rf.workspace("droneprojectsapir").project("drone-project-wvj7h")
version = project.version(1)
dataset = version.download("yolov11")


data_path = f"{dataset.location}/data.yaml"

def train():
    print(f"📂 Dataset location: {data_path}")
    print("🚀 Starting training on the Unified Drone Dataset...")

  
   
 
    model = YOLO('yolo11n.pt') 

    
    results = model.train(
        data=data_path,
        epochs=30,     
        imgsz=640,
        device=0,        
        batch=8,       
        name='drone_final_v1'
    )

if __name__ == '__main__':
    train()