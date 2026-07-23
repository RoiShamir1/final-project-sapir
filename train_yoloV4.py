from roboflow import Roboflow
from ultralytics import YOLO

if __name__ == '__main__':

    rf = Roboflow(api_key="kM2ZUfe7R97bZAn0ycWZ")
    project = rf.workspace("droneprojectsapir").project("drone-project-wvj7h")
    version = project.version(4)
    dataset = version.download("yolov11")
                

    
    model = YOLO("yolo11n.pt")                

    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=30,
        imgsz=640,
        device=0,                             
        batch=8,
        name="drone_v4",
    )

    print("✅ Training complete!")
    print("   Weights: runs/detect/drone_v4/weights/best.pt")