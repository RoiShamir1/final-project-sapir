from ultralytics import YOLO

def train():
    # 1. טען את המודל מהנקודה האחרונה
    model = YOLO('runs/detect/drone_weapon_model/weights/last.pt')

    # 2. פקודת ההמשך
    results = model.train(resume=True)

if __name__ == '__main__':
    train()