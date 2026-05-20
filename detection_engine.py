import torch
from ultralytics import YOLO
from typing import List
from models import ObjectDetection


class DetectionEngine:
    def __init__(self, weapon_model_path: str, conf_weapon: float = 0.4, conf_person: float = 0.5):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Engine on: {self.device}")
        self.conf_weapon = conf_weapon
        self.conf_person = conf_person

        print("⏳ Loading Models...")
        try:
            self.model_weapons = YOLO(weapon_model_path)
            self.model_general = YOLO('yolo11n.pt')
            print("✅ Models Loaded!")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise e

    def detect(self, frame) -> List[ObjectDetection]:
        detections = []

        # זיהוי נשקים עם המודל המאומן שלנו
        results_weapons = self.model_weapons(
            frame,
            device=self.device,
            verbose=False,
            conf=self.conf_weapon,
            agnostic_nms=False,
            iou=0.45
        )
        for r in results_weapons:
            for box in r.boxes:
                detections.append(ObjectDetection(
                    label=self.model_weapons.names[int(box.cls[0])],
                    confidence=float(box.conf[0]),
                    bbox=list(map(int, box.xyxy[0])),
                    type="threat"
                ))

        # זיהוי אנשים עם מודל COCO כללי
        results_people = self.model_general(
            frame,
            device=self.device,
            verbose=False,
            conf=self.conf_person,
            classes=[0]
        )
        for r in results_people:
            for box in r.boxes:
                detections.append(ObjectDetection(
                    label="person",
                    confidence=float(box.conf[0]),
                    bbox=list(map(int, box.xyxy[0])),
                    type="civilian"
                ))

        return detections