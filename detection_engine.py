import torch
from ultralytics import YOLO
from typing import List, Tuple
from models import ObjectDetection


class DetectionEngine:
    def __init__(self, weapon_model_path: str, conf_weapon: float = 0.4, conf_person: float = 0.5):
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Initializing Engine on: {self.device}")
        self.conf_weapon = conf_weapon
        self.conf_person = conf_person

        print("⏳ Loading Models...")
        try:
            # מודל V4 שלנו — נשקים + אנשים מהאוויר
            self.model_v4 = YOLO(weapon_model_path)
            # מודל בסיס YOLO26 — אנשים כלליים
            self.model_base = YOLO('yolo26n.pt')
            print(f"✅ V4 loaded. Classes: {self.model_v4.names}")
            print("✅ YOLO26 base loaded.")
        except Exception as e:
            print(f"❌ Error: {e}")
            raise e

    def _iou(self, boxA, boxB) -> float:
        """חישוב חפיפה בין שני ריבועים — למניעת כפילויות"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter == 0:
            return 0.0
        areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter)

    def _merge_persons(self, list1: List[ObjectDetection], list2: List[ObjectDetection],
                       iou_thresh: float = 0.5) -> List[ObjectDetection]:
        """
        ממזג שתי רשימות של אנשים.
        אם שני זיהויים חופפים (IOU > סף) — שומר את זה עם הביטחון הגבוה יותר.
        """
        merged = list(list1)
        for candidate in list2:
            duplicate = False
            for existing in merged:
                if self._iou(candidate.bbox, existing.bbox) > iou_thresh:
                    duplicate = True
                    # אם המועמד החדש יותר בטוח — נחליף
                    if candidate.confidence > existing.confidence:
                        existing.confidence = candidate.confidence
                    break
            if not duplicate:
                merged.append(candidate)
        return merged

    def detect(self, frame) -> Tuple[List[ObjectDetection], List[ObjectDetection]]:
        """
        מחזיר (weapons, persons).
        נשקים: מודל V4 בלבד.
        אנשים: איחוד תוצאות V4 + YOLO26 base (ללא כפילויות).
        """
        weapons  = []
        persons_v4   = []
        persons_base = []

        # ── מודל V4: נשקים + אנשים מהאוויר ──────────────────────────────────
        results_v4 = self.model_v4(
            frame,
            device=self.device,
            verbose=False,
            conf=min(self.conf_weapon, self.conf_person),
            agnostic_nms=False,
            iou=0.45
        )
        for r in results_v4:
            for box in r.boxes:
                label      = self.model_v4.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                bbox       = list(map(int, box.xyxy[0]))

                if label.lower() == "person":
                    if confidence >= self.conf_person:
                        persons_v4.append(ObjectDetection(
                            label="person", confidence=confidence,
                            bbox=bbox, type="civilian"))
                else:
                    if confidence >= self.conf_weapon:
                        weapons.append(ObjectDetection(
                            label=label, confidence=confidence,
                            bbox=bbox, type="threat"))

        # ── מודל בסיס YOLO26: אנשים כלליים (class 0 בלבד) ───────────────────
        results_base = self.model_base(
            frame,
            device=self.device,
            verbose=False,
            conf=self.conf_person,
            classes=[0]
        )
        for r in results_base:
            for box in r.boxes:
                confidence = float(box.conf[0])
                bbox       = list(map(int, box.xyxy[0]))
                persons_base.append(ObjectDetection(
                    label="person", confidence=confidence,
                    bbox=bbox, type="civilian"))

        # ── מיזוג אנשים ───────────────────────────────────────────────────────
        persons = self._merge_persons(persons_v4, persons_base)

        return weapons, persons