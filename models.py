from dataclasses import dataclass, asdict, field
from typing import List
import datetime
import json

@dataclass
class ObjectDetection:
    """מייצג אובייקט בודד שזוהה בפריים"""
    label: str
    confidence: float
    bbox: list
    type: str  # 'threat' / 'civilian'

@dataclass
class Event:
    """מייצג אירוע דיווח מלא שנשלח לשרת"""
    drone_id: str
    timestamp: str
    location: dict  # {lat, lon, alt}
    detections: List[ObjectDetection]
    event_type: str = "routine" # routine / alert

    def to_json(self):
        """פונקציית עזר להמרת האירוע ל-JSON תקני"""
        return json.dumps(asdict(self), default=str, indent=4)