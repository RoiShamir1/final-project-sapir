from dataclasses import dataclass, asdict, field
from typing import List
import datetime
import json

@dataclass
class ObjectDetection:
    
    label: str
    confidence: float
    bbox: list
    type: str  # 'threat' / 'civilian'

@dataclass
class Event:
    
    drone_id: str
    timestamp: str
    location: dict  # {lat, lon, alt}
    detections: List[ObjectDetection]
    event_type: str = "routine" # routine / alert

    def to_json(self):
        
        return json.dumps(asdict(self), default=str)
    
    def to_dict(self):
        return asdict(self)