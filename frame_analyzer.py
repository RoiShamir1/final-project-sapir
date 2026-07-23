import torch
from ultralytics import YOLO
from typing import List, Union
import cv2
import numpy as np
from models import ObjectDetection


class FrameAnalyzer:
    """
    מחלקה לנתוח פריים בודד דרך המודל המאומן בתיקייה drone_final_v1
    """
    
    def __init__(self, model_path: str = 'runs/detect/drone_final_v1/weights/best.pt', 
                 conf_threshold: float = 0.5):
        
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"🚀 FrameAnalyzer Initializing on device: {self.device}")
        
        self.conf_threshold = conf_threshold
        self.model_path = model_path
        
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully from: {model_path}")
            print(f"📊 Model classes: {self.model.names}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def analyze_frame(self, frame: Union[np.ndarray, str]) -> List[ObjectDetection]:
        detections = []
        
        try:
            results = self.model(
                frame,
                device=self.device,
                verbose=False,
                conf=self.conf_threshold,
            )
            
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = list(map(int, box.xyxy[0])) 
                    
                    detection = ObjectDetection(
                        label=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        type="threat"
                    )
                    detections.append(detection)
                    
                    print(f"✅ Detected: {class_name} (confidence: {confidence:.2f})")
        
        except Exception as e:
            print(f"❌ Error analyzing frame: {e}")
            raise e
        
        return detections
    
    def analyze_frame_with_visualization(self, 
                                        frame: Union[np.ndarray, str],
                                        draw_boxes: bool = True) -> tuple[List[ObjectDetection], np.ndarray]:
        
        if isinstance(frame, str):
            image = cv2.imread(frame)
        else:
            image = frame.copy()
        
        detections = self.analyze_frame(frame)
        
        if draw_boxes and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                color = (0, 0, 255) if detection.type == "threat" else (0, 255, 0)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label = f"{detection.label} ({detection.confidence:.2f})"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, image
    
    def get_detections_summary(self, detections: List[ObjectDetection]) -> dict:
        
        summary = {
            "total_detections": len(detections),
            "threats": sum(1 for d in detections if d.type == "threat"),
            "civilians": sum(1 for d in detections if d.type == "civilian"),
            "average_confidence": sum(d.confidence for d in detections) / len(detections) if detections else 0,
            "detections_by_class": {}
        }
        
        for detection in detections:
            if detection.label not in summary["detections_by_class"]:
                summary["detections_by_class"][detection.label] = []
            summary["detections_by_class"][detection.label].append(detection.confidence)
        
        return summary


if __name__ == "__main__":
    analyzer = FrameAnalyzer(conf_threshold=0.5)
    
    detections = analyzer.analyze_frame("C:\\Users\\roish\\Documents\\GitHub\\final-project-sapir\\manPistol.png")
    
    cv2.waitKey(0)
    
    summary = analyzer.get_detections_summary(detections)
    print(f"📊 Summary: {summary}")
    
    print("✅ FrameAnalyzer ready to use!")
