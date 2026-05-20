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
        """
        אתחול ה-FrameAnalyzer
        
        Args:
            model_path: נתיב למודל המאומן (ברירת מחדל: best.pt מ-drone_final_v1)
            conf_threshold: סף ביטחון מינימלי להצגת זיהויים (0-1)
        """
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
        """
        נתוח פריים בודד דרך המודל
        
        Args:
            frame: יכול להיות:
                   - numpy array של תמונה (כמו מ-cv2.imread או מ-frame מ-drone)
                   - נתיב לקובץ תמונה (string)
        
        Returns:
            רשימה של ObjectDetection עם כל הזיהויים שנמצאו
        """
        detections = []
        
        try:
            # הרצת המודל על הפריים
            results = self.model(
                frame,
                device=self.device,
                verbose=False,
                conf=self.conf_threshold,
            )
            
            # חילוץ התוצאות
            for result in results:
                for box in result.boxes:
                    # קבלת מידע על הזיהוי
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    bbox = list(map(int, box.xyxy[0]))  # [x1, y1, x2, y2]
                    
                    # יצירת ObjectDetection
                    detection = ObjectDetection(
                        label=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        type="threat"  # יכול להיות "threat" או "civilian" לפי הצורך
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
        """
        נתוח פריים עם ציור של תוצאות על התמונה
        
        Args:
            frame: פריים או נתיב לתמונה
            draw_boxes: האם לצייר boxes על התמונה
        
        Returns:
            tuple של (רשימת זיהויים, תמונה עם ציור)
        """
        # קריאת התמונה אם זה string
        if isinstance(frame, str):
            image = cv2.imread(frame)
        else:
            image = frame.copy()
        
        # ניתוח הפריים
        detections = self.analyze_frame(frame)
        
        # ציור של boxes אם בקשנו
        if draw_boxes and len(detections) > 0:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # בחירת צבע לפי סוג הזיהוי
                color = (0, 0, 255) if detection.type == "threat" else (0, 255, 0)
                
                # ציור ה-box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # ציור התווית
                label = f"{detection.label} ({detection.confidence:.2f})"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, image
    
    def get_detections_summary(self, detections: List[ObjectDetection]) -> dict:
        """
        קבלת סיכום של הזיהויים
        
        Args:
            detections: רשימת הזיהויים
        
        Returns:
            dict עם סטטיסטיקות על הזיהויים
        """
        summary = {
            "total_detections": len(detections),
            "threats": sum(1 for d in detections if d.type == "threat"),
            "civilians": sum(1 for d in detections if d.type == "civilian"),
            "average_confidence": sum(d.confidence for d in detections) / len(detections) if detections else 0,
            "detections_by_class": {}
        }
        
        # ספירה לפי סוג
        for detection in detections:
            if detection.label not in summary["detections_by_class"]:
                summary["detections_by_class"][detection.label] = []
            summary["detections_by_class"][detection.label].append(detection.confidence)
        
        return summary


# דוגמה של שימוש
if __name__ == "__main__":
    # יצירת analyzer
    analyzer = FrameAnalyzer(conf_threshold=0.5)
    
    # דוגמה 1: ניתוח תמונה מ-file
    detections = analyzer.analyze_frame("C:\\Users\\roish\\Documents\\GitHub\\final-project-sapir\\manPistol.png")
    
    # דוגמה 2: ניתוח עם ציור
    #detections, visualized = analyzer.analyze_frame_with_visualization("C:\\Users\\roish\\Documents\\GitHub\\final-project-sapir\\manPistol.png")
    #cv2.imshow("Detections", visualized)
    cv2.waitKey(0)
    
    # דוגמה 3: סיכום הזיהויים
    summary = analyzer.get_detections_summary(detections)
    print(f"📊 Summary: {summary}")
    
    print("✅ FrameAnalyzer ready to use!")
