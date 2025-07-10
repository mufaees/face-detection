from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_faces(self, frame):
        results = self.model(frame)[0]
        faces = []
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            conf = float(det.conf[0])
            if conf > self.conf_threshold:
                faces.append((x1, y1, x2, y2))
        return faces
