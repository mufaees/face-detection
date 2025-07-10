from insightface.app import FaceAnalysis
import numpy as np
import cv2
import os

class FaceEmbedder:
    def __init__(self):
        self.model = FaceAnalysis(name='buffalo_l')
        self.model.prepare(ctx_id=-1)  # Use CPU

    def get_embedding(self, face_img, track_id=None):
        if face_img is None or face_img.size == 0:
            print("[DEBUG] ❌ Empty face crop")
            return None

        try:
            face_img = cv2.resize(face_img, (112, 112))
            rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            faces = self.model.get(rgb_face)
            if not faces:
                print("[DEBUG] ❌ No face detected inside crop")
                if track_id:
                    os.makedirs("debug_failed_crops", exist_ok=True)
                    cv2.imwrite(f"debug_failed_crops/track_{track_id}.jpg", face_img)
                return None
            return faces[0].embedding
        except Exception as e:
            print(f"[DEBUG] ⚠️ Embedding error for Track ID {track_id}: {e}")
            return None

    def is_same_person(self, emb1, emb2, threshold=0.6):
        if emb1 is None or emb2 is None:
            return False
        diff = emb1 - emb2
        distance = np.linalg.norm(diff)
        return distance < threshold
