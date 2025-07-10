import cv2
import os
import datetime
from ultralytics import YOLO
from embedings.face_embedder import FaceEmbedder
from deep_sort_realtime.deepsort_tracker import DeepSort
from db.database import create_db, log_event

# ✅ Initialize
model = YOLO("yolov8n-face.pt")
embedder = FaceEmbedder()
tracker = DeepSort(max_age=30)
create_db()

cap = cv2.VideoCapture("C:/FACE TRACKER  KATAMARAN/record_20250620_184504.mp4W")
known_faces = {}
previous_ids = set()
exited_log_ids = set()
frame_count = 0
skip_rate = 2  # Speed boost

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % skip_rate != 0:
        continue

    resized_frame = cv2.resize(frame, (640, 480))
    results = model(resized_frame, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        confidence = float(box.conf[0])
        if confidence < 0.4:
            continue
        detections.append(([x1, y1, w, h], confidence, 'face'))

    tracks = tracker.update_tracks(detections, frame=resized_frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        l, t, r, b = map(int, track.to_ltrb())
        current_ids.add(track_id)

        h, w, _ = resized_frame.shape
        l, t = max(0, l), max(0, t)
        r, b = min(w, r), min(h, b)

        face_crop = resized_frame[t:b, l:r]
        if face_crop is None or face_crop.size == 0 or (b - t) < 60 or (r - l) < 60:
            print(f"[DEBUG] ⚠️ Invalid crop size for Track ID: {track_id}")
            continue

        embedding = embedder.get_embedding(face_crop, track_id=track_id)
        if embedding is None:
            print(f"[DEBUG] ⚠️ Embedding failed for Track ID: {track_id}")
            continue

        if track_id not in known_faces:
            known_faces[track_id] = embedding
            date_folder = datetime.date.today().isoformat()
            save_path = f"logs/entries/{date_folder}"
            os.makedirs(save_path, exist_ok=True)
            filename = f"{save_path}/{track_id}.jpg"
            cv2.imwrite(filename, face_crop)

            now = datetime.datetime.now().isoformat()
            with open("logs/events.log", "a") as f:
                f.write(f"[ENTRY] {track_id} at {now}\n")
            log_event(track_id, filename, "ENTRY")
            print(f"🟢 Entry Logged: {track_id}")

        # Draw Box
        cv2.rectangle(resized_frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Detect exits
    exited_ids = previous_ids - current_ids
    for exited_id in exited_ids:
        if exited_id not in exited_log_ids:
            now = datetime.datetime.now().isoformat()
            with open("logs/events.log", "a") as f:
                f.write(f"[EXIT] {exited_id} at {now}\n")
            log_event(exited_id, "-", "EXIT")
            print(f"🔴 Exit Logged: {exited_id}")
            exited_log_ids.add(exited_id)

        if exited_id in known_faces:
            del known_faces[exited_id]

    previous_ids = current_ids

    # Visitor counter
    cv2.putText(resized_frame, f"Unique Visitors: {len(known_faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Face Tracker", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
another main.py