import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import csv
from datetime import datetime

# --- Optional: local face recognition with InsightFace ---
try:
    import insightface
    _INSIGHTFACE_AVAILABLE = True
except Exception:
    insightface = None
    _INSIGHTFACE_AVAILABLE = False

# --- Optional: lightweight popup using Tkinter ---
try:
    from tkinter import messagebox, Tk
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


#      Pose Utilities

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#      UI Text Config

# Edit these strings to change what is written on the screen
UI_TEXT = {
    "angle": "Elbow Angle: {angle}°",
    "reps": "Push-ups: {count}",
    "user": "User: {id_text}",
    "user_detected_no": "User detected: No (Unknown)",
    "user_detected_yes": "User detected: Yes ({name})",
    "face_active": "Face Recognition: ACTIVE",
    "face_stopped": "Face Recognition: STOPPED",
    "frame": "Frame: {frame}",
    "face_label": "{name}",  # label shown above the face box
}

#   Detection/Counting Tuning

# Push-up angle thresholds (degrees)
UP_ANGLE_DEG = 150
DOWN_ANGLE_DEG = 135

# How often to refresh the face bounding box after recognition (frames)
FACE_BBOX_UPDATE_INTERVAL = 10

# How long to show "not detected" regardless of recognition (seconds)
DETECTION_GRACE_SEC = 2.0

def calculate_angle(a, b, c):
    """Returns angle ABC (in degrees) for points a, b, c.
    Points can be (x, y) lists/tuples in pixel coordinates.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

#      Face DB Helpers

class LocalFaceDB:
    """Simple local face DB using InsightFace.

    Directory structure:
        faces/
          Alice/
            alice1.jpg
            alice2.jpg
          Bob/
            bob1.jpg

    The class builds one embedding per person by averaging embeddings of their images.
    """
    def __init__(self, face_dir="faces", ctx_id=0, det_size=(640, 640)):
        self.face_dir = face_dir
        self.app = None
        self.known = {}  # name -> embedding (L2-normalized)
        if _INSIGHTFACE_AVAILABLE:
            try:
                self.app = insightface.app.FaceAnalysis(allowed_modules=["detection", "recognition"])  # recognition for embeddings
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                self._build_db()
            except Exception as e:
                print(f"[FaceDB] Failed to init InsightFace: {e}")
                self.app = None
        else:
            print("[FaceDB] InsightFace not installed. Face recognition disabled.")

    def _list_images(self, folder):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        for root, _, files in os.walk(folder):
            for f in files:
                if f.lower().endswith(exts):
                    yield os.path.join(root, f)

    def _build_db(self):
        if not os.path.isdir(self.face_dir):
            print(f"[FaceDB] Face dir '{self.face_dir}' does not exist. Continuing without known faces.")
            return
        people = [d for d in os.listdir(self.face_dir) if os.path.isdir(os.path.join(self.face_dir, d))]
        for person in people:
            imgs = list(self._list_images(os.path.join(self.face_dir, person)))
            if not imgs:
                continue
            embs = []
            for path in imgs:
                img = cv2.imread(path)
                if img is None:
                    continue
                faces = self.app.get(img)
                if not faces:
                    continue
                # pick the largest face
                f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
                if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                    embs.append(f.normed_embedding)
            if embs:
                avg_emb = np.mean(np.stack(embs, axis=0), axis=0)
                # re-normalize
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-12)
                self.known[person] = avg_emb
        print(f"[FaceDB] Loaded {len(self.known)} identities from '{self.face_dir}'.")

    def recognize(self, frame, thresh=0.35):
        """Recognize largest face in the frame. Returns (name, score) or ("Unknown", score).
        'thresh' is cosine similarity threshold (higher is more similar). Typical 0.3-0.4.
        """
        if self.app is None or not self.known:
            return "Unknown", 0.0
        faces = self.app.get(frame)
        if not faces:
            return "Unknown", 0.0
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        if not hasattr(f, "normed_embedding") or f.normed_embedding is None:
            return "Unknown", 0.0
        emb = f.normed_embedding
        best_name, best_score = "Unknown", -1.0
        for name, known_emb in self.known.items():
            # cosine similarity because embeddings are L2-normalized
            score = float(np.dot(emb, known_emb))
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= thresh:
            return best_name, best_score
        return "Unknown", best_score

    def recognize_with_bbox(self, frame, thresh=0.35):
        """Like recognize(), but also returns bbox for the largest face as (x1, y1, x2, y2).
        Returns (name, score, bbox or None).
        """
        if self.app is None or not self.known:
            return "Unknown", 0.0, None
        faces = self.app.get(frame)
        if not faces:
            return "Unknown", 0.0, None
        f = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
        bbox = None
        if hasattr(f, "bbox") and f.bbox is not None:
            try:
                x1, y1, x2, y2 = map(int, f.bbox)
                bbox = (x1, y1, x2, y2)
            except Exception:
                bbox = None
        if not hasattr(f, "normed_embedding") or f.normed_embedding is None:
            return "Unknown", 0.0, bbox
        emb = f.normed_embedding
        best_name, best_score = "Unknown", -1.0
        for name, known_emb in self.known.items():
            score = float(np.dot(emb, known_emb))
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= thresh:
            return best_name, best_score, bbox
        return "Unknown", best_score, bbox


#      Logging Helpers

LOG_CSV = "workout_log.csv"

def log_session(name, reps, total_seconds, pace_rpm):
    new_file = not os.path.isfile(LOG_CSV)
    with open(LOG_CSV, mode="a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["datetime", "name", "reps", "duration_sec", "pace_rpm"])  # header
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, reps, f"{total_seconds:.1f}", f"{pace_rpm:.2f}"])

#          Main


def main(video_path=None, output_path=None):
    # --- Session timers ---
    session_start = time.time()

    # --- Face DB (local) ---
    face_db = LocalFaceDB(face_dir="faces")  # create a folder named 'faces' with subfolders per person
    recognized_name = "Unknown"
    recog_conf = 0.0
    recog_attempt_frames = 240  # allow ~8 seconds at 30 FPS
    face_recognition_active = True  # Flag to control face recognition processing
    user_found = False  # Flag to track if user has been found
    face_bbox = None  # Last seen face bbox
    bbox_persist_frames = 300  # default; will be recalculated from FPS to ~2–3s
    bbox_countdown = 0
    last_bbox_update_frame = -9999

    # --- Video input setup ---
    if video_path:
        cap = cv2.VideoCapture(video_path)
        print(f"[Video] Processing video file: {video_path}")
    else:
        cap = cv2.VideoCapture(0)
        print("[Video] Using webcam")
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_path if video_path else 'webcam'}")
        return

    # --- Video output setup ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Calculate bbox persistence to ~2–3 seconds based on FPS
    if fps is None or fps <= 0:
        fps = 30
    bbox_persist_frames = int(2.5 * fps)
    grace_frames = int(DETECTION_GRACE_SEC * fps)
    
    if output_path:
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[Video] Output will be saved to: {output_path}")
    else:
        out = None

    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only flip if using webcam (not for video files)
            if not video_path:
                frame = cv2.flip(frame, 1)
            
            height, width, _ = frame.shape

            # --- Optimized face recognition: only run until user is found ---
            if _INSIGHTFACE_AVAILABLE:
                if frame_idx < recog_attempt_frames:
                    name, score, bbox = face_db.recognize_with_bbox(frame)
                    # Keep the best match seen so far
                    if score > recog_conf:
                        recognized_name, recog_conf = name, score
                        print(f"[Face Recognition] Found: {name} with confidence: {score:.2f}")
                    # Update bbox drawing if any face detected
                    if name != "Unknown" and bbox is not None:
                        face_bbox = bbox
                        bbox_countdown = bbox_persist_frames
                        last_bbox_update_frame = frame_idx
                    
                    # If we found Gaurav with high confidence, stop face recognition
                    if recognized_name.lower() == "gaurav" and recog_conf > 0.35:
                        user_found = True
                        face_recognition_active = False
                        print(f"[Face Recognition] ✅ User '{recognized_name}' confirmed! Stopping face recognition to reduce lag.")
                else:
                    # If we haven't found the user after max attempts, stop face recognition
                    face_recognition_active = False
                    if not user_found:
                        print(f"[Face Recognition] ⚠️ User not found after {recog_attempt_frames} frames. Continuing without face recognition.")
            # If we lost the bbox early, try to refresh it every few frames while active
            if face_recognition_active and _INSIGHTFACE_AVAILABLE and (frame_idx - last_bbox_update_frame) > FACE_BBOX_UPDATE_INTERVAL:
                _n, _s, bbox = face_db.recognize_with_bbox(frame)
                if bbox is not None:
                    face_bbox = bbox
                    bbox_countdown = bbox_persist_frames
                    last_bbox_update_frame = frame_idx

            # --- Pose inference ---
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- Header overlay (taller to avoid text overlap) ---
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, 150), (50, 50, 50), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            try:
                landmarks = results.pose_landmarks.landmark

                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]

                angle = calculate_angle(shoulder, elbow, wrist)

                # Angle text (left)
                angle_text = UI_TEXT["angle"].format(angle=int(angle))
                cv2.putText(frame, angle_text, (30, 90),
                            cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)

                # Rep counting: wait until user is found; tuned thresholds with hysteresis
                if not user_found:
                    pass
                elif angle > UP_ANGLE_DEG:
                    stage = "up"
                elif angle < DOWN_ANGLE_DEG and stage == 'up':
                    stage = "down"
                    counter += 1

                # Counter text (centered at bottom of header)
                reps_text = UI_TEXT["reps"].format(count=counter)
                (tw, th), _ = cv2.getTextSize(reps_text, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)
                cv2.putText(frame, reps_text, ((width - tw)//2, 125),
                            cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 255, 255), 3)

                # Side progress bars (mapped DOWN->UP degrees)
                bar_fill = np.interp(angle, (DOWN_ANGLE_DEG, UP_ANGLE_DEG), (350, 0))
                # Left bar
                for i in range(int(350 - bar_fill), 350):
                    # gradient from purple to yellow
                    t = (i - (350 - bar_fill)) / (bar_fill + 1e-6)
                    color = (0, int(255 * t), 255 - int(255 * t))
                    cv2.line(frame, (100, i + 150), (150, i + 150), color, 1)
                # Right bar
                for i in range(int(350 - bar_fill), 350):
                    t = (i - (350 - bar_fill)) / (bar_fill + 1e-6)
                    color = (0, int(255 * t), 255 - int(255 * t))
                    cv2.line(frame, (width - 150, i + 150), (width - 100, i + 150), color, 1)

                # Bar outlines + caps
                cv2.rectangle(frame, (100, 150), (150, 500), (255, 255, 255), 2)
                cv2.rectangle(frame, (width - 150, 150), (width - 100, 500), (255, 255, 255), 2)
                cv2.circle(frame, (125, 150), 10, (255, 0, 255), -1)
                cv2.circle(frame, (width - 125, 150), 10, (255, 0, 255), -1)

                # Draw pose landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=6, circle_radius=8),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=6)
                )

            except Exception:
                pass

            # --- Show face bounding box if available ---
            if (bbox_countdown > 0) and face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                # Clamp to frame
                x1 = max(0, min(x1, width-1)); y1 = max(0, min(y1, height-1))
                x2 = max(0, min(x2, width-1)); y2 = max(0, min(y2, height-1))
                box_color = (0, 255, 0) if user_found else (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                label_name = recognized_name.title() if recognized_name != "Unknown" else "Person"
                label = UI_TEXT["face_label"].format(name=label_name)
                cv2.putText(frame, label, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_DUPLEX, 0.8, box_color, 2)
                bbox_countdown -= 1

            # --- User detected banner ---
            # Use frame-based grace window so it doesn't elapse during init
            if frame_idx < grace_frames:
                # Always show NO for the first DETECTION_GRACE_SEC seconds
                status_line = UI_TEXT["user_detected_no"]
                color = (0, 255, 255)
            elif user_found:
                status_line = UI_TEXT["user_detected_yes"].format(name=recognized_name.title())
                color = (0, 255, 0)
            else:
                status_line = UI_TEXT["user_detected_no"]
                color = (0, 255, 255)

            # Right-aligned and slightly higher for visibility
            (uw, uh), _ = cv2.getTextSize(status_line, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            cv2.putText(frame, status_line, (width - 20 - uw, 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

            # (Removed) Face recognition status text hidden per user request

            # Show frame info for video processing (bottom-right)
            if video_path:
                frame_info = UI_TEXT["frame"].format(frame=frame_idx)
                cv2.putText(frame, frame_info, (width - 170, height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Write frame to output video if specified
            if out:
                out.write(frame)

            # Only show display window if not processing video file (to avoid display errors)
            if not video_path:
                cv2.imshow('Push-Up Tracker', frame)
            frame_idx += 1

            # Show progress for video processing
            if video_path and frame_idx % 30 == 0:  # Show progress every 30 frames
                print(f"[Progress] Processed {frame_idx} frames...")

            # Only handle key events if not processing video file
            if not video_path:
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # --- Session end: compute stats, log, popup ---
    cap.release()
    if out:
        out.release()
    if not video_path:
        cv2.destroyAllWindows()

    total_time = time.time() - session_start
    pace_rpm = (counter / total_time) * 60.0 if total_time > 0 else 0.0

    # Persist to CSV
    log_session(recognized_name, counter, total_time, pace_rpm)

    # Popup summary (optional)
    summary = f"Total Reps: {counter}\nTime: {total_time:.1f} sec\nPace: {pace_rpm:.2f} reps/min\nUser: {recognized_name}"
    if output_path:
        summary += f"\nOutput saved to: {output_path}"
    
    print("\n=== Session Summary ===\n" + summary)
    if _TK_AVAILABLE:
        try:
            root = Tk()
            root.withdraw()
            messagebox.showinfo("Session Summary", summary)
            root.destroy()
        except Exception as e:
            print(f"[Popup] Tkinter failed: {e}")
    else:
        print("[Popup] Tkinter not available; printed summary to console instead.")


if __name__ == "__main__":
    import sys
    
    # Check if video file is provided as command line argument
    if len(sys.argv) > 1:
        input_video = sys.argv[1]
        output_video = "output_pushup_tracker.mp4"
        if len(sys.argv) > 2:
            output_video = sys.argv[2]
        
        print(f"🎥 Processing video: {input_video}")
        print(f"💾 Output will be saved as: {output_video}")
        main(video_path=input_video, output_path=output_video)
    else:
        # Use webcam if no video file provided
        print("📹 Using webcam (provide video file as argument to process video)")
        main()