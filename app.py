import os
import sqlite3
import sys
import threading
import time
from datetime import datetime
from collections import deque
from functools import wraps
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fight_detection.Fight_utils import streaming_framesInference


import cv2
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash

from notification import pushbullet_noti


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "visioguard.db"
LOG_DIR = BASE_DIR / "logs"
PROOF_DIR = BASE_DIR / "static" / "proofs"
MODEL_PATH = BASE_DIR / os.getenv("WEAPON_MODEL", "best.pt")

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
CAMERA_BACKEND = os.getenv("CAMERA_BACKEND", "dshow").lower()
ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "30"))
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "5"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "416"))
STREAM_DELAY_SECONDS = float(os.getenv("STREAM_DELAY_SECONDS", "0.08"))
WEAPON_DETECTION_ENABLED = os.getenv("WEAPON_DETECTION_ENABLED", "1") == "1"
FIGHT_DETECTION_ENABLED = os.getenv("FIGHT_DETECTION_ENABLED", "1") == "1"
FIGHT_EVERY_N_FRAMES = int(os.getenv("FIGHT_EVERY_N_FRAMES", "3"))
FIGHT_MOTION_THRESHOLD = float(os.getenv("FIGHT_MOTION_THRESHOLD", "2.2"))
FIGHT_AREA_THRESHOLD = float(os.getenv("FIGHT_AREA_THRESHOLD", "0.08"))
FIGHT_CONFIRMATION_WINDOWS = int(os.getenv("FIGHT_CONFIRMATION_WINDOWS", "3"))
FIGHT_CONFIDENCE_THRESHOLD = 0.90
END_FIGHT_THRESHOLD = 0.85

app = Flask(__name__)
app.secret_key = os.getenv("VISIOGUARD_SECRET_KEY", "change-this-secret-key")


def configure_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "visioguard.log"
    for handler in app.logger.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_path:
            return
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    file_handler.setLevel("INFO")
    if app.logger.handlers:
        file_handler.setFormatter(app.logger.handlers[0].formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel("INFO")


def camera_source_value():
    return int(CAMERA_SOURCE) if CAMERA_SOURCE.isdigit() else CAMERA_SOURCE


def camera_backend_value():
    if not CAMERA_SOURCE.isdigit():
        return 0
    if CAMERA_BACKEND == "dshow":
        return cv2.CAP_DSHOW
    if CAMERA_BACKEND == "msmf":
        return cv2.CAP_MSMF
    return 0


def open_camera():
    source = camera_source_value()
    backend = camera_backend_value()
    capture = cv2.VideoCapture(source, backend) if backend else cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    configure_logging()
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anomaly_type TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL,
                proof_image TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def log_anomaly(anomaly_type, label, confidence, proof_image):
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO anomalies (anomaly_type, label, confidence, proof_image, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (anomaly_type, label, confidence, proof_image, created_at),
        )


class FightDetector:
    """Fight detector that prefers a frame-level library API when available.

    The fight_detection package's public example starts its own camera loop.
    This class keeps VisioGuard on one shared camera stream to avoid lag and
    falls back to lightweight OpenCV motion analysis when no frame API exists.
    """

    def __init__(self):
        self.backend_name = "opencv-motion"
        self.library_method = self.find_library_method()
        self.previous_gray = None
        self.recent_positive_windows = deque(
            maxlen=max(1, FIGHT_CONFIRMATION_WINDOWS)
        )
        self.frame_buffer = deque(maxlen=16)
        self.mc3_history = deque(maxlen=3)

        self.fight_active = False

    def predict_mc3(self, frame):
        self.frame_buffer.append(frame)

        if len(self.frame_buffer) < 16:
            return False, 0.0

        try:
            print("\n========== FIGHT DETECTOR DEBUG ==========")
            print("Buffer size before inference:", len(self.frame_buffer))

            label, classes = streaming_framesInference(list(self.frame_buffer))

            print("RAW RESULT:", label)
            print("Classes:", classes)
            print(f"MC3 Prediction: {label} | Buffer Size: {len(self.frame_buffer)}")
            print("=========================================\n")

            # Convert the model's string output into (bool, confidence)
            fight_conf = 0.0

            for cls, conf in classes:
                if cls == "fight":
                    fight_conf = conf
                    break

            print(f"Fight confidence: {fight_conf:.3f}")

            # You can adjust this threshold
            # Start fight
            if not self.fight_active and fight_conf >= FIGHT_CONFIDENCE_THRESHOLD:
                self.fight_active = True

            # End fight
            elif self.fight_active and fight_conf < END_FIGHT_THRESHOLD:
                self.fight_active = False

            return self.fight_active, fight_conf

        except Exception as e:
            print("FIGHT DETECTOR ERROR:", str(e))
            import traceback
            traceback.print_exc()
            return False, 0.0


    def find_library_method(self):
        if os.getenv("USE_FIGHT_LIBRARY", "1") != "1":
            return None

        try:
            from fight_detection import Fight_utils
        except Exception as exc:
            app.logger.info("fight_detection package not available: %s", exc)
            return None

        for method_name in (
            "predict_frame",
            "detect_frame",
            "process_frame",
            "predict",
            "detect",
            "is_fight",
        ):
            method = getattr(Fight_utils, method_name, None)
            if callable(method):
                self.backend_name = f"fight_detection.{method_name}"
                app.logger.info("Using fight detection backend: %s", self.backend_name)
                return method

        app.logger.info(
            "fight_detection is installed, but only standalone streaming API was found. "
            "Using OpenCV motion fallback."
        )
        return None

    def predict(self, frame):

        motion_fight, motion_conf = self.predict_with_motion(frame)
        mc3_fight, mc3_conf = self.predict_mc3(frame)

        print(f"Motion={motion_fight}, MC3={mc3_fight}")

        # Save MC3 result history
        self.mc3_history.append(mc3_fight)
        print("MC3 History: ", list(self.mc3_history))

        # If both detectors agree
        if motion_fight and mc3_fight:
            print("Fight detected by Motion + MC3")
            return True, max(motion_conf, mc3_conf)

        # MC3 alone can trigger if detected in 2 of last 3 windows
        if len(self.mc3_history) == self.mc3_history.maxlen and sum(self.mc3_history) >= 2:
            print("Fight detected by MC3 history")
            return True, mc3_conf

        return False, max(motion_conf, mc3_conf)


    def predict_with_library(self, frame):
        result = self.library_method(frame)

        if isinstance(result, bool):
            return result, 0.90 if result else 0.0

        if isinstance(result, dict):
            detected = bool(
                result.get("fight")
                or result.get("is_fight")
                or result.get("detected")
                or result.get("prediction") in ("fight", "Fight", 1, True)
            )
            confidence = float(
                result.get("confidence")
                or result.get("probability")
                or result.get("score")
                or (0.90 if detected else 0.0)
            )
            return detected, min(0.99, confidence)

        if isinstance(result, (tuple, list)) and result:
            detected_value = result[0]
            detected = detected_value in (True, 1, "fight", "Fight")
            confidence = float(result[1]) if len(result) > 1 else (0.90 if detected else 0.0)
            return detected, min(0.99, confidence)

        return False, 0.0

    def predict_with_motion(self, frame):
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.previous_gray is None:
            self.previous_gray = gray
            return False, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self.previous_gray,
            gray,
            None,
            0.5,
            2,
            11,
            2,
            5,
            1.1,
            0,
        )
        self.previous_gray = gray

        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        active_motion = magnitude > FIGHT_MOTION_THRESHOLD
        active_ratio = float(active_motion.mean())
        active_strength = float(magnitude[active_motion].mean()) if active_motion.any() else 0.0

        is_positive_window = (
            active_ratio >= FIGHT_AREA_THRESHOLD
            and active_strength >= FIGHT_MOTION_THRESHOLD
        )
        self.recent_positive_windows.append(is_positive_window)

        confidence = min(0.99, (active_ratio / max(FIGHT_AREA_THRESHOLD, 0.01)) * 0.45)
        if active_strength > 0:
            confidence += min(0.45, active_strength / 12)
        confidence = min(0.99, confidence)

        is_fight = (
            len(self.recent_positive_windows) == self.recent_positive_windows.maxlen
            and all(self.recent_positive_windows)
        )
        return is_fight, confidence


class VisionGuardCamera:
    def __init__(self):
        self.model = None
        if WEAPON_DETECTION_ENABLED:
            from ultralytics import YOLO

            self.model = YOLO(str(MODEL_PATH))
        self.fight_detector = FightDetector() if FIGHT_DETECTION_ENABLED else None
        self.capture = open_camera()
        self.lock = threading.Lock()
        self.latest_raw_frame = None
        self.latest_weapon_frame = None
        self.latest_fight_frame = None
        self.latest_weapon_objects = []
        self.latest_fight_objects = []
        self.weapon_overlay_until = 0
        self.fight_overlay_until = 0
        self.latest_error = None
        self.alert_error = None
        self.running = False
        self.capture_thread = None
        self.weapon_thread = None
        self.fight_thread = None
        self.frame_count = 0
        self.failed_grabs = 0
        self.last_alert_at = {}

    def start(self):
        if self.running:
            return
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

        if WEAPON_DETECTION_ENABLED and self.model:
            self.weapon_thread = threading.Thread(target=self.weapon_loop, daemon=True)
            self.weapon_thread.start()

        if self.fight_detector:
            self.fight_thread = threading.Thread(target=self.fight_loop, daemon=True)
            self.fight_thread.start()

    def stop(self):
        self.running = False
        if self.capture:
            self.capture.release()

    def reopen_camera(self):
        if self.capture:
            self.capture.release()
        time.sleep(0.5)
        self.capture = open_camera()
        self.failed_grabs = 0

    def get_status(self):
        with self.lock:
            return {
                "objects": list(self.latest_weapon_objects + self.latest_fight_objects),
                "error": self.latest_error or self.alert_error,
                "has_frame": self.latest_raw_frame is not None,
                "fight_backend": (
                    self.fight_detector.backend_name if self.fight_detector else "disabled"
                ),
            }

    def capture_loop(self):
        while self.running:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.failed_grabs += 1
                with self.lock:
                    self.latest_error = "Camera stream is not available."
                if self.failed_grabs >= 5:
                    app.logger.warning("Reopening camera after repeated failed frame grabs")
                    self.reopen_camera()
                time.sleep(1)
                continue

            self.failed_grabs = 0
            self.frame_count += 1
            with self.lock:
                now = time.time()
                self.latest_raw_frame = frame.copy()
                if self.latest_weapon_frame is None or now > self.weapon_overlay_until:
                    self.latest_weapon_frame = frame.copy()
                if self.latest_fight_frame is None or now > self.fight_overlay_until:
                    self.latest_fight_frame = frame.copy()
                self.latest_error = self.alert_error

    def get_latest_raw_frame(self):
        with self.lock:
            return None if self.latest_raw_frame is None else self.latest_raw_frame.copy()

    def weapon_loop(self):
        last_processed_count = -1
        while self.running:
            if self.frame_count == last_processed_count or self.frame_count % DETECT_EVERY_N_FRAMES != 0:
                time.sleep(0.02)
                continue

            frame = self.get_latest_raw_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            last_processed_count = self.frame_count
            try:
                annotated, objects = self.detect_weapons(frame)
                if objects:
                    self.handle_anomalies(annotated, objects)
                with self.lock:
                    self.latest_weapon_frame = annotated
                    if objects:
                        self.weapon_overlay_until = time.time() + 1.5
                    self.latest_weapon_objects = objects
            except Exception as exc:
                app.logger.exception("Weapon detection failed")
                with self.lock:
                    self.latest_error = f"Weapon detection failed: {exc}"
                time.sleep(1)

    def fight_loop(self):
        last_processed_count = -1
        while self.running:
            if self.frame_count == last_processed_count or self.frame_count % FIGHT_EVERY_N_FRAMES != 0:
                time.sleep(0.02)
                continue

            frame = self.get_latest_raw_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            last_processed_count = self.frame_count
            annotated = frame.copy()
            objects = []
            try:
                fight_detected, fight_confidence = self.fight_detector.predict(frame)
                if fight_detected:
                    objects.append(
                        {
                            "type": "Fight",
                            "label": "Fight",
                            "confidence": fight_confidence,
                        }
                    )
                    annotated = self.draw_fight_warning(annotated, fight_confidence)
                    self.handle_anomalies(annotated, objects)

                with self.lock:
                    self.latest_fight_frame = annotated
                    if objects:
                        self.fight_overlay_until = time.time() + 1.5
                    self.latest_fight_objects = objects
            except Exception as exc:
                app.logger.exception("Fight detection failed")
                with self.lock:
                    self.latest_error = f"Fight detection failed: {exc}"
                time.sleep(1)

    def detect_weapons(self, frame):
        import torch
        from ultralytics.utils.plotting import Annotator

        objects = []
        annotated = frame.copy()
        results = self.model.predict(
            annotated,
            conf=CONFIDENCE_THRESHOLD,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
        )

        for result in results:
            annotator = Annotator(annotated)
            for box in result.boxes:
                coords = box.xyxy[0].to(dtype=torch.float)
                class_id = int(box.cls)
                label = self.model.names[class_id]
                confidence = float(box.conf[0]) if box.conf is not None else None
                annotator.box_label(coords, f"{label} {confidence:.2f}" if confidence else label)
                objects.append(
                    {
                        "type": "Weapon",
                        "label": label,
                        "confidence": confidence,
                    }
                )
            annotated = annotator.result()

        return annotated, objects

    def draw_fight_warning(self, frame, confidence):
        label = f"Fight {confidence:.2f}"
        cv2.rectangle(frame, (14, 14), (195, 58), (20, 20, 220), -1)
        cv2.putText(
            frame,
            label,
            (26, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def handle_anomalies(self, frame, objects):
        for detected in objects:
            label = detected["label"]
            anomaly_type = detected.get("type", "Anomaly")
            key = f"{anomaly_type}:{label}"
            now = time.time()
            if now - self.last_alert_at.get(key, 0) < ALERT_COOLDOWN_SECONDS:
                continue

            self.last_alert_at[key] = now
            proof_image = self.save_proof(frame, label)
            log_anomaly(anomaly_type, label, detected["confidence"], proof_image)
            self.send_alert_async(anomaly_type, label, proof_image)

    def save_proof(self, frame, label):
        safe_label = "".join(ch if ch.isalnum() else "_" for ch in label).strip("_")
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_label}.jpg"
        proof_path = PROOF_DIR / filename
        cv2.imwrite(str(proof_path), frame)
        return f"proofs/{filename}"

    def send_alert_async(self, anomaly_type, label, proof_image):
        thread = threading.Thread(
            target=self.send_alert,
            args=(anomaly_type, label, proof_image),
            daemon=True,
        )
        thread.start()

    def send_alert(self, anomaly_type, label, proof_image):
        try:
            pushbullet_noti(
                "VisioGuard Alert",
                (
                    f"{anomaly_type} anomaly detected: {label} at "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
                    f"Proof: {proof_image}"
                ),
            )
            self.alert_error = None
        except Exception as exc:
            self.alert_error = f"Alert could not be sent: {exc}"
            app.logger.exception("Pushbullet alert failed")

    def jpeg_stream(self, feed_type="weapon"):
        while True:
            with self.lock:
                if feed_type == "fight":
                    source_frame = (
                        self.latest_fight_frame
                        if self.latest_fight_frame is not None
                        else self.latest_raw_frame
                    )
                elif feed_type == "raw":
                    source_frame = self.latest_raw_frame
                else:
                    source_frame = (
                        self.latest_weapon_frame
                        if self.latest_weapon_frame is not None
                        else self.latest_raw_frame
                    )
                frame = None if source_frame is None else source_frame.copy()

            if frame is None:
                frame = self.placeholder_frame()

            ok, buffer = cv2.imencode(".jpg", frame)
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
            time.sleep(STREAM_DELAY_SECONDS)

    @staticmethod
    def placeholder_frame(message="Starting camera..."):
        import numpy as np

        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        lines = []
        current = ""
        for word in message.split():
            test_line = f"{current} {word}".strip()
            if len(test_line) > 42:
                lines.append(current)
                current = word
            else:
                current = test_line
        if current:
            lines.append(current)

        start_y = 220 - (len(lines) * 18)
        for index, line in enumerate(lines[:6]):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)[0]
            x = max(20, (640 - text_size[0]) // 2)
            y = start_y + (index * 34)
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (32, 44, 58),
                2,
                cv2.LINE_AA,
            )
        return frame


camera_service = None
camera_service_error = None


def get_camera_service():
    global camera_service, camera_service_error
    if camera_service is None:
        try:
            camera_service = VisionGuardCamera()
            camera_service.start()
            camera_service_error = None
        except Exception as exc:
            camera_service_error = (
                "Detector could not start. Check that the active Python environment "
                f"has the correct Ultralytics/Torch versions for {MODEL_PATH.name}. "
                f"Details: {exc}"
            )
            app.logger.exception("Detector startup failed")
            return None
    return camera_service


def get_camera_status():
    service = get_camera_service()
    if service is None:
        return {
            "objects": [],
            "error": camera_service_error or "Detector is not available.",
            "has_frame": False,
            "fight_backend": "unavailable",
        }
    return service.get_status()


def fetch_recent_anomalies(limit=5):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM anomalies ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows


def placeholder_stream(message):
    while True:
        frame = VisionGuardCamera.placeholder_frame(message)
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
        time.sleep(1)


#@app.before_first_request
def prepare_app():
    init_db()


@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/health")
def health():
    versions = {}
    for package_name in ("flask", "cv2", "torch", "ultralytics"):
        try:
            module = __import__(package_name)
            versions[package_name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            versions[package_name] = f"not available: {exc}"

    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "camera_source": CAMERA_SOURCE,
        "camera_backend": CAMERA_BACKEND,
        "weapon_detection_enabled": WEAPON_DETECTION_ENABLED,
        "fight_detection_enabled": FIGHT_DETECTION_ENABLED,
        "alert_cooldown_seconds": ALERT_COOLDOWN_SECONDS,
        "fight_every_n_frames": FIGHT_EVERY_N_FRAMES,
        "weapon_every_n_frames": DETECT_EVERY_N_FRAMES,
        "yolo_image_size": YOLO_IMAGE_SIZE,
        "stream_delay_seconds": STREAM_DELAY_SECONDS,
        "versions": versions,
    }


@app.route("/register", methods=("GET", "POST"))
def register():
    if request.method == "POST":
        name = request.form["name"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("register.html")

        try:
            with get_db() as conn:
                conn.execute(
                    """
                    INSERT INTO users (name, email, password_hash, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        name,
                        email,
                        generate_password_hash(password),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
            flash("Account created. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("An account with this email already exists.", "error")

    return render_template("register.html")


@app.route("/login", methods=("GET", "POST"))
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]

        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    status = get_camera_status()
    recent_anomalies = fetch_recent_anomalies()
    return render_template(
        "dashboard.html",
        status=status,
        cooldown=ALERT_COOLDOWN_SECONDS,
        recent_anomalies=recent_anomalies,
    )


@app.route("/test_alert", methods=("POST",))
@login_required
def test_alert():
    try:
        pushbullet_noti(
            "VisioGuard Test Alert",
            f"Test alert sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        )
        flash("Pushbullet test alert sent.", "success")
    except Exception as exc:
        app.logger.exception("Pushbullet test alert failed")
        flash(f"Pushbullet test alert failed: {exc}", "error")
    return redirect(url_for("dashboard"))


@app.route("/video_feed")
@login_required
def video_feed():
    return weapon_feed()


@app.route("/weapon_feed")
@login_required
def weapon_feed():
    service = get_camera_service()
    if service is None:
        return Response(
            placeholder_stream(camera_service_error or "Detector is not available."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    return Response(
        service.jpeg_stream("weapon"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/fight_feed")
@login_required
def fight_feed():
    service = get_camera_service()
    if service is None:
        return Response(
            placeholder_stream(camera_service_error or "Detector is not available."),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
    return Response(
        service.jpeg_stream("fight"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/recent_anomalies")
@login_required
def api_recent_anomalies():
    rows = fetch_recent_anomalies()
    response = jsonify(
        [
            {
                "id": row["id"],
                "anomaly_type": row["anomaly_type"],
                "label": row["label"],
                "confidence": row["confidence"],
                "proof_image": url_for("static", filename=row["proof_image"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/api/status")
@login_required
def api_status():
    response = jsonify(get_camera_status())
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/records")
@login_required
def records():
    with get_db() as conn:
        anomalies = conn.execute(
            "SELECT * FROM anomalies ORDER BY created_at DESC"
        ).fetchall()
    return render_template("records.html", anomalies=anomalies)


if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
