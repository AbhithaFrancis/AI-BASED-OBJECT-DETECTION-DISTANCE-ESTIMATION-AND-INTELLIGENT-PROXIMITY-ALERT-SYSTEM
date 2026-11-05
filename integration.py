import cv2
import numpy as np
import time
import threading
import requests
from collections import deque, Counter

from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ---------- CONFIGURATION ----------

PHONE_IP = "100.65.213.21"
CAMERA_URL = f"http://{PHONE_IP}:8080/video"

ESP32_IP = "10.141.174.200"
DISTANCE_URL = f"http://{ESP32_IP}/distance"

MODEL_PATH = "yolo11m.pt"

WARNING_DISTANCE = 100.0
DANGER_DISTANCE = 30.0
MAX_HISTORY = 20

LOW_LIGHT_THRESHOLD = 55.0
BLUR_THRESHOLD = 50.0
LOW_CONTRAST_THRESHOLD = 25.0

# Detection stability configuration
STABILITY_WINDOW = 10
MIN_STABLE_FRAMES = 6

# ---------- INITIALIZATION ----------

model = YOLO(MODEL_PATH)
camera = cv2.VideoCapture(CAMERA_URL)

state = {
    "camera_connected": camera.isOpened(),
    "esp32_connected": False,
    "objects": [],
    "object_count": 0,
    "distance_cm": None,
    "proximity": "UNKNOWN",
    "intelligent_alert": "UNKNOWN",
    "alert_message": "Waiting for camera and sensor readings.",
    "image_quality": {
        "brightness": None,
        "sharpness": None,
        "contrast": None,
        "quality_status": "UNKNOWN",
        "quality_issues": []
    },
    "detection_stability": {
        "score": None,
        "status": "UNKNOWN",
        "stable_classes": [],
        "unstable_classes": []
    },
    "reliability": {
        "status": "UNKNOWN",
        "reasons": []
    },
    "updated_at": None
}

latest_frame = None
alert_history = []
last_history_key = None

# Store class presence for each recent frame.
detection_window = deque(maxlen=STABILITY_WINDOW)

lock = threading.Lock()


# ---------- IMAGE QUALITY ANALYSIS ----------

def analyze_image_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    contrast = float(np.std(gray))

    issues = []

    if brightness < LOW_LIGHT_THRESHOLD:
        issues.append("LOW_LIGHT")

    if sharpness < BLUR_THRESHOLD:
        issues.append("POSSIBLE_BLUR")

    if contrast < LOW_CONTRAST_THRESHOLD:
        issues.append("LOW_CONTRAST")

    if "LOW_LIGHT" in issues:
        status = "LOW_LIGHT"
    elif "POSSIBLE_BLUR" in issues:
        status = "BLURRY"
    elif "LOW_CONTRAST" in issues:
        status = "LOW_CONTRAST"
    else:
        status = "GOOD"

    return {
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "quality_status": status,
        "quality_issues": issues
    }


# ---------- DETECTION STABILITY ANALYSIS ----------

def analyze_detection_stability(detected_objects):
    """
    Measures how often each class appears in recent frames.

    This is class-presence consistency, not physical object
    tracking. Two different objects of the same class can
    contribute to the same class-level score.
    """

    current_classes = {
        obj["class"] for obj in detected_objects
    }

    detection_window.append(current_classes)

    if len(detection_window) < STABILITY_WINDOW:
        return {
            "score": None,
            "status": "COLLECTING",
            "stable_classes": [],
            "unstable_classes": [],
            "frames_analyzed": len(detection_window),
            "window_size": STABILITY_WINDOW
        }

    all_classes = set()

    for frame_classes in detection_window:
        all_classes.update(frame_classes)

    if not all_classes:
        return {
            "score": 0.0,
            "status": "NO_DETECTIONS",
            "stable_classes": [],
            "unstable_classes": [],
            "frames_analyzed": len(detection_window),
            "window_size": STABILITY_WINDOW
        }

    class_scores = {}

    for class_name in all_classes:
        count = sum(
            1 for frame_classes in detection_window
            if class_name in frame_classes
        )

        class_scores[class_name] = (
            count / len(detection_window) * 100
        )

    stable_classes = [
        name for name, score in class_scores.items()
        if score >= (MIN_STABLE_FRAMES / STABILITY_WINDOW * 100)
    ]

    unstable_classes = [
        name for name, score in class_scores.items()
        if score < (MIN_STABLE_FRAMES / STABILITY_WINDOW * 100)
    ]

    # Mean presence consistency across classes observed
    # anywhere in the analysis window.
    overall_score = float(np.mean(list(class_scores.values())))

    if overall_score >= 80:
        status = "STABLE"
    elif overall_score >= 50:
        status = "INCONSISTENT"
    else:
        status = "UNSTABLE"

    return {
        "score": round(overall_score, 1),
        "status": status,
        "stable_classes": stable_classes,
        "unstable_classes": unstable_classes,
        "class_scores": {
            name: round(score, 1)
            for name, score in class_scores.items()
        },
        "frames_analyzed": len(detection_window),
        "window_size": STABILITY_WINDOW
    }


# ---------- PRELIMINARY RELIABILITY ASSESSMENT ----------

def assess_reliability(quality, stability):
    reasons = []

    quality_status = quality.get(
        "quality_status", "UNKNOWN"
    )

    stability_status = stability.get(
        "status", "UNKNOWN"
    )

    if quality_status == "UNKNOWN":
        reasons.append("Image quality is unavailable.")
    elif quality_status != "GOOD":
        reasons.append(
            f"Image quality issue: {quality_status}."
        )

    if stability_status == "COLLECTING":
        reasons.append(
            "Collecting enough frames to assess stability."
        )
    elif stability_status == "INCONSISTENT":
        reasons.append(
            "Object class presence is inconsistent across frames."
        )
    elif stability_status == "UNSTABLE":
        reasons.append(
            "Low class-presence consistency across recent frames."
        )
    elif stability_status == "NO_DETECTIONS":
        reasons.append(
            "No object classes detected in the analysis window."
        )
    elif stability_status == "UNKNOWN":
        reasons.append(
            "Detection stability is unavailable."
        )

    if quality_status == "UNKNOWN":
        reliability_status = "UNKNOWN"
    elif stability_status in ("COLLECTING", "UNKNOWN"):
        reliability_status = "ASSESSING"
    elif (
        quality_status == "GOOD"
        and stability_status == "STABLE"
    ):
        reliability_status = "RELIABLE"
    elif (
        quality_status != "GOOD"
        or stability_status in (
            "INCONSISTENT",
            "UNSTABLE",
            "NO_DETECTIONS"
        )
    ):
        reliability_status = "UNCERTAIN"
    else:
        reliability_status = "ASSESSING"

    return {
        "status": reliability_status,
        "reasons": reasons
    }


# ---------- INTELLIGENT PROXIMITY ALERT ----------

def create_intelligent_alert(
    objects,
    proximity,
    distance,
    camera_connected,
    esp32_connected,
    quality,
    reliability
):
    detected_names = sorted({
        obj["class"] for obj in objects
    })

    object_text = ", ".join(detected_names[:4])

    if len(detected_names) > 4:
        object_text += ", and other objects"

    if not esp32_connected or proximity == "UNKNOWN":
        alert_level = "UNKNOWN"
        message = (
            "Ultrasonic sensor unavailable. "
            "Check ESP32 connection and sensor readings."
        )

    else:
        distance_text = (
            f"{distance:.1f} cm"
            if distance is not None
            else "an unknown distance"
        )

        if proximity == "DANGER":
            alert_level = "DANGER"

            if camera_connected and detected_names:
                message = (
                    f"DANGER: Camera detects {object_text}. "
                    f"Ultrasonic sensor reports an obstacle "
                    f"at {distance_text}."
                )
            elif camera_connected:
                message = (
                    f"DANGER: Ultrasonic sensor reports an "
                    f"obstacle at {distance_text}, but YOLO "
                    f"has not confidently detected an object."
                )
            else:
                message = (
                    f"DANGER: Ultrasonic sensor reports an "
                    f"obstacle at {distance_text}. "
                    f"Camera is unavailable."
                )

        elif proximity == "WARNING":
            alert_level = "WARNING"

            if camera_connected and detected_names:
                message = (
                    f"WARNING: Camera detects {object_text}. "
                    f"Ultrasonic sensor reports an obstacle "
                    f"at {distance_text}."
                )
            elif camera_connected:
                message = (
                    f"WARNING: An obstacle is reported at "
                    f"{distance_text}, but YOLO has not "
                    f"confidently detected an object."
                )
            else:
                message = (
                    f"WARNING: An obstacle is reported at "
                    f"{distance_text}. Camera is unavailable."
                )

        elif proximity == "CLEAR":
            alert_level = "CLEAR"

            if not camera_connected:
                message = (
                    "Ultrasonic reading is clear, but the "
                    "camera is unavailable."
                )
            elif detected_names:
                message = (
                    f"CLEAR: Camera detects {object_text}, "
                    f"but the ultrasonic sensor reports no "
                    f"obstacle within {WARNING_DISTANCE:.0f} cm."
                )
            else:
                message = (
                    "CLEAR: No obstacle detected within the "
                    "ultrasonic warning range."
                )

        else:
            alert_level = "UNKNOWN"
            message = "Waiting for a valid sensor reading."

    # A quality warning is supplementary; it does not
    # override an ultrasonic DANGER or WARNING state.
    quality_status = quality.get("quality_status", "UNKNOWN")

    if quality_status not in ("GOOD", "UNKNOWN"):
        message += (
            f" Camera quality warning: {quality_status}."
        )

    reliability_status = reliability.get("status", "UNKNOWN")

    if reliability_status == "UNCERTAIN":
        message += (
            " Visual detection reliability is uncertain."
        )

    return alert_level, message


# ---------- CAMERA / YOLO WORKER ----------

def detection_worker():
    global latest_frame

    while True:
        ok, frame = camera.read()

        if not ok:
            with lock:
                state["camera_connected"] = False
                state["image_quality"] = {
                    "brightness": None,
                    "sharpness": None,
                    "contrast": None,
                    "quality_status": "UNKNOWN",
                    "quality_issues": []
                }

            time.sleep(1)
            continue

        quality = analyze_image_quality(frame)

        result = model.predict(
            frame,
            imgsz=640,
            conf=0.40,
            verbose=False
        )[0]

        detected = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            detected.append({
                "class": model.names[class_id],
                "confidence": round(confidence * 100, 1),
                "box": [x1, y1, x2, y2]
            })

        annotated = result.plot()

        with lock:
            stability = analyze_detection_stability(
                detected
            )

            reliability = assess_reliability(
                quality,
                stability
            )

            state["camera_connected"] = True
            state["objects"] = detected
            state["object_count"] = len(detected)
            state["image_quality"] = quality
            state["detection_stability"] = stability
            state["reliability"] = reliability
            state["updated_at"] = time.strftime("%H:%M:%S")

            latest_frame = annotated


# ---------- ESP32 SENSOR WORKER ----------

def sensor_worker():
    while True:
        try:
            response = requests.get(
                DISTANCE_URL,
                timeout=1
            )
            response.raise_for_status()

            distance = float(response.text.strip())

            with lock:
                state["esp32_connected"] = True

                if distance < 0:
                    state["distance_cm"] = None
                    state["proximity"] = "UNKNOWN"

                else:
                    state["distance_cm"] = round(distance, 1)

                    if distance <= DANGER_DISTANCE:
                        state["proximity"] = "DANGER"
                    elif distance <= WARNING_DISTANCE:
                        state["proximity"] = "WARNING"
                    else:
                        state["proximity"] = "CLEAR"

        except (requests.RequestException, ValueError):
            with lock:
                state["esp32_connected"] = False
                state["distance_cm"] = None
                state["proximity"] = "UNKNOWN"

        time.sleep(0.25)


# ---------- LIVE VIDEO STREAM ----------

def video_stream():
    while True:
        with lock:
            frame = latest_frame

            if frame is not None:
                frame = frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        ok, encoded = cv2.imencode(".jpg", frame)

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + encoded.tobytes()
            + b"\r\n"
        )


# ---------- API: STATUS + ALERT HISTORY ----------

@app.route("/api/status")
def api_status():
    global last_history_key

    with lock:
        objects = [
            dict(obj) for obj in state["objects"]
        ]

        quality = {
            **state["image_quality"],
            "quality_issues": list(
                state["image_quality"]["quality_issues"]
            )
        }

        stability = dict(state["detection_stability"])
        stability["stable_classes"] = list(
            stability.get("stable_classes", [])
        )
        stability["unstable_classes"] = list(
            stability.get("unstable_classes", [])
        )

        if "class_scores" in stability:
            stability["class_scores"] = dict(
                stability["class_scores"]
            )

        reliability = {
            **state["reliability"],
            "reasons": list(state["reliability"]["reasons"])
        }

        alert_level, alert_message = create_intelligent_alert(
            objects=objects,
            proximity=state["proximity"],
            distance=state["distance_cm"],
            camera_connected=state["camera_connected"],
            esp32_connected=state["esp32_connected"],
            quality=quality,
            reliability=reliability
        )

        object_names = sorted({
            obj["class"] for obj in objects
        })

        history_key = (
            alert_level,
            tuple(object_names)
            if alert_level in ("DANGER", "WARNING")
            else ()
        )

        if history_key != last_history_key:
            alert_history.insert(0, {
                "time": time.strftime("%H:%M:%S"),
                "level": alert_level,
                "message": alert_message
            })

            del alert_history[MAX_HISTORY:]
            last_history_key = history_key

        data = {
            **state,
            "objects": objects,
            "image_quality": quality,
            "detection_stability": stability,
            "reliability": reliability,
            "intelligent_alert": alert_level,
            "alert_message": alert_message,
            "alert_history": list(alert_history)
        }

    return jsonify(data)


# ---------- API: LIVE CAMERA FEED ----------

@app.route("/video_feed")
def video_feed():
    return Response(
        video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------- START SERVER ----------

if __name__ == "__main__":
    threading.Thread(
        target=detection_worker,
        daemon=True
    ).start()

    threading.Thread(
        target=sensor_worker,
        daemon=True
    ).start()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
