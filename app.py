import cv2
import time
import base64
import torch
import threading
import numpy as np
from flask import Flask, render_template, jsonify
from collections import deque
from PIL import Image
import torchvision.transforms as transforms

from model import MultiBranchDrowsinessModel
from live_crop_extractor import LiveCropExtractor

WEIGHT_PATH     = "weights/model10.pth"
LANDMARKER_PATH = "models/face_landmarker.task"
IMG_SIZE        = 224
SMOOTH_WINDOW   = 15
ALERT_THRESHOLD = 0.20

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASS_NAMES  = ["notdrowsy", "drowsy"]
state        = {"raw_label": "–", "smooth_label": "–", "drowsy_prob": 0.0,
                "notdrowsy_prob": 0.0, "smooth_drowsy_prob": 0.0,
                "face_detected": False, "alert": False}
state_lock   = threading.Lock()
frame_lock   = threading.Lock()
latest_frame = None


def load_model():
    ckpt  = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    model = MultiBranchDrowsinessModel(branch_dim=ckpt["branch_dim"],
                                       num_classes=ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt.get("class_names", CLASS_NAMES)


def preprocess_crop(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return transform(Image.fromarray(rgb)).unsqueeze(0).to(device)


def inference_loop(model, class_names, extractor):
    global latest_frame
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    buf   = deque(maxlen=SMOOTH_WINDOW)
    ts    = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        ts = max(ts + 1, int(time.perf_counter() * 1000))
        data    = extractor.extract(frame, ts)
        display = frame.copy()

        if data is not None:
            with torch.no_grad():
                logits = model(preprocess_crop(data["eyes_crop"]),
                               preprocess_crop(data["mouth_crop"]),
                               preprocess_crop(data["face_crop"]))
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

            dp  = float(probs[1])
            ndp = float(probs[0])
            buf.append(dp)
            sp    = float(np.mean(buf))
            raw   = class_names[int(np.argmax(probs))]
            smooth = "drowsy" if sp >= ALERT_THRESHOLD else "notdrowsy"
            color  = (0, 0, 255) if smooth == "drowsy" else (0, 200, 80)

            display = extractor.draw_debug(display, data)
            cv2.putText(display, f"RAW: {raw} ({dp:.2f})",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 50), 2)
            cv2.putText(display, f"SMOOTH: {smooth} ({sp:.2f})",
                        (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            with state_lock:
                state.update({"raw_label": raw, "smooth_label": smooth,
                              "drowsy_prob": round(dp, 3), "notdrowsy_prob": round(ndp, 3),
                              "smooth_drowsy_prob": round(sp, 3),
                              "face_detected": True, "alert": smooth == "drowsy"})
        else:
            cv2.putText(display, "No face detected",
                        (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 255), 2)
            with state_lock:
                state.update({"face_detected": False, "alert": False,
                              "smooth_label": "–", "raw_label": "–"})

        _, jpeg = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/frame")
def frame_endpoint():
    with frame_lock:
        f = latest_frame
    if f is None:
        return jsonify({"image": None})
    return jsonify({"image": base64.b64encode(f).decode("utf-8")})


@app.route("/stats")
def stats():
    with state_lock:
        return jsonify(dict(state))


if __name__ == "__main__":
    model, class_names = load_model()
    extractor = LiveCropExtractor(model_path=LANDMARKER_PATH)
    threading.Thread(target=inference_loop,
                     args=(model, class_names, extractor), daemon=True).start()
    app.run(host="0.0.0.0", port=5001, debug=False)