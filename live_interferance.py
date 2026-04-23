import cv2
import time
import torch
import numpy as np
from collections import deque
from PIL import Image
import torchvision.transforms as transforms

from model import MultiBranchDrowsinessModel
from live_crop_extractor import LiveCropExtractor

# =========================
# CONFIG
# =========================
# CHANGE THESE PATHS TO YOUR OWN
WEIGHT_PATH = "/Users/macbookpro/Desktop/crop/weights/model10.pth"
LANDMARKER_PATH = "/Users/macbookpro/Desktop/crop/models/face_landmarker.task"
IMG_SIZE = 224
SMOOTH_WINDOW = 15
ALERT_THRESHOLD = 0.60   # average drowsy prob threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

CLASS_NAMES = ["notdrowsy", "drowsy"]


def preprocess_crop(crop_bgr):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor


def load_model():
    checkpoint = torch.load(WEIGHT_PATH, map_location=device)

    model = MultiBranchDrowsinessModel(
        branch_dim=checkpoint["branch_dim"],
        num_classes=checkpoint["num_classes"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_names = checkpoint.get("class_names", CLASS_NAMES)
    return model, class_names


def main():
    model, class_names = load_model()
    extractor = LiveCropExtractor(model_path=LANDMARKER_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    drowsy_prob_buffer = deque(maxlen=SMOOTH_WINDOW)
    last_timestamp_ms = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            data = extractor.extract(frame, timestamp_ms)

            display = frame.copy()

            if data is not None:
                eyes_tensor = preprocess_crop(data["eyes_crop"])
                mouth_tensor = preprocess_crop(data["mouth_crop"])
                face_tensor = preprocess_crop(data["face_crop"])

                with torch.no_grad():
                    logits = model(eyes_tensor, mouth_tensor, face_tensor)
                    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

                raw_pred_idx = int(np.argmax(probs))
                raw_label = class_names[raw_pred_idx]
                drowsy_prob = float(probs[1])
                notdrowsy_prob = float(probs[0])

                drowsy_prob_buffer.append(drowsy_prob)
                smooth_drowsy_prob = float(np.mean(drowsy_prob_buffer))

                smooth_label = "drowsy" if smooth_drowsy_prob >= ALERT_THRESHOLD else "notdrowsy"
                color = (0, 0, 255) if smooth_label == "drowsy" else (0, 255, 0)

                display = extractor.draw_debug(display, data)

                cv2.putText(display, f"RAW: {raw_label} ({drowsy_prob:.2f})",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                cv2.putText(display, f"SMOOTH: {smooth_label} ({smooth_drowsy_prob:.2f})",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.putText(display, f"notdrowsy={notdrowsy_prob:.2f} drowsy={drowsy_prob:.2f}",
                            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                eyes_preview = cv2.resize(data["eyes_crop"], (320, 120))
                mouth_preview = cv2.resize(data["mouth_crop"], (240, 140))
                face_preview = cv2.resize(data["face_crop"], (240, 240))

                cv2.imshow("eyes_crop", eyes_preview)
                cv2.imshow("mouth_crop", mouth_preview)
                cv2.imshow("face_crop", face_preview)

            else:
                cv2.putText(display, "No face detected",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("live_inference", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    finally:
        cap.release()
        extractor.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()