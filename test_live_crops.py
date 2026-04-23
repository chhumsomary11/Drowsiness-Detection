import cv2
import time
from live_crop_extractor import LiveCropExtractor

MODEL_PATH = "/Users/macbookpro/Desktop/crop/models/face_landmarker.task"

extractor = LiveCropExtractor(model_path=MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp_ms = int(time.time() * 1000)
        data = extractor.extract(frame, timestamp_ms)

        display = frame.copy()

        if data is not None:
            display = extractor.draw_debug(display, data)

            eyes = cv2.resize(data["eyes_crop"], (320, 120))
            mouth = cv2.resize(data["mouth_crop"], (220, 140))
            face = cv2.resize(data["face_crop"], (220, 220))

            cv2.imshow("eyes_crop", eyes)
            cv2.imshow("mouth_crop", mouth)
            cv2.imshow("face_crop", face)

        cv2.imshow("live_debug", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

finally:
    cap.release()
    extractor.close()
    cv2.destroyAllWindows()