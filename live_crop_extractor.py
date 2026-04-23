import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class LiveCropExtractor:
    def __init__(self, model_path: str, num_faces: int = 1):
        self.mp = mp
        self.vision = vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        # Landmark groups
        self.LEFT_EYE_IDX = [
            33, 246, 161, 160, 159, 158, 157, 173,
            133, 155, 154, 153, 145, 144, 163, 7
        ]
        self.RIGHT_EYE_IDX = [
            362, 398, 384, 385, 386, 387, 388, 466,
            263, 249, 390, 373, 374, 380, 381, 382
        ]
        self.MOUTH_IDX = [
            61, 146, 91, 181, 84, 17, 314, 405,
            321, 375, 291, 409, 270, 269, 267, 0,
            37, 39, 40, 185
        ]

    def close(self):
        self.landmarker.close()

    def _landmarks_to_pixels(self, landmarks, w, h):
        pts = []
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))
            pts.append((x, y))
        return np.array(pts, dtype=np.int32)

    def _bbox_from_indices(self, pts, indices, frame_shape, margin=0.15):
        region = pts[indices]
        x1 = np.min(region[:, 0])
        y1 = np.min(region[:, 1])
        x2 = np.max(region[:, 0])
        y2 = np.max(region[:, 1])

        bw = x2 - x1
        bh = y2 - y1

        pad_x = max(4, int(bw * margin))
        pad_y = max(4, int(bh * margin))

        H, W = frame_shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W - 1, x2 + pad_x)
        y2 = min(H - 1, y2 + pad_y)

        return (x1, y1, x2, y2)

    def _bbox_from_all_points(self, pts, frame_shape, margin=0.08):
        x1 = np.min(pts[:, 0])
        y1 = np.min(pts[:, 1])
        x2 = np.max(pts[:, 0])
        y2 = np.max(pts[:, 1])

        bw = x2 - x1
        bh = y2 - y1

        pad_x = max(6, int(bw * margin))
        pad_y = max(6, int(bh * margin))

        H, W = frame_shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(W - 1, x2 + pad_x)
        y2 = min(H - 1, y2 + pad_y)

        return (x1, y1, x2, y2)

    def _crop(self, frame, box):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        return crop

    def extract(self, frame_bgr, timestamp_ms: int):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        pts = self._landmarks_to_pixels(result.face_landmarks[0], w, h)

        left_eye_box = self._bbox_from_indices(
            pts, self.LEFT_EYE_IDX, frame_bgr.shape, margin=0.20
        )
        right_eye_box = self._bbox_from_indices(
            pts, self.RIGHT_EYE_IDX, frame_bgr.shape, margin=0.20
        )

        # merged eyes box
        ex1 = min(left_eye_box[0], right_eye_box[0])
        ey1 = min(left_eye_box[1], right_eye_box[1])
        ex2 = max(left_eye_box[2], right_eye_box[2])
        ey2 = max(left_eye_box[3], right_eye_box[3])
        eyes_box = (ex1, ey1, ex2, ey2)

        mouth_box = self._bbox_from_indices(
            pts, self.MOUTH_IDX, frame_bgr.shape, margin=0.25
        )
        face_box = self._bbox_from_all_points(
            pts, frame_bgr.shape, margin=0.08
        )

        eyes_crop = self._crop(frame_bgr, eyes_box)
        mouth_crop = self._crop(frame_bgr, mouth_box)
        face_crop = self._crop(frame_bgr, face_box)

        if eyes_crop.size == 0 or mouth_crop.size == 0 or face_crop.size == 0:
            return None

        return {
            "eyes_crop": eyes_crop,
            "mouth_crop": mouth_crop,
            "face_crop": face_crop,
            "eyes_box": eyes_box,
            "mouth_box": mouth_box,
            "face_box": face_box,
            "landmarks_xy": pts,
        }

    @staticmethod
    def draw_debug(frame_bgr, data):
        out = frame_bgr.copy()

        for box, color, name in [
            (data["eyes_box"], (255, 0, 0), "eyes"),
            (data["mouth_box"], (0, 0, 255), "mouth"),
            (data["face_box"], (0, 255, 0), "face"),
        ]:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out, name, (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

        return out