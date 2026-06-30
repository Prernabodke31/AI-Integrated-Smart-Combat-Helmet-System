"""
threat_detection.py
===================
Lightweight threat detection engine for Raspberry Pi 4 (CPU-only).

Priority order:
  1. YOLOv8n via Ultralytics (if installed)
  2. YOLOv5n via torch.hub (if PyTorch available)
  3. YOLOv5n ONNX via OpenCV DNN (fallback – no PyTorch needed)

Only 'person' class is flagged; motion hits are handled in camera_handler.py.
"""

import logging
import time
import cv2
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)

# COCO class IDs we consider threats
THREAT_CLASS_IDS   = {0}   # 0 = person in COCO
THREAT_CLASS_NAMES = {0: "Person"}

# Detection hyperparameters
CONF_THRESHOLD = 0.40   # minimum confidence to accept a detection
NMS_THRESHOLD  = 0.45   # IoU threshold for non-maximum suppression
INPUT_SIZE     = 320    # YOLO input size (320×320 is fastest on Pi)

# ONNX model download URL (YOLOv5n exported to ONNX)
ONNX_MODEL_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n.onnx"
)
ONNX_MODEL_PATH = "yolov5n.onnx"


# ─────────────────────────────────────────────────────────────
# Public factory function
# ─────────────────────────────────────────────────────────────
def build_detector() -> "ThreatDetector":
    """
    Try to load the best available backend and return a ThreatDetector.
    Always returns a working object; falls back gracefully.
    """
    # 1) Try Ultralytics YOLOv8n
    try:
        from ultralytics import YOLO  # noqa: F401
        logger.info("Using Ultralytics YOLOv8n backend.")
        return UltralyticsDetector("yolov8n.pt")
    except Exception as exc:
        logger.warning("YOLOv8 unavailable (%s), trying YOLOv5…", exc)

    # 2) Try YOLOv5n via torch.hub
    try:
        import torch  # noqa: F401
        logger.info("Using torch.hub YOLOv5n backend.")
        return TorchHubDetector()
    except Exception as exc:
        logger.warning("torch.hub YOLOv5 unavailable (%s), trying ONNX…", exc)

    # 3) ONNX via OpenCV DNN
    try:
        det = OnnxDetector(ONNX_MODEL_PATH)
        logger.info("Using ONNX/OpenCV DNN backend.")
        return det
    except Exception as exc:
        logger.error("ONNX backend failed (%s). Motion-only mode.", exc)
        return NullDetector()


# ─────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────
class ThreatDetector:
    """Abstract base – subclasses implement _raw_detect()."""

    def detect(self, frame: np.ndarray) -> List:
        """
        Run detection on a BGR frame.
        Returns a list of camera_handler.ThreatInfo objects, sorted by
        confidence descending.
        """
        from camera_handler import ThreatInfo
        try:
            raw = self._raw_detect(frame)
        except Exception as exc:
            logger.warning("Detection error: %s", exc)
            return []

        results = []
        for (x1, y1, x2, y2, conf, cls_id) in raw:
            if conf < CONF_THRESHOLD:
                continue
            if int(cls_id) not in THREAT_CLASS_IDS:
                continue
            h_px = max(1, y2 - y1)
            dist = self._estimate_distance(h_px, frame.shape[0])
            results.append(ThreatInfo(
                bbox       = (int(x1), int(y1), int(x2), int(y2)),
                label      = THREAT_CLASS_NAMES.get(int(cls_id), "Threat"),
                confidence = float(conf),
                distance_m = dist,
                direction  = "",   # filled by CameraHandler
            ))
        results.sort(key=lambda t: t.confidence, reverse=True)
        return results

    def _raw_detect(self, frame: np.ndarray):
        """Return list of (x1,y1,x2,y2,conf,cls_id)."""
        raise NotImplementedError

    @staticmethod
    def _estimate_distance(bbox_h_px: int, frame_h: int) -> float:
        """Same heuristic as camera_handler for consistency."""
        if bbox_h_px <= 0:
            return 99.0
        ref_dist   = 1.7
        ref_height = frame_h * 0.85
        distance   = (ref_height / bbox_h_px) * ref_dist
        return round(min(max(distance, 0.5), 50.0), 1)


# ─────────────────────────────────────────────────────────────
# Backend: Ultralytics YOLOv8n
# ─────────────────────────────────────────────────────────────
class UltralyticsDetector(ThreatDetector):
    def __init__(self, model_path: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._model.fuse()   # fuse BN+Conv for speed
        logger.info("YOLOv8n loaded from %s", model_path)

    def _raw_detect(self, frame: np.ndarray):
        results = self._model(
            frame,
            imgsz    = INPUT_SIZE,
            conf     = CONF_THRESHOLD,
            iou      = NMS_THRESHOLD,
            classes  = list(THREAT_CLASS_IDS),
            verbose  = False,
        )
        out = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0])
                out.append((x1, y1, x2, y2, conf, cls_id))
        return out


# ─────────────────────────────────────────────────────────────
# Backend: torch.hub YOLOv5n
# ─────────────────────────────────────────────────────────────
class TorchHubDetector(ThreatDetector):
    def __init__(self):
        import torch
        self._model = torch.hub.load(
            "ultralytics/yolov5", "yolov5n",
            pretrained=True, verbose=False,
        )
        self._model.conf = CONF_THRESHOLD
        self._model.iou  = NMS_THRESHOLD
        self._model.classes = list(THREAT_CLASS_IDS)
        self._model.eval()
        logger.info("YOLOv5n loaded via torch.hub")

    def _raw_detect(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._model(rgb, size=INPUT_SIZE)
        out = []
        for *xyxy, conf, cls_id in results.xyxy[0].tolist():
            out.append((*xyxy, conf, cls_id))
        return out


# ─────────────────────────────────────────────────────────────
# Backend: ONNX via OpenCV DNN (no PyTorch required)
# ─────────────────────────────────────────────────────────────
class OnnxDetector(ThreatDetector):
    """
    Loads a YOLOv5n ONNX model and runs inference with cv2.dnn.
    The ONNX must be the standard YOLOv5 export (output shape [1,N,85]).
    """

    def __init__(self, model_path: str):
        import os
        if not os.path.exists(model_path):
            self._download_model(model_path)
        self._net = cv2.dnn.readNetFromONNX(model_path)
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("ONNX model loaded: %s", model_path)

    @staticmethod
    def _download_model(path: str):
        import urllib.request
        logger.info("Downloading ONNX model… (first run only)")
        try:
            urllib.request.urlretrieve(ONNX_MODEL_URL, path)
            logger.info("ONNX model saved to %s", path)
        except Exception as exc:
            raise RuntimeError(f"Cannot download ONNX model: {exc}") from exc

    def _raw_detect(self, frame: np.ndarray):
        h0, w0 = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE),
            swapRB=True, crop=False,
        )
        self._net.setInput(blob)
        output = self._net.forward()   # shape [1, N, 85]

        scale_x = w0 / INPUT_SIZE
        scale_y = h0 / INPUT_SIZE
        boxes, confs, cls_ids = [], [], []

        for det in output[0]:
            obj_conf = float(det[4])
            if obj_conf < 0.3:
                continue
            cls_scores = det[5:]
            cls_id     = int(np.argmax(cls_scores))
            confidence = float(obj_conf * cls_scores[cls_id])
            if confidence < CONF_THRESHOLD:
                continue
            if cls_id not in THREAT_CLASS_IDS:
                continue
            cx, cy, bw, bh = det[:4]
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confs.append(confidence)
            cls_ids.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRESHOLD, NMS_THRESHOLD)
        out = []
        for i in (indices.flatten() if len(indices) > 0 else []):
            x, y, w, h = boxes[i]
            out.append((x, y, x + w, y + h, confs[i], cls_ids[i]))
        return out


# ─────────────────────────────────────────────────────────────
# Null detector (no-op; motion detection still runs in handler)
# ─────────────────────────────────────────────────────────────
class NullDetector(ThreatDetector):
    def _raw_detect(self, frame: np.ndarray):
        return []
