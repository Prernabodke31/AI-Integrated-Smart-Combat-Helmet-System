"""
camera_handler.py
=================
Manages 4 USB cameras for 360° helmet-based threat detection.

Camera Mapping:
  Index 0 → Front
  Index 1 → Back
  Index 2 → Left
  Index 3 → Right

State Machine:
  MONITORING → (10s no threat)  → SWITCH_CAMERA
  MONITORING → (threat/motion)  → ALERT (2–3 s, repeating voice) → SWITCH_CAMERA

Motion Detection:
  Uses Lucas-Kanade optical flow to detect purposeful motion:
  walking, running, hand movements — ignores camera shake / static noise.
"""

import cv2
import time
import logging
import threading
import numpy as np
from voice_alert import speak_alert
from collections import deque
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Configuration constants  (edit here to tune behaviour)
# ──────────────────────────────────────────────────────────────
CAMERA_DIRECTIONS  = {0: "Front", 1: "Back", 2: "Left", 3: "Right"}
MONITOR_DURATION   = 10    # seconds per camera when no threat detected
ALERT_DURATION     = 3.0   # seconds to stay on alert then switch camera
VOICE_REPEAT_SEC   = 2.5   # repeat the voice message every N seconds while in ALERT
FRAME_WIDTH        = 640
FRAME_HEIGHT       = 480
CAMERA_FPS         = 15    # target FPS per active camera

# ── Motion detection tuning ───────────────────────────────────
MOTION_THRESHOLD   = 1500  # contour area (px²) to flag as motion (fallback)
MOTION_BLUR_KSIZE  = 21    # Gaussian blur kernel for frame-diff fallback

# Optical-flow motion classification thresholds
OF_MAX_CORNERS     = 150   # Shi-Tomasi corners to track
OF_QUALITY         = 0.3   # corner quality level
OF_MIN_DIST        = 7     # min distance between corners
OF_FLOW_MIN_MAG    = 2.0   # min flow magnitude (px) to count as real motion
OF_MOTION_RATIO    = 0.18  # fraction of tracked points that must be moving
OF_WALK_MAG        = 3.0   # flow magnitude → walking (≥ this)
OF_RUN_MAG         = 8.0   # flow magnitude → running (≥ this)
OF_HAND_SPREAD     = 60    # px spread between moving vectors → hand/erratic

# How many consecutive frames must show motion before we raise an alert
MOTION_CONFIRM_FRAMES = 3


# ──────────────────────────────────────────────────────────────
# State machine states
# ──────────────────────────────────────────────────────────────
class CamState(Enum):
    MONITORING = auto()   # reading frames, no threat
    ALERT      = auto()   # threat locked; repeating voice + blink overlay
    SWITCHING  = auto()   # transitional while opening next camera


@dataclass
class ThreatInfo:
    """Carries everything the display layer needs about a detected threat."""
    bbox:        Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    label:       str
    confidence:  float
    distance_m:  float
    direction:   str
    motion_type: str = "Unknown"             # "Walking", "Running", "Hand Movement", "Person"
    timestamp:   float = field(default_factory=time.time)


# ──────────────────────────────────────────────────────────────
# CameraHandler
# ──────────────────────────────────────────────────────────────
class CameraHandler:
    """
    Opens one camera at a time, rotates every MONITOR_DURATION seconds,
    and stays on a camera for ALERT_DURATION when a threat/motion is found,
    repeating the voice alert every VOICE_REPEAT_SEC during that window.

    Thread-safe: a background thread drives the state machine.
    The main thread calls get_annotated_frame() to receive the latest frame.
    """

    def __init__(self, detector, alert_callback=None):
        """
        Parameters
        ----------
        detector       : ThreatDetector instance (has .detect(frame) method)
        alert_callback : callable(ThreatInfo) invoked on each voice-repeat cycle
        """
        self.detector        = detector
        self.alert_callback  = alert_callback

        self._lock           = threading.Lock()
        self._stop_event     = threading.Event()

        # Current camera state
        self._cam_index      = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._state          = CamState.MONITORING
        self._state_started  = time.time()

        # Latest annotated frame (BGR numpy array)
        self._latest_frame:  Optional[np.ndarray] = None
        self._latest_threat: Optional[ThreatInfo] = None

        # Optical-flow state
        self._prev_gray:     Optional[np.ndarray] = None
        self._prev_pts:      Optional[np.ndarray] = None   # tracked points

        # Frame-diff fallback
        self._fd_prev_gray:  Optional[np.ndarray] = None

        # Motion confirmation buffer (avoid single-frame false positives)
        self._motion_buf:    deque = deque(maxlen=MOTION_CONFIRM_FRAMES)

        # Blink control (red circle overlay)
        self._blink_on       = False
        self._last_blink     = time.time()
        self._blink_interval = 0.4   # seconds

        # Voice repeat timer
        self._last_voice_t   = 0.0   # epoch time of last callback fire

        # Threat log
        self.threat_log: list = []

        logger.info("CameraHandler initialised. Detector: %s", type(detector).__name__)

    # ── Public API ────────────────────────────────────────────

    def start(self):
        """Open first camera and start the background processing thread."""
        self._open_camera(self._cam_index)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="cam-loop")
        self._thread.start()
        logger.info("CameraHandler started.")

    def stop(self):
        """Signal background thread to stop and release camera."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self._release_camera()
        logger.info("CameraHandler stopped.")

    def get_annotated_frame(self) -> Optional[np.ndarray]:
        """Return the most recent annotated frame (thread-safe copy)."""
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_status(self) -> dict:
        """Return a dict snapshot of current state for the UI."""
        with self._lock:
            threat = self._latest_threat
            return {
                "camera_index": self._cam_index,
                "direction":    CAMERA_DIRECTIONS.get(self._cam_index, "Unknown"),
                "state":        self._state.name,
                "threat":       threat,
            }

    # ── Internal helpers ──────────────────────────────────────

    def _open_camera(self, index: int):
        self._release_camera()
        logger.info("Opening camera %d (%s)…", index, CAMERA_DIRECTIONS.get(index))
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            logger.error("Cannot open camera %d – check USB connection.", index)
            cap = None
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        self._cap        = cap
        self._prev_gray  = None
        self._prev_pts   = None
        self._fd_prev_gray = None
        self._motion_buf.clear()

    def _release_camera(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _next_camera(self):
        self._cam_index = (self._cam_index + 1) % len(CAMERA_DIRECTIONS)
        logger.info("Switching to camera %d (%s).",
                    self._cam_index, CAMERA_DIRECTIONS.get(self._cam_index))
        self._state = CamState.SWITCHING

    # ── Main loop (runs in background thread) ─────────────────

    def _loop(self):
        while not self._stop_event.is_set():

            # ── Handle SWITCHING state ─────────────────────────
            if self._state == CamState.SWITCHING:
                self._open_camera(self._cam_index)
                self._state         = CamState.MONITORING
                self._state_started = time.time()
                self._last_voice_t  = 0.0
                with self._lock:
                    self._latest_threat = None
                continue

            # ── Read frame ────────────────────────────────────
            if self._cap is None or not self._cap.isOpened():
                logger.warning("Camera %d unavailable; attempting reopen…", self._cam_index)
                time.sleep(1)
                self._open_camera(self._cam_index)
                continue

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame from camera %d.", self._cam_index)
                time.sleep(0.05)
                continue

            elapsed   = time.time() - self._state_started
            direction = CAMERA_DIRECTIONS.get(self._cam_index, "Unknown")

            # ── MONITORING state ───────────────────────────────
            if self._state == CamState.MONITORING:
                threat = self._detect_threat(frame)

                if threat is not None:
                    threat.direction = direction
                    logger.info("THREAT on %s: %s (%s) @ %.1f m",
                                direction, threat.label, threat.motion_type, threat.distance_m)
                    self.threat_log.append({
                        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
                        "direction":   direction,
                        "label":       threat.label,
                        "motion_type": threat.motion_type,
                        "distance":    threat.distance_m,
                        "confidence":  threat.confidence,
                    })
                    with self._lock:
                        self._latest_threat = threat
                    self._state         = CamState.ALERT
                    self._state_started = time.time()
                    self._last_voice_t  = 0.0   # fire callback immediately

                    # First voice alert fires right away
                    self._fire_alert(threat)

                else:
                    if elapsed >= MONITOR_DURATION:
                        self._next_camera()
                        continue

                annotated = self._annotate_safe(frame, direction, elapsed)

            # ── ALERT state ────────────────────────────────────
            elif self._state == CamState.ALERT:
                with self._lock:
                    threat = self._latest_threat
                    # Refresh detection — update position/distance if still visible
                    new_threat = self._detect_threat(frame)
                    if new_threat is not None:
                        new_threat.direction = direction
                        self._latest_threat  = new_threat
                        threat               = new_threat

                # Repeat voice every VOICE_REPEAT_SEC
                now = time.time()
                if threat and (now - self._last_voice_t) >= VOICE_REPEAT_SEC:
                    self._fire_alert(threat)

                annotated = self._annotate_threat(frame, threat, elapsed)

                # After ALERT_DURATION, switch to next camera
                if elapsed >= ALERT_DURATION:
                    logger.info("Alert window ended on %s — switching camera.", direction)
                    self._next_camera()
                    continue

            with self._lock:
                self._latest_frame = annotated

            time.sleep(max(0, 1.0 / CAMERA_FPS - 0.005))

    # ── Alert callback helper ──────────────────────────────────

    def _fire_alert(self, threat: ThreatInfo):
        """Invoke the alert_callback and voice alert with correct direction."""
        self._last_voice_t = time.time()

    try:
        # 🔊 DIRECT VOICE (MAIN FIX)
        direction = threat.direction.lower()
        distance = threat.distance_m

        speak_alert(direction, distance)

        # Optional: keep your existing callback if used elsewhere
        if self.alert_callback:
            self.alert_callback(threat)

    except Exception as exc:
        logger.error("Alert error: %s", exc)

    # ── Threat detection ───────────────────────────────────────

    def _detect_threat(self, frame: np.ndarray) -> Optional[ThreatInfo]:
        """Run YOLO first, then optical-flow motion, then frame-diff fallback."""
        # 1) YOLO-based detection
        try:
            results = self.detector.detect(frame)
            if results:
                best = results[0]
                best.motion_type = "Person"
                return best
        except Exception as exc:
            logger.warning("Detector error: %s", exc)

        # 2) Optical-flow motion detection (walking / running / hand movement)
        motion = self._optical_flow_detect(frame)
        if motion is not None:
            return motion

        # 3) Frame-diff fallback (coarse, no classification)
        return self._frame_diff_detect(frame)

    # ── Optical-flow motion detector ──────────────────────────

    def _optical_flow_detect(self, frame: np.ndarray) -> Optional[ThreatInfo]:
        """
        Tracks Shi-Tomasi corners with Lucas-Kanade optical flow.
        Classifies motion type based on magnitude and spatial spread.
        Returns ThreatInfo only when MOTION_CONFIRM_FRAMES consecutive
        frames show meaningful motion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        result = None

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 8:
            # (Re-)seed corners
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners    = OF_MAX_CORNERS,
                qualityLevel  = OF_QUALITY,
                minDistance   = OF_MIN_DIST,
            )
            self._prev_gray = gray
            self._motion_buf.append(False)
            return None

        # Track points
        lk_params = dict(
            winSize   = (15, 15),
            maxLevel  = 2,
            criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **lk_params
        )

        if curr_pts is None or status is None:
            self._prev_gray = gray
            self._prev_pts  = None
            self._motion_buf.append(False)
            return None

        good_prev = self._prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) < 4:
            self._prev_gray = gray
            self._prev_pts  = None
            self._motion_buf.append(False)
            return None

        # Compute flow vectors
        flow   = good_curr - good_prev          # shape (N, 2)
        mags   = np.linalg.norm(flow, axis=1)  # magnitude per point

        # Count significantly moving points
        moving_mask = mags > OF_FLOW_MIN_MAG
        n_moving    = np.sum(moving_mask)
        ratio       = n_moving / len(mags)

        motion_detected = ratio >= OF_MOTION_RATIO
        self._motion_buf.append(motion_detected)

        # Only confirm if enough consecutive frames show motion
        if sum(self._motion_buf) >= MOTION_CONFIRM_FRAMES:
            moving_flow = flow[moving_mask]
            moving_mags = mags[moving_mask]
            moving_pts  = good_curr[moving_mask]

            avg_mag = float(np.mean(moving_mags))

            # Spread of moving points (tells us localised vs full-body)
            if len(moving_pts) > 1:
                spread = float(np.std(moving_pts[:, 0]) + np.std(moving_pts[:, 1]))
            else:
                spread = 0.0

            # Classify motion type
            if avg_mag >= OF_RUN_MAG:
                motion_type = "Running"
            elif avg_mag >= OF_WALK_MAG:
                motion_type = "Walking"
            elif spread < OF_HAND_SPREAD and avg_mag >= OF_FLOW_MIN_MAG:
                motion_type = "Hand Movement"
            else:
                motion_type = "Suspicious Motion"

            # Build bounding box around moving points
            if len(moving_pts) > 1:
                x_coords = moving_pts[:, 0].astype(int)
                y_coords = moving_pts[:, 1].astype(int)
                x1 = max(0, int(np.min(x_coords)) - 20)
                y1 = max(0, int(np.min(y_coords)) - 20)
                x2 = min(frame.shape[1], int(np.max(x_coords)) + 20)
                y2 = min(frame.shape[0], int(np.max(y_coords)) + 20)
            else:
                x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]

            h_px = max(1, y2 - y1)
            dist = self._estimate_distance(h_px, frame.shape[0])
            conf = min(0.95, 0.55 + ratio * 0.4)

            result = ThreatInfo(
                bbox        = (x1, y1, x2, y2),
                label       = "Motion",
                confidence  = conf,
                distance_m  = dist,
                direction   = "",
                motion_type = motion_type,
            )

        # Update prev state — periodically re-seed corners
        self._prev_gray = gray
        if len(good_curr) < 20:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray,
                maxCorners   = OF_MAX_CORNERS,
                qualityLevel = OF_QUALITY,
                minDistance  = OF_MIN_DIST,
            )
        else:
            self._prev_pts = good_curr.reshape(-1, 1, 2)

        return result

    # ── Frame-diff fallback ────────────────────────────────────

    def _frame_diff_detect(self, frame: np.ndarray) -> Optional[ThreatInfo]:
        """Simple frame differencing — last-resort fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (MOTION_BLUR_KSIZE, MOTION_BLUR_KSIZE), 0)

        if self._fd_prev_gray is None:
            self._fd_prev_gray = gray
            return None

        diff  = cv2.absdiff(self._fd_prev_gray, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        conts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self._fd_prev_gray = gray

        for c in conts:
            area = cv2.contourArea(c)
            if area > MOTION_THRESHOLD:
                x, y, w, h = cv2.boundingRect(c)
                dist = self._estimate_distance(h, frame.shape[0])
                return ThreatInfo(
                    bbox        = (x, y, x + w, y + h),
                    label       = "Motion",
                    confidence  = 0.55,
                    distance_m  = dist,
                    direction   = "",
                    motion_type = "Suspicious Motion",
                )
        return None

    # ── Distance heuristic ────────────────────────────────────

    @staticmethod
    def _estimate_distance(bbox_height_px: int, frame_height: int) -> float:
        if bbox_height_px <= 0:
            return 99.0
        ref_dist   = 1.7
        ref_height = frame_height * 0.85
        distance   = (ref_height / bbox_height_px) * ref_dist
        return round(min(max(distance, 0.5), 50.0), 1)

    # ── Frame annotation helpers ──────────────────────────────

    def _annotate_safe(self, frame: np.ndarray,
                       direction: str, elapsed: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        self._put_label(out, f"DIR: {direction}", (10, 30),
                        color=(0, 220, 0), scale=0.9, thickness=2)
        self._put_label(out, "STATUS: SAFE", (w - 220, 30),
                        color=(0, 220, 0), scale=0.9, thickness=2)
        remaining = max(0, MONITOR_DURATION - elapsed)
        bar_w = int((remaining / MONITOR_DURATION) * w)
        cv2.rectangle(out, (0, h - 8), (bar_w, h), (0, 200, 0), -1)
        return out

    def _annotate_threat(self, frame: np.ndarray,
                         threat: Optional[ThreatInfo], elapsed: float) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        direction = CAMERA_DIRECTIONS.get(self._cam_index, "Unknown")

        if threat is not None:
            x1, y1, x2, y2 = threat.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label_txt = (f"{threat.motion_type}  {threat.confidence:.0%} "
                         f"~{threat.distance_m:.1f} m")
            cv2.rectangle(out, (x1, y1 - 26), (x2, y1), (0, 0, 200), -1)
            cv2.putText(out, label_txt, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Blinking red indicator
        self._update_blink()
        if self._blink_on:
            cx = w // 2
            cv2.circle(out, (cx, 40), 22, (0, 0, 255), -1)
            cv2.putText(out, "!", (cx - 7, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        self._put_label(out, f"DIR: {direction}", (10, 30),
                        color=(0, 0, 255), scale=0.9, thickness=2)
        self._put_label(out, "STATUS: ALERT", (w - 240, 30),
                        color=(0, 0, 255), scale=0.9, thickness=2)
        if threat:
            self._put_label(out,
                            f"{threat.motion_type}  ~{threat.distance_m:.1f} m",
                            (10, h - 20),
                            color=(0, 0, 255), scale=0.8, thickness=2)

        # Countdown bar (red)
        remaining = max(0, ALERT_DURATION - elapsed)
        bar_w = int((remaining / ALERT_DURATION) * w)
        cv2.rectangle(out, (0, h - 8), (bar_w, h), (0, 0, 200), -1)
        return out

    def _update_blink(self):
        now = time.time()
        if now - self._last_blink >= self._blink_interval:
            self._blink_on   = not self._blink_on
            self._last_blink = now

    @staticmethod
    def _put_label(img, text, pos, color=(255, 255, 255),
                   scale=0.7, thickness=2):
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, pos,
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    thickness, cv2.LINE_AA)