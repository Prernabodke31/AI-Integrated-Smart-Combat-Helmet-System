"""
main.py
=======
Entry point for the Helmet-Based 360° Threat Detection & Soldier Monitoring System.

Orchestrates:
  • CameraHandler   – camera state machine + threat annotation
  • ThreatDetector  – YOLOv8n / YOLOv5n / ONNX / motion fallback
  • SensorHub       – MAX30102 heart-rate, MPU6050 IMU, NEO-6M GPS
  • TTS engine      – pyttsx3 (offline, no internet required)
  • Buzzer          – optional GPIO buzzer
  • Display loop    – single-camera feed with full overlay

Voice alert behaviour:
  • On threat/motion: "Look at your <direction>. <motion_type> detected
    at <distance> metres." repeats every 2.5 s while camera is in ALERT.
  • After ~3 s the camera automatically switches to the next direction.

Run:
  python main.py [--no-display] [--no-tts] [--no-buzzer]
"""

import argparse
import logging
import os
import queue
import signal
import sys
import threading
import time

import cv2

from camera_handler import CameraHandler, ThreatInfo, CAMERA_DIRECTIONS
from threat_detection import build_detector
from sensor_module import SensorHub, EmergencyEvent

# ──────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("helmet_system.log"),
    ],
)
logger = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────
# TTS engine (offline)
# ──────────────────────────────────────────────────────────────
class TTSEngine:
    """
    Wraps pyttsx3 in a dedicated worker thread to avoid blocking the main loop.
    Queued messages are spoken in order; excess messages are dropped so the
    voice never lags behind reality.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._queue   = queue.Queue(maxsize=2)   # keep queue short — drop stale alerts
        self._engine  = None
        if enabled:
            self._init_engine()
            t = threading.Thread(target=self._worker, daemon=True, name="tts")
            t.start()

    def _init_engine(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate",   140)   # slightly slower = clearer in field
            self._engine.setProperty("volume", 1.0)
            logger.info("TTS engine ready.")
        except Exception as exc:
            logger.warning("pyttsx3 init failed (%s) – TTS disabled.", exc)
            self._enabled = False

    def speak(self, text: str, priority: bool = False):
        """
        Queue a TTS message.
        priority=True clears the queue first so urgent alerts jump the line.
        """
        if not self._enabled:
            logger.info("[TTS-sim] %s", text)
            return
        if priority:
            # Drain any pending low-priority messages
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass   # drop — a fresher message will arrive in the next repeat cycle

    def _worker(self):
        while True:
            text = self._queue.get()
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as exc:
                logger.debug("TTS speak error: %s", exc)


# ──────────────────────────────────────────────────────────────
# Buzzer (GPIO – optional)
# ──────────────────────────────────────────────────────────────
class Buzzer:
    """Uses RPi.GPIO to drive a passive or active buzzer on a GPIO pin."""

    BUZZER_PIN = 18   # BCM numbering – change to your wiring

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._gpio    = None
        if enabled:
            self._init_gpio()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)
            logger.info("Buzzer ready on GPIO %d.", self.BUZZER_PIN)
        except Exception as exc:
            logger.warning("GPIO/Buzzer init failed (%s) – buzzer disabled.", exc)
            self._enabled = False
            self._gpio    = None

    def beep(self, duration: float = 0.3, pulses: int = 1):
        if not self._enabled or self._gpio is None:
            return
        def _do():
            try:
                for _ in range(pulses):
                    self._gpio.output(self.BUZZER_PIN, self._gpio.HIGH)
                    time.sleep(duration)
                    self._gpio.output(self.BUZZER_PIN, self._gpio.LOW)
                    if pulses > 1:
                        time.sleep(0.1)
            except Exception:
                pass
        threading.Thread(target=_do, daemon=True).start()

    def cleanup(self):
        if self._gpio:
            try:
                self._gpio.cleanup()
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────
# Alert message builder
# ──────────────────────────────────────────────────────────────
def build_threat_message(threat: ThreatInfo) -> str:
    """
    Compose the voice alert message.

    Examples:
      "Look at your front. Person detected at 4 metres."
      "Look at your left. Running detected at 8 metres."
      "Look at your right. Hand movement detected at 2 metres."
    """
    direction   = threat.direction.lower() if threat.direction else "unknown"
    motion_type = threat.motion_type if threat.motion_type else "Threat"
    distance    = threat.distance_m

    # Round to nearest half-metre for a more natural reading
    dist_str = f"{distance:.0f}" if distance == int(distance) else f"{distance:.1f}"

    return (
        f"Look at your {direction}. "
        f"{motion_type} detected at {dist_str} metres."
    )


# ──────────────────────────────────────────────────────────────
# Overlay rendering helpers
# ──────────────────────────────────────────────────────────────
def draw_sensor_overlay(frame, reading, emergency: bool):
    """Draw heart-rate, GPS, and fall status in bottom-left corner."""
    h, w = frame.shape[:2]
    lines = []

    hr = reading.heart_rate_bpm
    if hr is not None:
        color = (0, 0, 255) if hr < 50 else (0, 220, 120)
        lines.append((f"HR: {hr:.0f} BPM", color))
    else:
        lines.append(("HR: --", (180, 180, 180)))

    if reading.gps_fixed and reading.latitude is not None:
        lines.append((f"GPS: {reading.latitude:.4f}, {reading.longitude:.4f}",
                      (0, 200, 255)))
    else:
        lines.append(("GPS: no fix", (180, 180, 180)))

    if reading.fall_detected:
        lines.append(("FALL DETECTED!", (0, 0, 255)))

    if emergency:
        lines.append(("** EMERGENCY **", (0, 0, 255)))

    y0 = h - 10 - len(lines) * 22
    for i, (txt, col) in enumerate(lines):
        y = y0 + i * 22
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col,    1, cv2.LINE_AA)


def draw_emergency_banner(frame):
    """Red banner across the top for emergency mode."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
    cv2.putText(frame, "!!! SOLDIER EMERGENCY !!!",
                (w // 2 - 200, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────
# HelmetSystem – top-level orchestrator
# ──────────────────────────────────────────────────────────────
class HelmetSystem:
    def __init__(self, args):
        self._args          = args
        self._emergency     = False
        self._stop_event    = threading.Event()

        # Build subsystems
        logger.info("Initialising threat detector…")
        self.detector = build_detector()

        logger.info("Initialising TTS…")
        self.tts = TTSEngine(enabled=not args.no_tts)

        logger.info("Initialising buzzer…")
        self.buzzer = Buzzer(enabled=not args.no_buzzer)

        logger.info("Initialising camera handler…")
        self.cam = CameraHandler(
            detector       = self.detector,
            alert_callback = self._on_threat,   # called every VOICE_REPEAT_SEC during ALERT
        )

        logger.info("Initialising sensor hub…")
        self.sensors = SensorHub(emergency_callback=self._on_emergency)

        # Register OS signal handlers for clean shutdown
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    # ── Lifecycle ─────────────────────────────────────────────

    def run(self):
        logger.info("Starting all subsystems…")
        self.sensors.start()
        self.cam.start()

        self.tts.speak("Helmet system online. Monitoring active.")
        logger.info("System running. Press Ctrl+C to stop.")

        if not self._args.no_display:
            self._display_loop()
        else:
            while not self._stop_event.is_set():
                self._log_status()
                time.sleep(5)

    def stop(self):
        logger.info("Shutting down…")
        self._stop_event.set()
        self.cam.stop()
        self.sensors.stop()
        self.buzzer.cleanup()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")

    def _handle_signal(self, signum, frame):
        logger.info("Signal %d received – shutting down.", signum)
        self.stop()
        sys.exit(0)

    # ── Display loop ──────────────────────────────────────────

    def _display_loop(self):
        window = "Helmet 360 System"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, 800, 600)

        while not self._stop_event.is_set():
            frame = self.cam.get_annotated_frame()
            if frame is None:
                blank = __import__("numpy").zeros((480, 640, 3), dtype="uint8")
                cv2.putText(blank, "Waiting for camera…", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                cv2.imshow(window, blank)
            else:
                reading = self.sensors.get_reading()
                draw_sensor_overlay(frame, reading, self._emergency)
                if self._emergency:
                    draw_emergency_banner(frame)
                cv2.imshow(window, frame)

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == 27:
                logger.info("User quit via display window.")
                break

        self.stop()

    # ── Callbacks ─────────────────────────────────────────────

    def _on_threat(self, threat: ThreatInfo):
        """
        Called by CameraHandler:
          • Immediately when a threat is first detected (ALERT entry)
          • Then every VOICE_REPEAT_SEC while still in ALERT state
          • Camera automatically switches after ALERT_DURATION (~3 s)

        Voice pattern example (every 2.5 s for ~3 s):
          "Look at your front. Person detected at 4 metres."
          "Look at your front. Person detected at 3 metres."   ← distance may update
          [camera switches to next direction]
        """
        msg = build_threat_message(threat)
        logger.warning("THREAT ALERT: %s", msg)
        self.tts.speak(msg, priority=True)
        self.buzzer.beep(duration=0.25, pulses=2)

    def _on_emergency(self, event: EmergencyEvent):
        """Called by SensorHub when heart rate drops or fall detected."""
        self._emergency = True
        loc = (f"{event.latitude:.6f}, {event.longitude:.6f}"
               if event.latitude else "unknown location")
        msg = (f"Emergency! {event.reason}. "
               f"Soldier location: {loc}. Requesting rescue.")
        logger.critical("EMERGENCY: %s", msg)
        self.tts.speak(msg, priority=True)
        self.buzzer.beep(duration=0.1, pulses=5)

    # ── Headless status logger ────────────────────────────────

    def _log_status(self):
        status  = self.cam.get_status()
        reading = self.sensors.get_reading()
        logger.info(
            "Cam=%s | State=%s | HR=%s | GPS=%s | Fall=%s",
            status["direction"],
            status["state"],
            f"{reading.heart_rate_bpm:.0f}" if reading.heart_rate_bpm else "--",
            f"{reading.latitude:.4f},{reading.longitude:.4f}"
            if reading.gps_fixed else "no fix",
            reading.fall_detected,
        )


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Helmet 360° Threat Detection & Soldier Monitoring System"
    )
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no OpenCV window)")
    p.add_argument("--no-tts",     action="store_true",
                   help="Disable text-to-speech alerts")
    p.add_argument("--no-buzzer",  action="store_true",
                   help="Disable GPIO buzzer")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    system = HelmetSystem(args)
    try:
        system.run()
    except KeyboardInterrupt:
        system.stop()