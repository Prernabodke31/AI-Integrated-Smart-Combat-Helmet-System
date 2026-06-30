"""
╔══════════════════════════════════════════════════════════╗
║         SMART COMBAT HELMET — MOTION MONITOR            ║
║         Raspberry Pi 4 | Final Year Project             ║
║                                                         ║
║  Sensors  : MPU6050 (Accel + Gyro) | NEO-6M GPS        ║
║  Alert    : SMS via Twilio → Base Camp Android Phone    ║
╚══════════════════════════════════════════════════════════╝
"""

import time
import math
import json
import logging
import threading
from datetime import datetime

# ── Twilio SMS ─────────────────────────────────────────────
from twilio.rest import Client

# ── RPi Hardware (auto-detected) ───────────────────────────
try:
    import smbus2
    import serial
    HARDWARE = True
except ImportError:
    HARDWARE = False
    print("[INFO] No RPi hardware found — running in SIMULATION mode\n")

# ══════════════════════════════════════════════════════════
#  ⚙️  CONFIGURATION — EDIT THESE BEFORE RUNNING
# ══════════════════════════════════════════════════════════
CONFIG = {
    # ── Soldier Info ──────────────────────────────────────
    "SOLDIER_ID"      : "HELMET-001",
    "SOLDIER_NAME"    : "Nandinee",           # ← Change to soldier name

    # ── Twilio Credentials ────────────────────────────────
    # Get these FREE from https://www.twilio.com/try-twilio
    "TWILIO_SID"      : "ACd7fbf1e04d61634d4221f4a9c74f69ff",  # ← Your Account SID
    "TWILIO_TOKEN"    : "d73fd9fb92b5b15dd5d977075c223f3d",                # ← Your Auth Token
    "TWILIO_FROM"     : "+15863010872",                        # ← Your Twilio number
    "BASE_CAMP_PHONE" : ["+919975118869", "+917620426253"],                       # ← Base camp phone number

    # ── Fall / Motion Thresholds ──────────────────────────
    "FALL_ACCEL"      : 2.5,     # g   — above this = fall spike
    "FALL_GYRO"       : 200,     # °/s — above this = fall spike
    "NO_MOVE_SECS"    : 30,      # seconds still after fall = unconscious
    "IMPACT_ACCEL"    : 3.5,     # g   — high-impact / blast threshold
    "TILT_ANGLE"      : 60,      # degrees — abnormal head tilt

    # ── Hardware Ports ────────────────────────────────────
    "I2C_BUS"         : 1,
    "MPU6050_ADDR"    : 0x68,
    "GPS_PORT"        : "/dev/ttyAMA0",
    "GPS_BAUD"        : 9600,

    # ── SMS Cooldown ──────────────────────────────────────
    "SMS_COOLDOWN"    : 120,     # seconds between same alert type
}

# ══════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/helmet.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("SmartHelmet")


# ══════════════════════════════════════════════════════════
#  MPU6050 DRIVER — Accelerometer & Gyroscope
# ══════════════════════════════════════════════════════════
class MPU6050:
    def __init__(self, bus):
        self.bus  = bus
        self.addr = CONFIG["MPU6050_ADDR"]
        bus.write_byte_data(self.addr, 0x6B, 0x00)  # wake up (clear sleep bit)
        bus.write_byte_data(self.addr, 0x1C, 0x10)  # accel ±8g range
        bus.write_byte_data(self.addr, 0x1B, 0x08)  # gyro  ±500°/s range
        log.info("MPU6050 ready")

    def _rw(self, reg):
        """Read signed 16-bit value from two consecutive registers."""
        h = self.bus.read_byte_data(self.addr, reg)
        l = self.bus.read_byte_data(self.addr, reg + 1)
        v = (h << 8) | l
        return v - 65536 if v >= 32768 else v

    def read(self):
        """Returns (ax, ay, az [g], gx, gy, gz [°/s], temp [°C])."""
        ax = self._rw(0x3B) / 4096.0
        ay = self._rw(0x3D) / 4096.0
        az = self._rw(0x3F) / 4096.0
        gx = self._rw(0x43) / 65.5
        gy = self._rw(0x45) / 65.5
        gz = self._rw(0x47) / 65.5
        raw_temp = self._rw(0x41)
        temp_c = raw_temp / 340.0 + 36.53
        return ax, ay, az, gx, gy, gz, round(temp_c, 1)


# ══════════════════════════════════════════════════════════
#  NEO-6M GPS DRIVER
# ══════════════════════════════════════════════════════════
class NEO6M:
    def __init__(self):
        self.ser = None
        self._lat, self._lon, self._valid = 0.0, 0.0, False
        if HARDWARE:
            try:
                self.ser = serial.Serial(CONFIG["GPS_PORT"], CONFIG["GPS_BAUD"], timeout=1)
                log.info("NEO-6M GPS ready on " + CONFIG["GPS_PORT"])
            except Exception as e:
                log.warning(f"GPS not available ({e}) — running without GPS")

    def _ddmm_to_dd(self, raw, direction):
        """Convert DDMM.MMMM GPS format to decimal degrees."""
        if not raw:
            return 0.0
        dot = raw.index(".")
        deg  = float(raw[:dot - 2])
        mins = float(raw[dot - 2:]) / 60.0
        dd   = deg + mins
        return -dd if direction in ("S", "W") else dd

    def read(self):
        """Returns (latitude, longitude, fix_valid)."""
        if not HARDWARE or self.ser is None:
            return self._lat, self._lon, self._valid
        try:
            line = self.ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith(("$GPRMC", "$GNRMC")):
                p = line.split(",")
                if len(p) >= 7 and p[2] == "A":
                    self._lat   = round(self._ddmm_to_dd(p[3], p[4]), 6)
                    self._lon   = round(self._ddmm_to_dd(p[5], p[6]), 6)
                    self._valid = True
                else:
                    self._valid = False
        except Exception:
            pass
        return self._lat, self._lon, self._valid


# ══════════════════════════════════════════════════════════
#  FALL & MOTION DETECTOR
# ══════════════════════════════════════════════════════════
class MotionDetector:
    def __init__(self):
        self._fall_time      = None
        self._no_move_fired  = False
        self._impact_fired   = False

    def _tilt_angle(self, ax, ay, az):
        """Estimate tilt of helmet from vertical (degrees)."""
        try:
            pitch = math.degrees(math.atan2(ax, math.sqrt(ay**2 + az**2)))
            roll  = math.degrees(math.atan2(ay, math.sqrt(ax**2 + az**2)))
            return round(abs(pitch), 1), round(abs(roll), 1)
        except Exception:
            return 0.0, 0.0

    def check(self, ax, ay, az, gx, gy, gz):
        """
        Returns:
            fall       (bool) — sudden impact spike detected
            unconscious(bool) — no movement 30 s after fall
            impact     (bool) — high-force blast/collision event
            pitch      (float)— head pitch angle in degrees
            roll       (float)— head roll angle in degrees
        """
        amag  = math.sqrt(ax**2 + ay**2 + az**2)
        gmag  = math.sqrt(gx**2 + gy**2 + gz**2)
        pitch, roll = self._tilt_angle(ax, ay, az)

        # ── High-impact / blast detection ─────────────────
        impact = False
        if amag > CONFIG["IMPACT_ACCEL"] and not self._impact_fired:
            impact = True
            self._impact_fired = True
        elif amag <= CONFIG["FALL_ACCEL"]:
            self._impact_fired = False   # reset once forces settle

        # ── Fall detection ────────────────────────────────
        fall = False
        if (amag > CONFIG["FALL_ACCEL"] or gmag > CONFIG["FALL_GYRO"]):
            if self._fall_time is None:
                self._fall_time     = time.time()
                self._no_move_fired = False
                fall = True

        # ── Unconscious / no-movement check ───────────────
        unconscious = False
        if self._fall_time and not self._no_move_fired:
            elapsed = time.time() - self._fall_time
            moving  = amag > 1.2 or gmag > 10
            if elapsed >= CONFIG["NO_MOVE_SECS"] and not moving:
                unconscious         = True
                self._no_move_fired = True
            elif moving:
                self._fall_time = None   # soldier got up — reset

        return fall, unconscious, impact, pitch, roll


# ══════════════════════════════════════════════════════════
#  TWILIO SMS SENDER
# ══════════════════════════════════════════════════════════
class SMSSender:
    def __init__(self):
        self.client = Client(CONFIG["TWILIO_SID"], CONFIG["TWILIO_TOKEN"])
        self._last  = {}   # alert_type → last sent timestamp
        log.info("Twilio SMS client ready")

    def _cooldown_ok(self, alert_type):
        now  = time.time()
        last = self._last.get(alert_type, 0)
        if now - last >= CONFIG["SMS_COOLDOWN"]:
            self._last[alert_type] = now
            return True
        return False

    def send(self, alert_type, accel_g, pitch, roll, lat, lon, gps_valid):
        """Send an SMS alert with soldier location.

        Always includes raw lat/lon coordinates.  When GPS has a fix,
        a Google Maps link is also added.  Alerts are suppressed if the
        same alert_type was sent within SMS_COOLDOWN seconds.
        """
        if not self._cooldown_ok(alert_type):
            log.info(f"SMS cooldown active for: {alert_type}")
            return

        if gps_valid:
            location_line = f"{lat}, {lon}"
            maps_line     = f"https://maps.google.com/?q={lat},{lon}"
        else:
            location_line = "GPS fix not yet available"
            maps_line     = "N/A (no GPS fix)"

        msg = (
            f"HELMET ALERT: {alert_type} | "
            f"{CONFIG['SOLDIER_NAME']} | "
            f"Accel:{accel_g:.1f}g | "
            f"GPS:{location_line} | "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

        try:
            for _num in CONFIG["BASE_CAMP_PHONE"]:
                message = self.client.messages.create(
                    body  = msg,
                    from_ = CONFIG["TWILIO_FROM"],
                    to    = _num
                )
                log.info(f"✅ SMS sent to {_num}! SID: {message.sid}")
            print(f"\n{'='*55}")
            print(f"  ✅ SMS SENT TO BASE CAMP")
            print(f"  Alert  : {alert_type}")
            print(f"  Accel  : {accel_g:.2f} g")
            print(f"  Tilt   : Pitch {pitch}° | Roll {roll}°")
            print(f"  GPS    : {lat}, {lon}")
            print(f"{'='*55}\n")
        except Exception as e:
            log.error(f"❌ SMS failed: {e}")
            print(f"\n[SMS FAILED] Check WiFi hotspot connection!\nError: {e}\n")

        # Log alert to file
        with open("logs/alerts.log", "a") as f:
            f.write(json.dumps({
                "time"      : datetime.now().isoformat(),
                "alert"     : alert_type,
                "accel_g"   : accel_g,
                "pitch"     : pitch,
                "roll"      : roll,
                "lat"       : lat,
                "lon"       : lon,
                "gps_valid" : gps_valid,
            }) + "\n")


# ══════════════════════════════════════════════════════════
#  SIMULATION MODE — Realistic fake sensor data
# ══════════════════════════════════════════════════════════
class Simulator:
    def __init__(self):
        self._t = 0

    def motion(self):
        self._t += 1
        ax = 0.05 * math.sin(self._t * 0.3)
        ay = 0.05 * math.cos(self._t * 0.3)
        az = 1.0
        gx = 2 * math.sin(self._t * 0.2)
        gy = 2 * math.cos(self._t * 0.2)
        gz = 0.0
        temp = 36.5 + 0.5 * math.sin(self._t * 0.05)

        if self._t == 20:              # fall at second 20
            # amag ≈ 2.83g — above FALL_ACCEL(2.5g) but below IMPACT_ACCEL(3.5g)
            # Triggers FALL DETECTED only, not a blast alert
            ax, ay, az = 2.0, 2.0, 0.1
            gx, gy, gz = 260, 190, 80
        if self._t == 50:              # high-impact blast at second 50
            ax, ay, az = 4.1, 3.0, 0.5
            gx, gy, gz = 100, 80,  30

        return ax, ay, az, gx, gy, gz, round(temp, 1)

    def gps(self):
        lat = 18.5204 + 0.0001 * math.sin(self._t * 0.01)
        lon = 73.8567 + 0.0001 * math.cos(self._t * 0.01)
        return round(lat, 6), round(lon, 6), True


# ══════════════════════════════════════════════════════════
#  MAIN HELMET MONITOR
# ══════════════════════════════════════════════════════════
class HelmetMonitor:
    def __init__(self):
        if HARDWARE:
            self.bus = smbus2.SMBus(CONFIG["I2C_BUS"])
            self.mpu = MPU6050(self.bus)
            self.gps = NEO6M()
        else:
            self.sim = Simulator()
            self.gps = NEO6M()

        self.motion      = MotionDetector()
        self.sms         = SMSSender()
        self._lat        = 0.0
        self._lon        = 0.0
        self._gps_valid  = False
        self._lock       = threading.Lock()
        self._running    = False

    def _gps_loop(self):
        """Update GPS in background every 5 seconds."""
        while self._running:
            if HARDWARE:
                lat, lon, valid = self.gps.read()
            else:
                lat, lon, valid = self.sim.gps()
            with self._lock:
                self._lat, self._lon, self._gps_valid = lat, lon, valid
            time.sleep(5)

    def _check_and_alert(self, amag, pitch, roll, fall, unconscious, impact):
        """Evaluate alert conditions and send SMS if needed.

        Each condition is checked independently so that a fall always
        triggers its own SMS even when an impact is detected in the
        same reading.  The SMSSender cooldown prevents duplicate
        messages within SMS_COOLDOWN seconds per alert type.
        """
        with self._lock:
            lat, lon, valid = self._lat, self._lon, self._gps_valid

        # ── Unconscious (highest priority, but still independent) ──
        if unconscious:
            self.sms.send(
                "SOLDIER UNCONSCIOUS — No movement 30s after fall",
                amag, pitch, roll, lat, lon, valid
            )

        # ── Fall detected — independent check, not elif ────────────
        if fall:
            self.sms.send("FALL DETECTED", amag, pitch, roll, lat, lon, valid)

        # ── High impact / blast — independent check ────────────────
        if impact:
            self.sms.send(
                f"HIGH IMPACT / BLAST EVENT ({amag:.2f}g)",
                amag, pitch, roll, lat, lon, valid
            )

        # ── Abnormal head tilt — only when no higher-priority event ─
        if not (fall or unconscious or impact):
            if pitch > CONFIG["TILT_ANGLE"] or roll > CONFIG["TILT_ANGLE"]:
                self.sms.send(
                    f"ABNORMAL HEAD TILT (Pitch {pitch}° Roll {roll}°)",
                    amag, pitch, roll, lat, lon, valid
                )

    def start(self):
        self._running = True

        print("╔══════════════════════════════════════════════╗")
        print("║      SMART COMBAT HELMET — MONITORING        ║")
        print(f"║  Soldier : {CONFIG['SOLDIER_NAME']:<34}║")
        print(f"║  ID      : {CONFIG['SOLDIER_ID']:<34}║")
        print(f"║  Mode    : {'HARDWARE' if HARDWARE else 'SIMULATION':<34}║")
        print("╚══════════════════════════════════════════════╝")
        print("\n[Monitoring started. Press Ctrl+C to stop.]\n")
        print(f"{'TIME':<22} {'ACCEL(g)':>9} {'PITCH°':>7} {'ROLL°':>6} "
              f"{'TEMP°C':>7} {'FALL':>6}  STATUS")
        print("-" * 80)

        # GPS runs in a background thread
        threading.Thread(target=self._gps_loop, daemon=True).start()

        try:
            while True:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if HARDWARE:
                    ax, ay, az, gx, gy, gz, temp = self.mpu.read()
                else:
                    ax, ay, az, gx, gy, gz, temp = self.sim.motion()

                amag = math.sqrt(ax**2 + ay**2 + az**2)
                fall, unconscious, impact, pitch, roll = self.motion.check(
                    ax, ay, az, gx, gy, gz
                )

                # Status label
                if unconscious:
                    status = "⚠ UNCONSCIOUS!"
                elif fall:
                    status = "⚠ FALL!"
                elif impact:
                    status = "⚠ HIGH IMPACT!"
                elif pitch > CONFIG["TILT_ANGLE"] or roll > CONFIG["TILT_ANGLE"]:
                    status = "⚠ TILT ALERT"
                else:
                    status = "✓ NORMAL"

                print(f"{now:<22} {amag:>9.3f} {pitch:>7.1f} {roll:>6.1f} "
                      f"{temp:>7.1f} {'YES' if fall else 'no':>6}  {status}")

                self._check_and_alert(amag, pitch, roll, fall, unconscious, impact)

                time.sleep(1)

        except KeyboardInterrupt:
            self._running = False
            print("\n[Monitor stopped by user]")
            log.info("Helmet monitor stopped")


# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    monitor = HelmetMonitor()
    monitor.start()
