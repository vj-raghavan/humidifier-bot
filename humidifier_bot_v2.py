#!/usr/bin/env python3
"""
Smart Humidifier Bot v2 — Humidity-Aware Control

Reads humidity from ThermoPro display via camera + local vision LLM (Qwen2.5-VL),
then controls humidifier to maintain target range for curry leaf plant.

Schedule: 7 AM – 8 PM
Check interval: 5 minutes
Hysteresis: ON below LOW%, OFF above HIGH%, hold between
Safety: 3 consecutive failed reads → shut off
P0: fail-safe OFF on start/stop, max ON duration, OCR plausibility, single-instance lock
P1: shortcuts CLI, unique frame files, wall-clock sleep, ON-trend verify
"""

import atexit
import base64
import fcntl
import json
import logging
import logging.handlers
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import requests

# --- Configuration (from .env, then process environment) ---
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def _env(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


_load_dotenv(os.environ.get("HUMIDIFIER_ENV_FILE", os.path.join(_PKG_DIR, ".env")))

NTFY_TOPIC = _env("NTFY_TOPIC", "your-ntfy-topic")
NTFY_URL = _env("NTFY_URL", f"https://ntfy.sh/{NTFY_TOPIC}")
SHORTCUT_ON_NAME = _env("SHORTCUT_ON_NAME", "PH On")
SHORTCUT_OFF_NAME = _env("SHORTCUT_OFF_NAME", "PH Off")

HUMIDITY_LOW = _env_int("HUMIDITY_LOW", 50)
HUMIDITY_HIGH = _env_int("HUMIDITY_HIGH", 60)

START_HOUR = _env_int("START_HOUR", 7)
END_HOUR = _env_int("END_HOUR", 20)

CHECK_INTERVAL = _env_int("CHECK_INTERVAL", 60)

RTSP_URL = _env("RTSP_URL", "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1")
FFMPEG_TIMEOUT = _env_int("FFMPEG_TIMEOUT", 15)
FFMPEG_BIN = _env("FFMPEG_BIN", "/opt/homebrew/bin/ffmpeg")
# ffmpeg crop=w:h:x:y  (empty = full frame). Remote OUT/CH1 on the 2304x1296 stream.
FFMPEG_CROP = _env("FFMPEG_CROP", "")

VISION_API_BASE = _env("VISION_API_BASE", "http://YOUR_LLM_HOST:1234/v1")
VISION_MODEL = _env("VISION_MODEL", "qwen/qwen3-vl-4b")
VISION_API_KEY = _env("VISION_API_KEY", "lm-studio")
# Back-compat alias used by the reader
VISION_MODEL_4B = VISION_MODEL

VISION_PROMPT = """This image is a cropped close-up of a ThermoPro display showing the REMOTE sensor (labeled OUT / CH1).
The large digits for the remote sensor are temperature in Celsius (upper part) and humidity with a % symbol (lower part).

If the bottom edge of the crop happens to show the top of the indoor/local (IN) section, IGNORE IT completely. 
Extract ONLY the remote (OUT/CH1) readings. Do not extract or invent local readings.

Reply with ONLY this JSON - no other text, no markdown:
{"remote_humidity": <humidity number>, "remote_temp": <temperature number>, "local_humidity": null, "local_temp": null}
Use null if you cannot read a value clearly."""

MAX_CONSECUTIVE_FAILURES = _env_int("MAX_CONSECUTIVE_FAILURES", 3)
MAX_ON_WITHOUT_READ_SECS = _env_int("MAX_ON_WITHOUT_READ_SECS", 300)
MAX_ON_SECS = _env_int("MAX_ON_SECS", 1800)  # consecutive ON cap even with good reads
MAX_ON_COOLDOWN_SECS = _env_int("MAX_ON_COOLDOWN_SECS", 600)

HUMIDITY_PLAUSIBLE_MIN = _env_int("HUMIDITY_PLAUSIBLE_MIN", 10)
HUMIDITY_PLAUSIBLE_MAX = _env_int("HUMIDITY_PLAUSIBLE_MAX", 95)
MAX_HUMIDITY_JUMP = _env_int("MAX_HUMIDITY_JUMP", 15)
REMOTE_LOCAL_MAX_DELTA = _env_int("REMOTE_LOCAL_MAX_DELTA", 40)

TEST_MODE = _env_bool("TEST_MODE", False)
SHORTCUTS_BIN = _env("SHORTCUTS_BIN", "/usr/bin/shortcuts")
SHORTCUT_TIMEOUT = _env_int("SHORTCUT_TIMEOUT", 30)
ON_VERIFY_CHECKS = _env_int("ON_VERIFY_CHECKS", 3)
ON_VERIFY_MIN_RISE = _env_int("ON_VERIFY_MIN_RISE", 2)
LOCK_FILE = os.path.join(_PKG_DIR, "humidifier_bot.lock")

# --- Logging ---
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "humidifier_v2.log")

logger = logging.getLogger("humidifier_v2")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

_fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

# --- Humidity reading history ---
HISTORY_FILE = os.path.join(LOG_DIR, "humidity_history.json")


def load_history():
    """Load humidity reading history."""
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"readings": [], "last_state": None}


def save_history(history):
    """Save humidity reading history."""
    # Keep last 288 readings (~24h at 5min intervals)
    history["readings"] = history["readings"][-288:]
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def capture_frame():
    """Capture a single frame from the RTSP camera stream into a unique temp file."""
    fd, tmp_path = tempfile.mkstemp(prefix="humidifier_frame_", suffix=".jpg")
    os.close(fd)
    try:
        result = subprocess.run(
            [
                FFMPEG_BIN, "-y",
                "-rtsp_transport", "tcp",
                "-i", RTSP_URL,
                *(["-vf", f"crop={FFMPEG_CROP}"] if FFMPEG_CROP else []),
                "-frames:v", "1",
                "-q:v", "2",
                "-update", "1",
                tmp_path
            ],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT
        )
        size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        if size > 10000:
            logger.info(f"Frame captured: {size} bytes")
            return tmp_path
        err = (result.stderr or "").replace(RTSP_URL, "rtsp://REDACTED")
        err_tail = " | ".join(line.strip() for line in err.strip().splitlines()[-3:] if line.strip())
        logger.warning(
            f"Frame capture failed (rc={result.returncode}, bytes={size}): {err_tail or 'no ffmpeg stderr'}"
        )
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out capturing frame")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None
    except Exception as e:
        logger.error(f"Frame capture error: {e}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None


def read_humidity_from_image(image_path, model_name=VISION_MODEL_4B, max_retries=3, retry_delay=10):
    """Use local vision LLM (OpenAI-compatible API) to read humidity from ThermoPro display.
    Retries on transient failures (e.g. model still loading)."""
    try:
        with open(image_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Failed to read image file: {e}")
        return None

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(
                f"{VISION_API_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {VISION_API_KEY}"},
                timeout=60
            )

            if r.status_code != 200:
                logger.warning(f"Vision API returned {r.status_code} (attempt {attempt}/{max_retries}): {r.text[:200]}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                return None

            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()

            # Extract JSON from response (handle possible markdown wrapping)
            # Strip markdown code fences if present
            text = re.sub(r'^```json\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                reading = json.loads(json_match.group())
                remote_humidity = reading.get("remote_humidity")
                if remote_humidity is not None and isinstance(remote_humidity, (int, float)):
                    logger.info(
                        f"Reading: remote_humidity={remote_humidity}%, "
                        f"remote_temp={reading.get('remote_temp')}, "
                        f"local_humidity={reading.get('local_humidity')}, "
                        f"local_temp={reading.get('local_temp')}"
                    )
                    return reading
                else:
                    logger.warning(f"No valid humidity in response: {text}")
                    return None
            else:
                logger.warning(f"Could not parse vision response: {text}")
                return None

        except requests.Timeout:
            logger.warning(f"Vision API request timed out (attempt {attempt}/{max_retries})")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            return None
        except Exception as e:
            logger.error(f"Vision API error (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            return None

    return None


def run_shortcut(shortcut_name):
    """Run a macOS Shortcut via `shortcuts run` (exit status is the actuator check)."""
    if TEST_MODE:
        logger.info(f"[TEST] Would run shortcut: '{shortcut_name}'")
        return True
    try:
        logger.info(f"Running shortcut: '{shortcut_name}'")
        result = subprocess.run(
            [SHORTCUTS_BIN, "run", shortcut_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=SHORTCUT_TIMEOUT,
        )
        if result.stderr:
            logger.info(f"Shortcut stderr: {result.stderr.strip()[:300]}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Shortcut '{shortcut_name}' failed (rc={e.returncode}): {(e.stderr or e.stdout or '')[:300]}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Shortcut '{shortcut_name}' timed out after {SHORTCUT_TIMEOUT}s")
        return False
    except FileNotFoundError:
        logger.error(f"'{SHORTCUTS_BIN}' not found — install Shortcuts / macOS Monterey+")
        return False


def sleep_seconds(seconds):
    """Sleep by wall clock, in short slices so SIGTERM can fail-safe OFF.

    After Mac sleep, a leftover wait ends once; we log if we woke late instead of
    stacking catch-up cycles.
    """
    if seconds <= 0 or _runtime["stopping"]:
        return
    deadline = time.time() + seconds
    while not _runtime["stopping"]:
        remaining = deadline - time.time()
        if remaining <= 0:
            late = time.time() - deadline
            if late > 5:
                logger.info(f"Wait ended {int(late)}s late (machine likely slept) — continuing once")
            return
        time.sleep(min(remaining, 15))


def send_ntfy(title, message, tags="potted_plant", priority="default"):
    """Send a push notification via ntfy."""
    try:
        # Use ASCII-safe title (emojis go in tags)
        safe_title = title.encode("ascii", errors="ignore").decode("ascii").strip()
        r = requests.post(
            NTFY_URL,
            data=message.encode("utf-8"),
            headers={
                "Title": safe_title or "Humidifier Bot",
                "Tags": tags,
                "Priority": priority,
            },
            timeout=10,
        )
        if r.status_code == 200:
            logger.info(f"ntfy sent: {title}")
        else:
            logger.warning(f"ntfy returned {r.status_code}")
    except Exception as e:
        logger.warning(f"ntfy error: {e}")


def set_humidifier(state, current_state, reading=None):
    """Turn humidifier ON or OFF if not already in that state."""
    if state == current_state:
        logger.info(f"Humidifier already {state}, no action needed")
        return state

    if state == "ON":
        success = run_shortcut(SHORTCUT_ON_NAME)
    else:
        success = run_shortcut(SHORTCUT_OFF_NAME)

    if success:
        logger.info(f"Humidifier → {state}")
        # Build message with humidity reading
        if reading:
            rh = reading.get("remote_humidity", "?")
            rt = reading.get("remote_temp", "?")
            lh = reading.get("local_humidity", "?")
            lt = reading.get("local_temp", "?")
            msg = (f"Humidifier turned {state}\n"
                   f"Sensor: {rh}% RH / {rt}°C\n"
                   f"Local: {lh}% RH / {lt}°C")
        else:
            msg = f"Humidifier turned {state}"
        send_ntfy(
            f"Humidifier {state}",
            msg,
            tags="potted_plant,droplet" if state == "ON" else "potted_plant,no_entry",
        )
        return state
    else:
        logger.error(f"Failed to set humidifier to {state}")
        send_ntfy("Humidifier Error", f"Failed to switch humidifier to {state}", tags="warning", priority="high")
        return current_state


def decide_action(humidity, current_state):
    """Decide whether to turn humidifier ON/OFF based on humidity."""
    if humidity < HUMIDITY_LOW:
        return "ON", f"Humidity {humidity}% < {HUMIDITY_LOW}% → ON"
    elif humidity > HUMIDITY_HIGH:
        return "OFF", f"Humidity {humidity}% > {HUMIDITY_HIGH}% → OFF"
    else:
        state = current_state if current_state else "OFF"
        return state, f"Humidity {humidity}% in range ({HUMIDITY_LOW}-{HUMIDITY_HIGH}%) → hold {state}"


def last_good_humidity(history):
    """Recent remote RH, skipping isolated jumps that slipped into history."""
    entries = [
        e for e in (history.get("readings") or [])
        if isinstance(e.get("remote_humidity"), (int, float))
    ]
    for i in range(len(entries) - 1, -1, -1):
        rh = entries[i]["remote_humidity"]
        t = entries[i].get("time")
        if t:
            try:
                age = time.time() - datetime.fromisoformat(t).timestamp()
            except ValueError:
                age = 0
            if age > CHECK_INTERVAL * 5:
                return None
        if i > 0 and abs(rh - entries[i - 1]["remote_humidity"]) > MAX_HUMIDITY_JUMP:
            continue
        return rh
    return None


def reading_is_plausible(reading, prev_humidity):
    """Reject OCR nonsense. Returns (ok, reason). Local sensor is optional."""
    rh = reading.get("remote_humidity")
    if not isinstance(rh, (int, float)):
        return False, "remote_humidity missing"
    if rh < HUMIDITY_PLAUSIBLE_MIN or rh > HUMIDITY_PLAUSIBLE_MAX:
        return False, f"remote_humidity {rh}% outside {HUMIDITY_PLAUSIBLE_MIN}-{HUMIDITY_PLAUSIBLE_MAX}%"
    if prev_humidity is not None and abs(rh - prev_humidity) > MAX_HUMIDITY_JUMP:
        return False, (
            f"remote_humidity jumped {prev_humidity}% → {rh}% "
            f"(max {MAX_HUMIDITY_JUMP} points)"
        )
    lh = reading.get("local_humidity")
    if isinstance(lh, (int, float)) and abs(rh - lh) > REMOTE_LOCAL_MAX_DELTA:
        return False, (
            f"remote {rh}% vs local {lh}% delta > {REMOTE_LOCAL_MAX_DELTA}"
        )
    return True, "ok"


def acquire_lock():
    """Single instance. Keep the fd open for the process lifetime."""
    lock_fd = open(LOCK_FILE, "a+")
    try:
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error(f"Another humidifier bot holds {LOCK_FILE} — exiting")
        lock_fd.close()
        sys.exit(1)
    lock_fd.seek(0)
    lock_fd.truncate()
    lock_fd.write(str(os.getpid()))
    lock_fd.flush()
    return lock_fd


def start_caffeinate():
    """Idle-sleep assertion as a child of Python.app (keeps Local Network TCC)."""
    try:
        proc = subprocess.Popen(
            ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"caffeinate -i -w {os.getpid()} started (pid {proc.pid})")
        return proc
    except Exception as e:
        logger.warning(f"caffeinate not started: {e}")
        return None


def persist_state(history, current_state):
    history["last_state"] = current_state
    save_history(history)


_runtime = {"state": None, "history": None, "stopping": False}


def fail_safe_off(reason):
    history = _runtime.get("history") or {"readings": [], "last_state": None}
    current = _runtime.get("state")
    logger.info(reason)
    if current != "OFF":
        current = set_humidifier("OFF", current if current else "ON")
        persist_state(history, current)
        _runtime["state"] = current
    return current


def _on_signal(signum, _frame):
    if _runtime["stopping"]:
        return
    _runtime["stopping"] = True
    name = signal.Signals(signum).name
    fail_safe_off(f"Signal {name}: fail-safe OFF")
    sys.exit(0)


def main():
    lock_fd = acquire_lock()
    caffeinate_proc = start_caffeinate()
    atexit.register(lambda: caffeinate_proc.terminate() if caffeinate_proc and caffeinate_proc.poll() is None else None)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    logger.info("=" * 60)
    logger.info("Smart Humidifier Bot v2 starting")
    logger.info(f"Target: {HUMIDITY_LOW}-{HUMIDITY_HIGH}% RH")
    logger.info(f"Schedule: {START_HOUR}:00 - {END_HOUR}:00")
    logger.info(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60}min)")
    logger.info(f"Model: {VISION_MODEL_4B} @ {VISION_API_BASE}")
    if FFMPEG_CROP:
        logger.info(f"Frame crop: {FFMPEG_CROP} (w:h:x:y)")
    logger.info(
        f"Safety: max ON {MAX_ON_SECS}s, cooldown {MAX_ON_COOLDOWN_SECS}s, "
        f"plausible RH {HUMIDITY_PLAUSIBLE_MIN}-{HUMIDITY_PLAUSIBLE_MAX}%, "
        f"max jump {MAX_HUMIDITY_JUMP}"
    )
    if TEST_MODE:
        logger.warning("!! TEST MODE — no shortcuts will run !!")
    logger.info("=" * 60)

    history = load_history()
    current_state = history.get("last_state")
    _runtime["history"] = history
    _runtime["state"] = current_state

    current_state = fail_safe_off("Startup fail-safe: forcing humidifier OFF")
    history = _runtime["history"]

    consecutive_failures = 0
    last_successful_read = time.time()
    on_since = None
    cooldown_until = 0
    on_humidity_at_switch = None
    on_verify_left = 0
    on_verify_alerted = False

    while True:
        now = datetime.now()
        current_hour = now.hour

        if not (START_HOUR <= current_hour < END_HOUR):
            if current_state == "ON":
                logger.info(f"Outside schedule ({START_HOUR}-{END_HOUR}h), turning OFF")
                current_state = set_humidifier("OFF", current_state)
                persist_state(history, current_state)
                _runtime["state"] = current_state
                on_since = None
            logger.info(f"[{now.strftime('%H:%M')}] Outside hours. Sleeping 5min...")
            sleep_seconds(300)
            continue

        # Capture frame and read humidity
        frame_path = capture_frame()
        reading = None

        if frame_path:
            reading = read_humidity_from_image(frame_path, model_name=VISION_MODEL_4B)
            try:
                os.remove(frame_path)
            except OSError:
                pass

        if reading is not None:
            ok, why = reading_is_plausible(reading, last_good_humidity(history))
            if not ok:
                logger.warning(f"Implausible reading discarded: {why}")
                reading = None

        if reading is None or reading.get("remote_humidity") is None:
            consecutive_failures += 1
            logger.warning(f"Read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")

            secs_since_read = time.time() - last_successful_read
            if current_state == "ON" and secs_since_read > MAX_ON_WITHOUT_READ_SECS:
                logger.error(f"Humidifier ON for {int(secs_since_read)}s without a successful read → safety OFF")
                send_ntfy(
                    "Humidifier Safety Shutoff (timeout)",
                    f"No successful reading for {int(secs_since_read)}s while humidifier was ON — shutting off",
                    tags="rotating_light,warning",
                    priority="high",
                )
                current_state = set_humidifier("OFF", current_state)
                persist_state(history, current_state)
                _runtime["state"] = current_state
                on_since = None
                consecutive_failures = 0
                sleep_seconds(CHECK_INTERVAL * 2)
                continue

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.error(f"{MAX_CONSECUTIVE_FAILURES} consecutive failures → safety OFF")
                send_ntfy(
                    "Humidifier Safety Shutoff",
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive read failures — shutting off humidifier as safety measure",
                    tags="rotating_light,warning",
                    priority="high",
                )
                current_state = set_humidifier("OFF", current_state)
                persist_state(history, current_state)
                _runtime["state"] = current_state
                on_since = None
                sleep_seconds(CHECK_INTERVAL * 2)
                consecutive_failures = 0
            else:
                sleep_seconds(CHECK_INTERVAL)
            continue

        # Successful plausible read
        consecutive_failures = 0
        last_successful_read = time.time()
        humidity = reading["remote_humidity"]

        history["readings"].append({
            "time": now.isoformat(),
            "remote_humidity": humidity,
            "remote_temp": reading.get("remote_temp"),
            "local_humidity": reading.get("local_humidity"),
            "local_temp": reading.get("local_temp"),
            "state": current_state
        })

        desired_state, reason = decide_action(humidity, current_state)
        now_ts = time.time()

        if desired_state == "ON" and now_ts < cooldown_until:
            remaining = int(cooldown_until - now_ts)
            reason = f"{reason} — blocked by max-ON cooldown ({remaining}s left)"
            desired_state = "OFF"

        if current_state == "ON" and on_since and (now_ts - on_since) >= MAX_ON_SECS:
            logger.error(
                f"Humidifier ON for {int(now_ts - on_since)}s (cap {MAX_ON_SECS}s) → safety OFF + cooldown"
            )
            send_ntfy(
                "Humidifier Safety Shutoff (max ON)",
                f"Ran for {int(now_ts - on_since)}s without dropping below the ON cap — shutting off for {MAX_ON_COOLDOWN_SECS}s",
                tags="rotating_light,warning",
                priority="high",
            )
            desired_state = "OFF"
            cooldown_until = now_ts + MAX_ON_COOLDOWN_SECS

        logger.info(reason)
        new_state = set_humidifier(desired_state, current_state, reading=reading)
        if new_state == "ON":
            if current_state != "ON":
                on_since = now_ts
                on_humidity_at_switch = humidity
                on_verify_left = ON_VERIFY_CHECKS
                on_verify_alerted = False
            elif on_verify_left > 0:
                on_verify_left -= 1
                if on_verify_left == 0 and on_humidity_at_switch is not None and not on_verify_alerted:
                    rise = humidity - on_humidity_at_switch
                    if rise < ON_VERIFY_MIN_RISE:
                        on_verify_alerted = True
                        logger.warning(
                            f"ON trend check: RH {on_humidity_at_switch}% → {humidity}% "
                            f"(need +{ON_VERIFY_MIN_RISE}) — shortcut may not have switched the plug"
                        )
                        send_ntfy(
                            "Humidifier ON not confirmed",
                            f"After {ON_VERIFY_CHECKS} checks, humidity {on_humidity_at_switch}% → {humidity}%. "
                            "The plug may still be off, or the tank may be empty.",
                            tags="warning",
                            priority="high",
                        )
        else:
            on_since = None
            on_humidity_at_switch = None
            on_verify_left = 0
        current_state = new_state
        persist_state(history, current_state)
        _runtime["state"] = current_state
        _runtime["history"] = history

        sleep_seconds(CHECK_INTERVAL)

    # lock_fd held until process exit
    _ = lock_fd


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail_safe_off("KeyboardInterrupt: fail-safe OFF")
