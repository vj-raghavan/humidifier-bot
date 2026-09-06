#!/usr/bin/env python3
"""
Smart Humidifier Bot v2 — Humidity-Aware Control

Reads humidity from a cropped ThermoPro LCD via local OCR first (Apple Vision
or Tesseract), with the vision LLM (Qwen2.5-VL on VISION_API_BASE) as fallback,
then controls humidifier to maintain target range for curry leaf plant.

Schedule: 7 AM – 8 PM
Check interval: 5 minutes
Hysteresis: ON below LOW%, OFF above HIGH%, hold between
Safety: N consecutive failed reads → shut off (keeps retrying humidity reads)
P0: fail-safe OFF on start/stop, max ON duration, OCR plausibility, single-instance lock
P1: shortcuts CLI, unique frame files, wall-clock sleep, ON-trend verify
P2: empty-tank pause, vision host backoff, digests, manual pause, crop-drift stills, FORCE_SYNC
"""

import atexit
import base64
import fcntl
import json
import logging
import logging.handlers
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import requests

import humidifier_ocr as ocr
import humidifier_ops as ops

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


def _env_int_any(names, default):
    for name in names:
        if os.environ.get(name) is not None:
            return int(os.environ[name])
    return int(default)


def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


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
VISION_MODEL_4B = VISION_MODEL
VISION_TIMEOUT = _env_int("VISION_TIMEOUT", 60)
VISION_SOFT_RETRIES = _env_int_any(("VISION_SOFT_RETRIES", "VISION_MAX_RETRIES"), 3)
VISION_SOFT_RETRY_DELAY = _env_int_any(("VISION_SOFT_RETRY_DELAY", "VISION_RETRY_DELAY"), 10)
VISION_MAX_RETRIES = VISION_SOFT_RETRIES
VISION_RETRY_DELAY = VISION_SOFT_RETRY_DELAY
VISION_HOST_DOWN_RETRIES = _env_int("VISION_HOST_DOWN_RETRIES", 1)
VISION_HOST_DOWN_RETRY_DELAY = _env_int("VISION_HOST_DOWN_RETRY_DELAY", 5)
VISION_HOST_DOWN_BACKOFF = _env_int("VISION_HOST_DOWN_BACKOFF", 180)
VISION_HOST_DOWN_NTFY = _env_bool("VISION_HOST_DOWN_NTFY", True)
VISION_HOST_DOWN_NTFY_SECS = _env_int("VISION_HOST_DOWN_NTFY_SECS", 3600)
# Wait after a failed capture/read before the next main-loop attempt (0 = CHECK_INTERVAL)
FAILED_READ_RETRY_SECS = _env_int("FAILED_READ_RETRY_SECS", 0)
# Extra wait after consecutive-failure safety OFF (0 = 2× CHECK_INTERVAL)
FAILED_READ_BACKOFF_SECS = _env_int("FAILED_READ_BACKOFF_SECS", 0)

# Local OCR first (Mac Mini); vision LLM only if OCR fails / is low-confidence.
HUMIDITY_READ_MODE = ocr.normalize_read_mode(_env("HUMIDITY_READ_MODE", ocr.READ_MODE_OCR_FIRST))
OCR_BACKEND = _env("OCR_BACKEND", "auto")
OCR_TIMEOUT = _env_int("OCR_TIMEOUT", 8)
OCR_MIN_CONFIDENCE = _env_float("OCR_MIN_CONFIDENCE", 0.5)
OCR_TESSERACT_BIN = _env("OCR_TESSERACT_BIN", "tesseract")
OCR_TESSERACT_LANG = _env("OCR_TESSERACT_LANG", "eng")
OCR_TESSERACT_PSM = _env("OCR_TESSERACT_PSM", "6")
OCR_TESSERACT_WHITELIST = _env("OCR_TESSERACT_WHITELIST", "0123456789.%C")
OCR_VISION_LEVEL = _env("OCR_VISION_LEVEL", "accurate")

VISION_PROMPT = """This image is a cropped close-up of a ThermoPro display showing the REMOTE sensor (labeled OUT / CH1).
The large digits for the remote sensor are temperature in Celsius (upper part) and humidity with a % symbol (lower part).

If the bottom edge of the crop happens to show the top of the indoor/local (IN) section, IGNORE IT completely. 
Extract ONLY the remote (OUT/CH1) readings. Do not extract or invent local readings.

Reply with ONLY this JSON - no other text, no markdown:
{"remote_humidity": <humidity number>, "remote_temp": <temperature number>, "local_humidity": null, "local_temp": null}
Use null if you cannot read a value clearly."""

MAX_CONSECUTIVE_FAILURES = _env_int("MAX_CONSECUTIVE_FAILURES", 3)
MAX_ON_WITHOUT_READ_SECS = _env_int("MAX_ON_WITHOUT_READ_SECS", 300)
MAX_ON_SECS = _env_int("MAX_ON_SECS", 1800)
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
EMPTY_TANK_FAILS = _env_int("EMPTY_TANK_FAILS", 3)

PAUSE_FILE = _env("PAUSE_FILE", os.path.join(_PKG_DIR, "humidifier_pause.json"))
PAUSE_LOG_INTERVAL = _env_int("PAUSE_LOG_INTERVAL", 300)
DIGEST_STATE_FILE = _env("DIGEST_STATE_FILE", os.path.join(_PKG_DIR, "humidifier_digest.json"))
DIGEST_ENABLED = _env_bool("DIGEST_ENABLED", True)
DIGEST_MORNING_HOUR = _env_int("DIGEST_MORNING_HOUR", START_HOUR)
DIGEST_EVENING_HOUR = _env_int("DIGEST_EVENING_HOUR", END_HOUR)

CROP_DRIFT_STREAK = _env_int("CROP_DRIFT_STREAK", 4)
CROP_DRIFT_NTFY_SECS = _env_int("CROP_DRIFT_NTFY_SECS", 3600)
CROP_DRIFT_SAVE_FULL = _env_bool("CROP_DRIFT_SAVE_FULL", True)
STILLS_DIR = _env("STILLS_DIR", os.path.join(_PKG_DIR, "stills"))

FORCE_OFF_SYNC_MARGIN = _env_int("FORCE_OFF_SYNC_MARGIN", 5)
FORCE_OFF_SYNC_INTERVAL = _env_int("FORCE_OFF_SYNC_INTERVAL", 900)

NTFY_ERROR_MIN_INTERVAL = _env_int("NTFY_ERROR_MIN_INTERVAL", 600)
HISTORY_MAX_READINGS = _env_int("HISTORY_MAX_READINGS", 1600)

LOCK_FILE = os.path.join(_PKG_DIR, "humidifier_bot.lock")

# --- Logging ---
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "humidifier_v2.log")

logger = logging.getLogger("humidifier_v2")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_fh = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=5)
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

HISTORY_FILE = os.path.join(LOG_DIR, "humidity_history.json")
_ntfy_limiter = ops.RateLimiter()


def load_history():
    """Load humidity reading history."""
    data = ops.load_json_file(HISTORY_FILE, {"readings": [], "last_state": None})
    data.setdefault("readings", [])
    data.setdefault("last_state", None)
    return data


def save_history(history):
    """Save humidity reading history."""
    history["readings"] = history["readings"][-HISTORY_MAX_READINGS:]
    ops.atomic_write_json(HISTORY_FILE, history)


def _ffmpeg_capture(dest_path, crop=None):
    vf = []
    if crop:
        vf = ["-vf", f"crop={crop}"]
    result = subprocess.run(
        [
            FFMPEG_BIN, "-y",
            "-rtsp_transport", "tcp",
            "-i", RTSP_URL,
            *vf,
            "-frames:v", "1",
            "-q:v", "2",
            "-update", "1",
            dest_path,
        ],
        capture_output=True,
        text=True,
        timeout=FFMPEG_TIMEOUT,
    )
    size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
    return result, size


def capture_frame(crop=True):
    """Capture a single frame from the RTSP camera stream into a unique temp file."""
    fd, tmp_path = tempfile.mkstemp(prefix="humidifier_frame_", suffix=".jpg")
    os.close(fd)
    crop_arg = FFMPEG_CROP if crop and FFMPEG_CROP else None
    try:
        result, size = _ffmpeg_capture(tmp_path, crop_arg)
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


def save_crop_preview(cropped_path):
    """Persist dated stills for crop-drift debugging. Returns paths written."""
    os.makedirs(STILLS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cropped_dest = os.path.join(STILLS_DIR, f"crop_drift_{stamp}_cropped.jpg")
    paths = []
    try:
        shutil.copy2(cropped_path, cropped_dest)
        paths.append(cropped_dest)
    except OSError as e:
        logger.warning(f"Could not save cropped still: {e}")
    if CROP_DRIFT_SAVE_FULL:
        full = capture_frame(crop=False)
        if full:
            full_dest = os.path.join(STILLS_DIR, f"crop_drift_{stamp}_full.jpg")
            try:
                shutil.move(full, full_dest)
                paths.append(full_dest)
            except OSError:
                try:
                    os.remove(full)
                except OSError:
                    pass
    return paths


def _failed_read_wait_secs(backoff=False, host_down=False):
    """Seconds to wait before the next capture/read after a failed humidity cycle."""
    if backoff:
        extra = FAILED_READ_BACKOFF_SECS
        wait = extra if extra > 0 else CHECK_INTERVAL * 2
    else:
        retry = FAILED_READ_RETRY_SECS
        wait = retry if retry > 0 else CHECK_INTERVAL
    if host_down:
        return max(wait, VISION_HOST_DOWN_BACKOFF)
    return wait


def _wait_then_retry_vision(attempt, max_retries, retry_delay, reason):
    """Log a failed vision attempt. Return True to retry the same frame."""
    logger.warning(f"{reason} (attempt {attempt}/{max_retries})")
    if attempt < max_retries and not _runtime["stopping"]:
        logger.info(f"Retrying humidity resolution in {retry_delay}s...")
        sleep_seconds(retry_delay)
        return not _runtime["stopping"]
    return False


def _ocr_try_read(image_path, prev_humidity):
    """Local OCR → parse → confidence/plausibility. Returns (reading, why)."""
    result = ocr.ocr_image(
        image_path,
        timeout=OCR_TIMEOUT,
        prefer=OCR_BACKEND,
        tesseract_bin=OCR_TESSERACT_BIN,
        tesseract_lang=OCR_TESSERACT_LANG,
        tesseract_psm=OCR_TESSERACT_PSM,
        tesseract_whitelist=OCR_TESSERACT_WHITELIST,
        vision_level=OCR_VISION_LEVEL,
    )
    if result.get("error"):
        return None, result["error"]
    reading, why = ocr.parse_thermopro_ocr(
        result.get("text") or "",
        result.get("observations") or [],
        plausible_min=HUMIDITY_PLAUSIBLE_MIN,
        plausible_max=HUMIDITY_PLAUSIBLE_MAX,
    )
    if reading is None:
        return None, why
    ok, reason = ocr.accept_ocr_reading(
        reading,
        min_confidence=OCR_MIN_CONFIDENCE,
        prev_humidity=prev_humidity,
        plausible_min=HUMIDITY_PLAUSIBLE_MIN,
        plausible_max=HUMIDITY_PLAUSIBLE_MAX,
        max_jump=MAX_HUMIDITY_JUMP,
        remote_local_max_delta=REMOTE_LOCAL_MAX_DELTA,
        plausibility_fn=ops.reading_is_plausible,
    )
    if not ok:
        return None, reason
    reading["source"] = "ocr"
    reading["ocr_backend"] = ocr.backend_label(result.get("backend"), result.get("detail"))
    return reading, "ok"


def resolve_humidity_reading(image_path, prev_humidity=None, model_name=None):
    """OCR-first humidity read with optional vision-LLM fallback.

    Returns (reading_or_None, kind) using the same VISION_* kinds as the LLM
    path so host-down backoff still applies when falling back.
    """
    mode = HUMIDITY_READ_MODE
    if mode != ocr.READ_MODE_LLM_ONLY:
        ocr_reading, ocr_why = _ocr_try_read(image_path, prev_humidity)
        if ocr_reading is not None:
            logger.info(
                f"Humidity via ocr ({ocr_reading.get('ocr_backend')} "
                f"conf={ocr_reading.get('ocr_confidence')} "
                f"method={ocr_reading.get('ocr_method')}): "
                f"remote_humidity={ocr_reading.get('remote_humidity')}%, "
                f"remote_temp={ocr_reading.get('remote_temp')}"
            )
            return ocr_reading, ops.VISION_OK
        if mode == ocr.READ_MODE_OCR_ONLY:
            logger.warning(f"OCR-only read failed: {ocr_why}")
            return None, ops.VISION_SOFT
        logger.info(f"OCR missed ({ocr_why}); falling back to vision LLM")

    reading, kind = read_humidity_from_image(
        image_path, model_name=model_name or VISION_MODEL_4B
    )
    if reading is not None:
        reading["source"] = "llm"
        logger.info(
            f"Humidity via llm: remote_humidity={reading.get('remote_humidity')}%, "
            f"remote_temp={reading.get('remote_temp')}"
        )
    return reading, kind


def read_humidity_from_image(
    image_path,
    model_name=VISION_MODEL_4B,
    max_retries=None,
    retry_delay=None,
):
    """Vision LLM humidity read.

    Returns (reading_or_None, kind) where kind is ops.VISION_OK, VISION_SOFT,
    or VISION_HOST_DOWN. Soft OCR/HTTP blips retry on the same frame
    (VISION_SOFT_RETRIES / VISION_MAX_RETRIES). Host-down uses fewer attempts
    then a longer main-loop backoff. The main loop recaptures after None.
    """
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
    except Exception as e:
        logger.error(f"Failed to read image file: {e}")
        return None, ops.VISION_SOFT

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0,
    }

    soft_retries = max(1, int(VISION_SOFT_RETRIES if max_retries is None else max_retries))
    soft_delay = max(0, int(VISION_SOFT_RETRY_DELAY if retry_delay is None else retry_delay))
    host_retries = max(1, int(VISION_HOST_DOWN_RETRIES))
    host_attempts = 0
    soft_attempts = 0
    last_kind = ops.VISION_SOFT

    while True:
        if _runtime["stopping"]:
            return None, last_kind
        try:
            r = requests.post(
                f"{VISION_API_BASE}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {VISION_API_KEY}"},
                timeout=VISION_TIMEOUT,
            )
            if r.status_code != 200:
                kind = ops.classify_http_status(r.status_code)
                last_kind = kind
                reason = f"Vision API returned {r.status_code} ({kind}): {r.text[:200]}"
                if kind == ops.VISION_HOST_DOWN:
                    host_attempts += 1
                    if _wait_then_retry_vision(
                        host_attempts, host_retries, VISION_HOST_DOWN_RETRY_DELAY, reason
                    ):
                        continue
                    return None, ops.VISION_HOST_DOWN
                soft_attempts += 1
                if _wait_then_retry_vision(soft_attempts, soft_retries, soft_delay, reason):
                    continue
                return None, ops.VISION_SOFT

            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```json\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            json_match = re.search(r"\{[^}]+\}", text)
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
                    return reading, ops.VISION_OK
                reason = f"No valid humidity in response: {text}"
            else:
                reason = f"Could not parse vision response: {text}"

            last_kind = ops.VISION_SOFT
            soft_attempts += 1
            if _wait_then_retry_vision(soft_attempts, soft_retries, soft_delay, reason):
                continue
            return None, ops.VISION_SOFT

        except Exception as e:
            kind = ops.classify_vision_exception(e)
            last_kind = kind
            if kind == ops.VISION_HOST_DOWN:
                host_attempts += 1
                if _wait_then_retry_vision(
                    host_attempts,
                    host_retries,
                    VISION_HOST_DOWN_RETRY_DELAY,
                    f"Vision host down: {e}",
                ):
                    continue
                return None, ops.VISION_HOST_DOWN
            soft_attempts += 1
            if _wait_then_retry_vision(
                soft_attempts, soft_retries, soft_delay, f"Vision API error: {e}"
            ):
                continue
            return None, last_kind

    return None, last_kind


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


def send_ntfy(title, message, tags="potted_plant", priority="default", rate_key=None, rate_secs=None):
    """Send a push notification via ntfy. Optional per-key rate limit."""
    if rate_key:
        interval = NTFY_ERROR_MIN_INTERVAL if rate_secs is None else rate_secs
        if not _ntfy_limiter.allow(rate_key, interval):
            logger.info(f"ntfy rate-limited ({rate_key}): {title}")
            return False
    try:
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
    return True


def set_humidifier(state, current_state, reading=None, force=False):
    """Turn humidifier ON or OFF.

    force=True (FORCE_SYNC) always sends the shortcut even if internal state
    already matches, so HomeKit/plug drift can be corrected. Force-sync of an
    already-matching state does not ntfy.
    """
    if state not in ("ON", "OFF"):
        raise ValueError(f"invalid humidifier state {state!r}")
    already = state == current_state
    if already and not force:
        logger.info(f"Humidifier already {state}, no action needed")
        return state

    if force:
        logger.info(f"FORCE_SYNC {state} (internal was {current_state})")

    if state == "ON":
        success = run_shortcut(SHORTCUT_ON_NAME)
    else:
        success = run_shortcut(SHORTCUT_OFF_NAME)

    if success:
        logger.info(f"Humidifier → {state}")
        if force and already:
            return state
        if reading:
            rh = reading.get("remote_humidity", "?")
            rt = reading.get("remote_temp", "?")
            lh = reading.get("local_humidity", "?")
            lt = reading.get("local_temp", "?")
            msg = (
                f"Humidifier turned {state}\n"
                f"Sensor: {rh}% RH / {rt}°C\n"
                f"Local: {lh}% RH / {lt}°C"
            )
        else:
            msg = f"Humidifier turned {state}"
        send_ntfy(
            f"Humidifier {state}",
            msg,
            tags="potted_plant,droplet" if state == "ON" else "potted_plant,no_entry",
        )
        return state

    logger.error(f"Failed to set humidifier to {state}")
    send_ntfy(
        "Humidifier Error",
        f"Failed to switch humidifier to {state}",
        tags="warning",
        priority="high",
        rate_key="shortcut_error",
    )
    return current_state if current_state in ("ON", "OFF") else "OFF"


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
    """Reject OCR/LLM nonsense. Returns (ok, reason). Local sensor is optional."""
    return ops.reading_is_plausible(
        reading,
        prev_humidity,
        plausible_min=HUMIDITY_PLAUSIBLE_MIN,
        plausible_max=HUMIDITY_PLAUSIBLE_MAX,
        max_jump=MAX_HUMIDITY_JUMP,
        remote_local_max_delta=REMOTE_LOCAL_MAX_DELTA,
    )


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


_runtime = {
    "state": None,
    "history": None,
    "stopping": False,
    "on_since": None,
    "last_force_off": 0,
    "last_pause_log": 0,
    "host_was_down": False,
    "consecutive_empty_tank": 0,
    "quality_fail_streak": 0,
}


def _credit_on_time(history, now_ts):
    on_since = _runtime.get("on_since")
    if on_since:
        ops.bump_stat(history, "on_seconds", max(0, int(now_ts - on_since)))
        _runtime["on_since"] = now_ts if _runtime.get("state") == "ON" else None


def fail_safe_off(reason):
    history = _runtime.get("history") or {"readings": [], "last_state": None}
    logger.info(reason)
    current = set_humidifier("OFF", _runtime.get("state"), force=True)
    _credit_on_time(history, time.time())
    persist_state(history, current)
    _runtime["state"] = current
    _runtime["on_since"] = None
    return current


def _on_signal(signum, _frame):
    if _runtime["stopping"]:
        return
    _runtime["stopping"] = True
    name = signal.Signals(signum).name
    fail_safe_off(f"Signal {name}: fail-safe OFF")
    sys.exit(0)


def maybe_send_digests(now, history):
    if not DIGEST_ENABLED:
        return
    last = ops.load_json_file(DIGEST_STATE_FILE, {})
    day = ops.day_key(now)
    pause = ops.load_pause(PAUSE_FILE)
    extra = 0
    if _runtime.get("state") == "ON" and _runtime.get("on_since"):
        extra = max(0, int(time.time() - _runtime["on_since"]))
    changed = False
    for kind, hour in (("morning", DIGEST_MORNING_HOUR), ("evening", DIGEST_EVENING_HOUR)):
        if ops.digest_due(last, kind=kind, day=day, hour=now.hour, target_hour=hour):
            title, body = ops.build_digest_message(
                history, when=now, kind=kind, pause=pause, extra_on_seconds=extra
            )
            send_ntfy(title, body, tags="potted_plant,bar_chart")
            last = ops.mark_digest_sent(last, kind, day)
            changed = True
            logger.info(f"Sent {kind} digest for {day}")
    if changed:
        ops.atomic_write_json(DIGEST_STATE_FILE, last)


def enter_empty_tank_pause(history, humidity_at, humidity_now):
    reason = (
        f"Humidity did not rise after {EMPTY_TANK_FAILS} ON-verify windows "
        f"(last {humidity_at}% → {humidity_now}%). Refill the tank or check the plug."
    )
    ops.save_pause(PAUSE_FILE, reason=reason, source="empty_tank", until=None)
    ops.bump_stat(history, "empty_tank", 1)
    logger.error(reason)
    send_ntfy(
        "Humidifier needs attention",
        reason + f"\nPaused until you refill / check the plug, then: python3 humidifier_bot_v2.py resume\n"
        f"(or delete {PAUSE_FILE})",
        tags="warning,droplet",
        priority="high",
        rate_key="empty_tank",
        rate_secs=NTFY_ERROR_MIN_INTERVAL,
    )


def maybe_crop_drift_alert(reason, frame_path, history):
    if not ops.is_crop_like_reason(reason):
        return
    _runtime["quality_fail_streak"] = _runtime.get("quality_fail_streak", 0) + 1
    if _runtime["quality_fail_streak"] < CROP_DRIFT_STREAK:
        return
    if not _ntfy_limiter.allow("crop_drift_gate", CROP_DRIFT_NTFY_SECS):
        return
    paths = []
    if frame_path:
        paths = save_crop_preview(frame_path)
    ops.bump_stat(history, "crop_drift_alerts", 1)
    crop = FFMPEG_CROP or "(none — full frame)"
    path_txt = "\n".join(paths) if paths else "(no still saved)"
    send_ntfy(
        "Humidifier crop drift?",
        f"OCR/range/split failures x{_runtime['quality_fail_streak']}: {reason}\n"
        f"Check FFMPEG_CROP={crop}\n"
        f"Preview:\n{path_txt}",
        tags="warning,camera",
        rate_key="crop_drift",
        rate_secs=CROP_DRIFT_NTFY_SECS,
    )


def apply_pause_hold(history, current_state, now_ts, log_always=False):
    """If pause is active, keep OFF and skip ON. Returns (state, paused_bool)."""
    pause = ops.load_pause(PAUSE_FILE)
    if not ops.pause_is_active(pause, datetime.now()):
        return current_state, False
    if current_state != "OFF":
        logger.info(f"Pause active ({pause.get('source')}): forcing OFF")
        current_state = set_humidifier("OFF", current_state, force=True)
        _credit_on_time(history, now_ts)
        persist_state(history, current_state)
        _runtime["state"] = current_state
        _runtime["on_since"] = None
    if log_always or (now_ts - _runtime.get("last_pause_log", 0) >= PAUSE_LOG_INTERVAL):
        _runtime["last_pause_log"] = now_ts
        logger.info(
            f"Paused ({pause.get('source')}): {ops.pause_expiry_label(pause)} — {pause.get('reason')}"
        )
    return current_state, True


def main():
    lock_fd = acquire_lock()
    caffeinate_proc = start_caffeinate()
    atexit.register(
        lambda: caffeinate_proc.terminate()
        if caffeinate_proc and caffeinate_proc.poll() is None
        else None
    )

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    logger.info("=" * 60)
    logger.info("Smart Humidifier Bot v2 starting")
    logger.info(f"Target: {HUMIDITY_LOW}-{HUMIDITY_HIGH}% RH")
    logger.info(f"Schedule: {START_HOUR}:00 - {END_HOUR}:00")
    logger.info(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60}min)")
    ocr_backend, ocr_detail = ocr.detect_ocr_backend(
        prefer=OCR_BACKEND, tesseract_bin=OCR_TESSERACT_BIN
    )
    logger.info(
        f"Humidity read mode: {HUMIDITY_READ_MODE} "
        f"(local OCR={ocr.backend_label(ocr_backend, ocr_detail)}, "
        f"min_conf={OCR_MIN_CONFIDENCE}, timeout={OCR_TIMEOUT}s)"
    )
    logger.info(f"Vision LLM fallback: {VISION_MODEL_4B} @ {VISION_API_BASE}")
    logger.info(
        f"Vision retries: {VISION_MAX_RETRIES} attempts, {VISION_RETRY_DELAY}s delay, "
        f"timeout {VISION_TIMEOUT}s"
    )
    if FFMPEG_CROP:
        logger.info(f"Frame crop: {FFMPEG_CROP} (w:h:x:y)")
    logger.info(
        f"Safety: max ON {MAX_ON_SECS}s, cooldown {MAX_ON_COOLDOWN_SECS}s, "
        f"plausible RH {HUMIDITY_PLAUSIBLE_MIN}-{HUMIDITY_PLAUSIBLE_MAX}%, "
        f"max jump {MAX_HUMIDITY_JUMP}, "
        f"{MAX_CONSECUTIVE_FAILURES} failed reads → OFF (then keep retrying)"
    )
    logger.info(
        f"Empty-tank pause after {EMPTY_TANK_FAILS} failed ON-verifies; "
        f"vision host-down backoff {VISION_HOST_DOWN_BACKOFF}s"
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
    cooldown_until = 0
    on_humidity_at_switch = None
    on_verify_left = 0
    on_verify_alerted = False

    while True:
        now = datetime.now()
        now_ts = time.time()
        current_hour = now.hour
        maybe_send_digests(now, history)

        current_state, paused = apply_pause_hold(history, current_state, now_ts)

        if not (START_HOUR <= current_hour < END_HOUR):
            if current_state == "ON":
                logger.info(f"Outside schedule ({START_HOUR}-{END_HOUR}h), turning OFF")
                current_state = set_humidifier("OFF", current_state)
                _credit_on_time(history, now_ts)
                persist_state(history, current_state)
                _runtime["state"] = current_state
                _runtime["on_since"] = None
            logger.info(f"[{now.strftime('%H:%M')}] Outside hours. Sleeping 5min...")
            sleep_seconds(300)
            continue

        if paused:
            sleep_seconds(CHECK_INTERVAL)
            continue

        frame_path = capture_frame()
        reading = None
        vision_kind = ops.VISION_SOFT

        if frame_path:
            reading, vision_kind = resolve_humidity_reading(
                frame_path,
                prev_humidity=last_good_humidity(history),
                model_name=VISION_MODEL_4B,
            )

        fail_reason = None
        if reading is not None:
            ok, why = reading_is_plausible(reading, last_good_humidity(history))
            if not ok:
                logger.warning(f"Implausible reading discarded: {why}")
                fail_reason = why
                maybe_crop_drift_alert(why, frame_path, history)
                reading = None
                vision_kind = ops.VISION_SOFT
        elif vision_kind == ops.VISION_SOFT:
            fail_reason = "no valid humidity / unparseable OCR"
            maybe_crop_drift_alert(fail_reason, frame_path, history)

        if frame_path:
            try:
                os.remove(frame_path)
            except OSError:
                pass

        if reading is None or reading.get("remote_humidity") is None:
            consecutive_failures += 1
            ops.bump_stat(history, "failed_reads", 1)
            persist_state(history, current_state)
            host_down = vision_kind == ops.VISION_HOST_DOWN

            if host_down:
                ops.bump_stat(history, "host_down", 1)
                if not _runtime.get("host_was_down"):
                    logger.warning(f"Vision host looks down ({VISION_API_BASE}) — backing off")
                _runtime["host_was_down"] = True
                if VISION_HOST_DOWN_NTFY:
                    send_ntfy(
                        "Vision host unreachable",
                        f"Cannot reach {VISION_API_BASE}. Backing off {VISION_HOST_DOWN_BACKOFF}s "
                        "instead of retrying the control interval.",
                        tags="warning,computer",
                        rate_key="host_down",
                        rate_secs=VISION_HOST_DOWN_NTFY_SECS,
                    )
            else:
                logger.warning(
                    f"Humidity read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}); "
                    "will recapture and retry"
                )

            secs_since_read = time.time() - last_successful_read
            if current_state == "ON" and secs_since_read > MAX_ON_WITHOUT_READ_SECS:
                wait_s = _failed_read_wait_secs(backoff=True, host_down=host_down)
                logger.error(
                    f"Humidifier ON for {int(secs_since_read)}s without a successful read "
                    f"→ safety OFF, then retry read in {wait_s}s"
                )
                send_ntfy(
                    "Humidifier Safety Shutoff (timeout)",
                    f"No successful reading for {int(secs_since_read)}s while humidifier was ON — shutting off",
                    tags="rotating_light,warning",
                    priority="high",
                    rate_key="safety_timeout",
                )
                current_state = set_humidifier("OFF", current_state, force=True)
                ops.bump_stat(history, "force_offs", 1)
                _credit_on_time(history, time.time())
                persist_state(history, current_state)
                _runtime["state"] = current_state
                _runtime["on_since"] = None
                consecutive_failures = 0
                logger.info(f"Retrying humidity read in {wait_s}s")
                sleep_seconds(wait_s)
                continue

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                wait_s = _failed_read_wait_secs(backoff=True, host_down=host_down)
                logger.error(
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive failures → safety OFF, "
                    f"then keep retrying humidity reads in {wait_s}s"
                )
                send_ntfy(
                    "Humidifier Safety Shutoff",
                    f"{MAX_CONSECUTIVE_FAILURES} consecutive read failures — shutting off humidifier as safety measure",
                    tags="rotating_light,warning",
                    priority="high",
                    rate_key="safety_failures",
                )
                current_state = set_humidifier("OFF", current_state, force=True)
                ops.bump_stat(history, "force_offs", 1)
                _credit_on_time(history, time.time())
                persist_state(history, current_state)
                _runtime["state"] = current_state
                _runtime["on_since"] = None
                consecutive_failures = 0
                logger.info(f"Retrying humidity read in {wait_s}s (not giving up)")
                sleep_seconds(wait_s)
            else:
                wait_s = _failed_read_wait_secs(backoff=False, host_down=host_down)
                logger.info(f"Retrying humidity read in {wait_s}s")
                sleep_seconds(wait_s)
            continue

        if _runtime.get("host_was_down"):
            logger.info("Vision host recovered — resuming normal check interval")
            _runtime["host_was_down"] = False

        consecutive_failures = 0
        _runtime["quality_fail_streak"] = 0
        last_successful_read = time.time()
        humidity = reading["remote_humidity"]
        source = reading.get("source") or "llm"
        logger.info(f"Using {source} humidity reading: {humidity}%")

        history["readings"].append({
            "time": now.isoformat(),
            "remote_humidity": humidity,
            "remote_temp": reading.get("remote_temp"),
            "local_humidity": reading.get("local_humidity"),
            "local_temp": reading.get("local_temp"),
            "source": source,
            "state": current_state,
        })

        desired_state, reason = decide_action(humidity, current_state)
        now_ts = time.time()

        if desired_state == "ON" and now_ts < cooldown_until:
            remaining = int(cooldown_until - now_ts)
            reason = f"{reason} — blocked by max-ON cooldown ({remaining}s left)"
            desired_state = "OFF"

        if current_state == "ON" and _runtime.get("on_since") and (now_ts - _runtime["on_since"]) >= MAX_ON_SECS:
            logger.error(
                f"Humidifier ON for {int(now_ts - _runtime['on_since'])}s (cap {MAX_ON_SECS}s) → safety OFF + cooldown"
            )
            send_ntfy(
                "Humidifier Safety Shutoff (max ON)",
                f"Ran for {int(now_ts - _runtime['on_since'])}s without dropping below the ON cap — shutting off for {MAX_ON_COOLDOWN_SECS}s",
                tags="rotating_light,warning",
                priority="high",
                rate_key="safety_max_on",
            )
            desired_state = "OFF"
            cooldown_until = now_ts + MAX_ON_COOLDOWN_SECS
            ops.bump_stat(history, "force_offs", 1)

        pause = ops.load_pause(PAUSE_FILE)
        if ops.pause_is_active(pause) and desired_state == "ON":
            reason = f"{reason} — blocked by pause ({pause.get('source')})"
            desired_state = "OFF"

        logger.info(reason)

        force_sync = False
        if ops.should_force_off_sync(
            desired_state=desired_state,
            current_state=current_state,
            humidity=humidity,
            humidity_high=HUMIDITY_HIGH,
            margin=FORCE_OFF_SYNC_MARGIN,
            now_ts=now_ts,
            last_force_ts=_runtime.get("last_force_off", 0),
            interval=FORCE_OFF_SYNC_INTERVAL,
        ):
            logger.warning(
                f"Humidity {humidity}% still high while internal state is OFF — FORCE_SYNC OFF "
                f"(HomeKit/plug drift?)"
            )
            force_sync = True
            _runtime["last_force_off"] = now_ts
            ops.bump_stat(history, "force_sync_offs", 1)

        prev_state = current_state
        new_state = set_humidifier(desired_state, current_state, reading=reading, force=force_sync)
        if new_state == "ON":
            if prev_state != "ON":
                _runtime["on_since"] = now_ts
                on_humidity_at_switch = humidity
                on_verify_left = ON_VERIFY_CHECKS
                on_verify_alerted = False
            elif on_verify_left > 0:
                on_verify_left -= 1
                if on_verify_left == 0 and on_humidity_at_switch is not None:
                    rise = humidity - on_humidity_at_switch
                    if rise < ON_VERIFY_MIN_RISE:
                        _runtime["consecutive_empty_tank"] = _runtime.get("consecutive_empty_tank", 0) + 1
                        logger.warning(
                            f"ON trend check: RH {on_humidity_at_switch}% → {humidity}% "
                            f"(need +{ON_VERIFY_MIN_RISE}) — "
                            f"fail {_runtime['consecutive_empty_tank']}/{EMPTY_TANK_FAILS}"
                        )
                        if not on_verify_alerted:
                            on_verify_alerted = True
                            send_ntfy(
                                "Humidifier ON not confirmed",
                                f"After {ON_VERIFY_CHECKS} checks, humidity {on_humidity_at_switch}% → {humidity}%. "
                                "The plug may still be off, or the tank may be empty.",
                                tags="warning",
                                priority="high",
                                rate_key="on_verify",
                            )
                        if ops.empty_tank_should_pause(_runtime["consecutive_empty_tank"], EMPTY_TANK_FAILS):
                            current_state = set_humidifier("OFF", new_state, force=True)
                            _credit_on_time(history, time.time())
                            _runtime["on_since"] = None
                            enter_empty_tank_pause(history, on_humidity_at_switch, humidity)
                            persist_state(history, current_state)
                            _runtime["state"] = current_state
                            _runtime["history"] = history
                            _runtime["consecutive_empty_tank"] = 0
                            on_humidity_at_switch = None
                            on_verify_left = 0
                            sleep_seconds(CHECK_INTERVAL)
                            continue
                        # Still ON but ineffective — start another verify window
                        on_humidity_at_switch = humidity
                        on_verify_left = ON_VERIFY_CHECKS
                        on_verify_alerted = True
                    else:
                        _runtime["consecutive_empty_tank"] = 0
        else:
            if prev_state == "ON":
                _credit_on_time(history, now_ts)
            _runtime["on_since"] = None
            on_humidity_at_switch = None
            on_verify_left = 0
        current_state = new_state
        persist_state(history, current_state)
        _runtime["state"] = current_state
        _runtime["history"] = history

        sleep_seconds(CHECK_INTERVAL)

    _ = lock_fd


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(ops.cli(sys.argv[1:], pause_path=PAUSE_FILE, start_hour=START_HOUR))
    try:
        main()
    except KeyboardInterrupt:
        fail_safe_off("KeyboardInterrupt: fail-safe OFF")
