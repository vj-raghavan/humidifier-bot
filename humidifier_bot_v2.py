#!/usr/bin/env python3
"""
Smart Humidifier Bot v2 — Humidity-Aware Control

Reads humidity from ThermoPro display via camera + local vision LLM (Qwen2.5-VL),
then controls humidifier to maintain target range for curry leaf plant.

Schedule: 7 AM – 8 PM
Check interval: 5 minutes
Hysteresis: ON below LOW%, OFF above HIGH%, hold between
Safety: 3 consecutive failed reads → shut off
"""

import base64
import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import requests

# --- Configuration ---
NTFY_TOPIC = "your-ntfy-topic"
NTFY_URL = f"https://ntfy.sh/{NTFY_TOPIC}"
SHORTCUT_ON_NAME = "PH On"
SHORTCUT_OFF_NAME = "PH Off"

# Humidity thresholds (remote sensor — near the plant)
HUMIDITY_LOW = 50    # Turn ON below this
HUMIDITY_HIGH = 60   # Turn OFF above this

# Schedule
START_HOUR = 7
END_HOUR = 20  # 8 PM

# Check interval (seconds)
CHECK_INTERVAL = 60  # 1 minute

# Camera RTSP
RTSP_URL = "rtsp://USER:PASSWORD@CAMERA_IP:554/stream1"
FFMPEG_TIMEOUT = 15  # seconds
FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"

# Vision LLM API (OpenAI-compatible — LM Studio via ngrok)
VISION_API_BASE = "http://YOUR_LLM_HOST:1234/v1"
VISION_MODEL_7B = "qwen/qwen2.5-vl-7b"
VISION_MODEL_4B = "qwen/qwen3-vl-4b"
VISION_MODEL_3B = "qwen2.5-vl-3b-instruct"
VISION_MODEL_GEMMA = "google/gemma-4-26b-a4b"
VISION_MODEL_QWEN3_5_27B = "qwen3.5-27b-claude-4.6-opus-reasoning-distilled"
VISION_API_KEY = "lm-studio"  # LM Studio doesn't check this

VISION_PROMPT = """This image is from a security camera showing a ThermoPro temperature/humidity display.
The display has TWO sections:
- TOP section: remote sensor readings (temperature on top line, humidity with % symbol below it)
- BOTTOM section: local readings (smaller numbers, temperature and humidity)

The humidity values have a % symbol next to them. Temperature values show degrees.
All readings are in Celsius.

Read the values and reply with ONLY this JSON - no other text, no markdown:
{"remote_humidity": <top humidity>, "remote_temp": <top temp>, "local_humidity": <bottom humidity>, "local_temp": <bottom temp>}
Use null if you cannot read a value clearly."""

# Safety
MAX_CONSECUTIVE_FAILURES = 3
MAX_ON_WITHOUT_READ_SECS = 300  # 5 min — auto-shutoff if ON with no successful read

# Test mode
TEST_MODE = False

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
    """Capture a single frame from the RTSP camera stream."""
    tmp_path = os.path.join(tempfile.gettempdir(), "humidifier_frame.jpg")
    try:
        result = subprocess.run(
            [
                FFMPEG_BIN, "-y",
                "-rtsp_transport", "tcp",
                "-i", RTSP_URL,
                "-frames:v", "1",
                "-q:v", "2",
                "-update", "1",
                tmp_path
            ],
            capture_output=True,
            text=True,
            timeout=FFMPEG_TIMEOUT
        )
        if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 10000:
            logger.info(f"Frame captured: {os.path.getsize(tmp_path)} bytes")
            return tmp_path
        else:
            logger.warning(f"Frame capture failed or too small")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg timed out capturing frame")
        return None
    except Exception as e:
        logger.error(f"Frame capture error: {e}")
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
    """Run a macOS Shortcut."""
    if TEST_MODE:
        logger.info(f"[TEST] Would run shortcut: '{shortcut_name}'")
        return True
    try:
        logger.info(f"Running shortcut via URL scheme: '{shortcut_name}'")
        url = f"shortcuts://run-shortcut?name={shortcut_name.replace(' ', '%20')}"
        subprocess.run(["open", url], check=True, timeout=30)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Shortcut error: {e}")
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Shortcut '{shortcut_name}' timed out")
        return False
    except FileNotFoundError:
        logger.error("'open' command not found")
        return False


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


def main():
    logger.info("=" * 60)
    logger.info("Smart Humidifier Bot v2 starting")
    logger.info(f"Target: {HUMIDITY_LOW}-{HUMIDITY_HIGH}% RH")
    logger.info(f"Schedule: {START_HOUR}:00 - {END_HOUR}:00")
    logger.info(f"Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL // 60}min)")
    logger.info(f"Model: {VISION_MODEL_4B} @ {VISION_API_BASE}")
    if TEST_MODE:
        logger.warning("!! TEST MODE — no shortcuts will run !!")
    logger.info("=" * 60)

    history = load_history()
    current_state = history.get("last_state")
    consecutive_failures = 0
    last_successful_read = time.time()  # Track last successful reading

    while True:
        now = datetime.now()
        current_hour = now.hour

        if not (START_HOUR <= current_hour < END_HOUR):
            # Outside hours — ensure OFF and sleep
            if current_state == "ON":
                logger.info(f"Outside schedule ({START_HOUR}-{END_HOUR}h), turning OFF")
                current_state = set_humidifier("OFF", current_state)
                history["last_state"] = current_state
                save_history(history)
            logger.info(f"[{now.strftime('%H:%M')}] Outside hours. Sleeping 5min...")
            time.sleep(300)
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

        if reading is None or reading.get("remote_humidity") is None:
            consecutive_failures += 1
            logger.warning(f"Read failed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")

            # Time-based safety: if humidifier is ON and we haven't had a good read in too long, shut off
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
                history["last_state"] = current_state
                save_history(history)
                consecutive_failures = 0
                time.sleep(CHECK_INTERVAL * 2)
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
                history["last_state"] = current_state
                save_history(history)
                time.sleep(CHECK_INTERVAL * 2)  # Wait longer before retrying
                consecutive_failures = 0
            else:
                time.sleep(CHECK_INTERVAL)
            continue

        # Successful read
        consecutive_failures = 0
        last_successful_read = time.time()
        humidity = reading["remote_humidity"]

        # Log reading
        history["readings"].append({
            "time": now.isoformat(),
            "remote_humidity": humidity,
            "remote_temp": reading.get("remote_temp"),
            "local_humidity": reading.get("local_humidity"),
            "local_temp": reading.get("local_temp"),
            "state": current_state
        })

        # Decide and act
        desired_state, reason = decide_action(humidity, current_state)
        logger.info(reason)
        current_state = set_humidifier(desired_state, current_state, reading=reading)
        history["last_state"] = current_state
        save_history(history)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nExiting Smart Humidifier Bot v2.")
