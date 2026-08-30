# Humidifier Bot

A smart plant humidifier automation script that uses a local vision LLM (Qwen2.5-VL via LM Studio) to read the humidity from a standard digital hygrometer via an RTSP camera stream, and toggles a smart plug via macOS Shortcuts.

## Setup

1. Create two macOS Shortcuts: `PH On` and `PH Off`. These should control your smart plug (e.g. via HomeKit).
2. Configure the following variables in `humidifier_bot_v2.py`:
   - `RTSP_URL`: The RTSP stream of your camera pointing at the hygrometer.
   - `VISION_API_BASE`: The base URL of your local OpenAI-compatible Vision API (e.g., LM Studio).
   - `NTFY_TOPIC`: Your ntfy.sh topic for push notifications.
3. Run `python3 humidifier_bot_v2.py`.
