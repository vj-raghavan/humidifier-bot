# Humidifier Bot

Humidity-aware plant humidifier control. A camera frame of a ThermoPro display is read by a local vision LLM; a HomeKit smart plug is toggled via macOS Shortcuts (`PH On` / `PH Off`).

Runtime settings live in `.env` (not committed). Copy `.env.example` and fill in camera, vision, and ntfy values.

## Always-on (macOS)

This package is the live bot, started by LaunchAgent `com.samwise.humidifier-bot` (`KeepAlive`, same `Python.app` binary as the original agent so Local Network / RTSP keep working).

```bash
cp com.samwise.humidifier-bot.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.samwise.humidifier-bot 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.samwise.humidifier-bot.plist
```

Logs: `humidifier_v2.log` in this directory, plus `/tmp/humidifier-bot.log` and `/tmp/humidifier-bot.err`.

On start and on SIGTERM the plug is forced OFF. Consecutive ON is capped (`MAX_ON_SECS`, default 30 min) with a cooldown. OCR values outside `HUMIDITY_PLAUSIBLE_MIN`–`MAX`, jumps above `MAX_HUMIDITY_JUMP`, or a large remote/local split are treated as failed reads. `caffeinate -i -w` is started as a **child** of Python.app so idle sleep is asserted without breaking Local Network.

Plug toggles use `/usr/bin/shortcuts run` (not the URL scheme). After an ON command, the next few humidity reads must rise by `ON_VERIFY_MIN_RISE` or you get an ntfy warning. Camera frames use unique temp files. Waits use wall-clock so a laptop sleep does not fire a burst of cycles.

## Crop (remote OUT/CH1 only)

The live Tapo stream is 2304×1296. `FFMPEG_CROP` is ffmpeg `w:h:x:y`. The checked-in default `640:380:580:170` is the top LCD (temperature + remote humidity), not the indoor IN row.

If the camera moves, grab a still and try a new box:

```bash
# one still (bot already has LAN access; or copy from /tmp)
ffmpeg -i debug_full_frame.jpg -vf crop=640:380:580:170 preview.jpg
open preview.jpg
```

Then set `FFMPEG_CROP` in `.env` and restart the LaunchAgent.

## Manual run

```bash
python3 -m pip install -r requirements.txt
python3 humidifier_bot_v2.py
```
