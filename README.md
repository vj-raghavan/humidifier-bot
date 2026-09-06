# Humidifier Bot

Humidity-aware plant humidifier control. A camera frame of a ThermoPro display is read by a local vision LLM; a HomeKit smart plug is toggled via macOS Shortcuts (`PH On` / `PH Off`).

Runtime settings live in `.env` (not committed). Copy `.env.example` and fill in camera, vision, and ntfy values. Do not commit `.env`.

The live process is `humidifier_bot_v2.py` (LaunchAgent, single process). `humidifier_bot.py` is the older timed-cycle bot and is not used.

## Always-on (macOS)

This package is the live bot, started by LaunchAgent `com.samwise.humidifier-bot` (`KeepAlive`, same `Python.app` binary as the original agent so Local Network / RTSP keep working).

```bash
cp com.samwise.humidifier-bot.plist ~/Library/LaunchAgents/
launchctl bootout gui/$(id -u)/com.samwise.humidifier-bot 2>/dev/null || true
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.samwise.humidifier-bot.plist
```

Logs: `humidifier_v2.log` in this directory, plus `/tmp/humidifier-bot.log` and `/tmp/humidifier-bot.err`.

On start and on SIGTERM the plug is **always commanded OFF** (`FORCE_SYNC`), even if the bot already thought it was off. Consecutive ON is capped (`MAX_ON_SECS`, default 30 min) with a cooldown. OCR values outside `HUMIDITY_PLAUSIBLE_MIN`–`MAX`, jumps above `MAX_HUMIDITY_JUMP`, or a large remote/local split are treated as failed reads. After `MAX_CONSECUTIVE_FAILURES` failed reads the plug is forced OFF; the loop keeps retrying. `caffeinate -i -w` is started as a **child** of Python.app so idle sleep is asserted without breaking Local Network.

Plug toggles use `/usr/bin/shortcuts run` (not the URL scheme). After an ON command, the next few humidity reads must rise by `ON_VERIFY_MIN_RISE` or you get an ntfy warning (and, after enough consecutive failed verifies, an empty-tank pause — below). Camera frames use unique temp files. Waits use wall-clock so a laptop sleep does not fire a burst of cycles.

If humidity stays high (`>= HUMIDITY_HIGH + FORCE_OFF_SYNC_MARGIN`) while the bot already believes the plug is OFF, it periodically re-sends OFF (`FORCE_OFF_SYNC_INTERVAL`, default 900s) in case HomeKit drifted.

## Manual pause / override

Durable file: `humidifier_pause.json` in the repo directory (or `PAUSE_FILE`). While active the bot keeps the plug OFF, skips ON decisions, logs occasionally, and still runs fail-safe OFF.

```bash
# hold OFF for 3 hours
python3 humidifier_bot_v2.py pause 3h

# 90 minutes, optional reason
python3 humidifier_bot_v2.py pause 90m guests over

# until next calendar day’s START_HOUR
python3 humidifier_bot_v2.py pause tomorrow

# until you explicitly clear it
python3 humidifier_bot_v2.py pause until-cleared

python3 humidifier_bot_v2.py status
python3 humidifier_bot_v2.py resume
```

Clearing is any of: `resume`, deleting the pause file, or waiting until `until` expires. Shortcuts can run the same `python3 … pause 3h` / `resume` commands.

Example file:

```json
{
  "until": "2026-09-06T16:00:00",
  "reason": "manual pause",
  "source": "manual",
  "created": "2026-09-06T13:00:00"
}
```

`"until": null` means until cleared.

## Empty-tank / no-effect

If ON is commanded but humidity does not rise by `ON_VERIFY_MIN_RISE` within `ON_VERIFY_CHECKS`, that window is a failed verify. After `EMPTY_TANK_FAILS` consecutive failed windows (default 3) the bot:

1. Forces the plug OFF
2. Writes `humidifier_pause.json` with `source=empty_tank` and `until=null`
3. ntfy: refill the tank / check the plug

It will **not** turn ON again until you clear the pause (`resume` or delete the file). After you resume, a later ON that actually raises humidity resets the failed-verify counter.

## Vision host (vLLM on another machine)

`VISION_API_BASE` may be a Mac/LM Studio box. Connection errors, timeouts, and 5xx/429 are **host-down**: at most `VISION_HOST_DOWN_RETRIES` same-frame attempts, then a long sleep (`VISION_HOST_DOWN_BACKOFF`, default 180s) so the bot does not hammer a sleeping laptop. Optional rate-limited ntfy (`VISION_HOST_DOWN_NTFY`). When the host answers again, the normal `CHECK_INTERVAL` resumes.

Bad OCR / unparseable JSON / implausible RH stay **soft** failures: same-frame retries (`VISION_SOFT_RETRIES`) and the usual consecutive-read safety OFF.

## Morning / evening digest

One ntfy near `DIGEST_MORNING_HOUR` (default `START_HOUR`) and `DIGEST_EVENING_HOUR` (default `END_HOUR`), idempotent per calendar day (`humidifier_digest.json`). Body is built from `humidity_history.json` plus daily counters (ON time, failed reads, force-OFFs, empty-tank, host-down, pause). Thin history still sends a short useful message. Set a slot’s hour to `-1` to disable it, or `DIGEST_ENABLED=false`.

## Crop drift helper

A streak of remote/local splits, out-of-range RH, or unparseable OCR (`CROP_DRIFT_STREAK`, default 4) saves dated JPEGs under `stills/` (cropped, and a full frame if `CROP_DRIFT_SAVE_FULL=true`) and ntfy with the current `FFMPEG_CROP` plus file paths. Alerts are rate-limited (`CROP_DRIFT_NTFY_SECS`).

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
python3 -m unittest test_humidifier_ops.py test_humidifier_bot_v2.py
```

## Tests

`humidifier_ops.py` holds pause/digest/vision-class/force-sync helpers so they can be tested without Shortcuts or a camera.
