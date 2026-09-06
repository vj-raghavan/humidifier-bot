#!/usr/bin/env python3
"""Shared helpers for humidifier_bot_v2: pause, digest, vision errors, alerts.

Import-safe (no dotenv, no logging handlers). The live bot and unit tests both
use this module. CLI: ``python3 humidifier_bot_v2.py pause|resume|status ...``
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta

VISION_OK = "ok"
VISION_SOFT = "soft"
VISION_HOST_DOWN = "host_down"

HOST_DOWN_HTTP = frozenset({408, 429, 500, 502, 503, 504})

_DURATION_RE = re.compile(
    r"^\s*(?:(\d+(?:\.\d+)?)\s*h(?:ours?)?)?\s*(?:(\d+(?:\.\d+)?)\s*m(?:in(?:utes?)?)?)?\s*(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?)?\s*$",
    re.I,
)


def atomic_write_json(path, data):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_json_file(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else default
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


# --- Vision failure classification ---


def classify_http_status(status_code):
    if status_code is None:
        return VISION_SOFT
    if int(status_code) in HOST_DOWN_HTTP or int(status_code) >= 500:
        return VISION_HOST_DOWN
    return VISION_SOFT


def classify_vision_exception(exc):
    """Distinguish unreachable vision host from a soft/parse failure."""
    if exc is None:
        return VISION_SOFT
    name = type(exc).__name__
    module = type(exc).__module__ or ""
    msg = str(exc).lower()

    timeout_names = {"Timeout", "ConnectTimeout", "ReadTimeout", "TimeoutError"}
    conn_names = {
        "ConnectionError",
        "ConnectError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "NewConnectionError",
        "MaxRetryError",
        "ProtocolError",
        "SSLError",
        "ProxyError",
        "ChunkedEncodingError",
    }
    if name in timeout_names or name in conn_names:
        return VISION_HOST_DOWN
    if "requests" in module and name in {"Timeout", "ConnectionError"}:
        return VISION_HOST_DOWN
    if any(
        token in msg
        for token in (
            "connection refused",
            "failed to establish",
            "name or service not known",
            "nodename nor servname",
            "network is unreachable",
            "connection reset",
            "timed out",
            "timeout",
            "temporarily unavailable",
        )
    ):
        return VISION_HOST_DOWN
    return VISION_SOFT


def is_crop_like_reason(reason):
    """OCR junk / range / remote-local split — not host-down or camera timeout."""
    if not reason:
        return False
    r = reason.lower()
    needles = (
        "delta",
        "remote ",
        "local",
        "outside",
        "jumped",
        "missing",
        "no valid humidity",
        "could not parse",
        "implausible",
        "plausible",
        "unparseable",
        "ocr",
    )
    return any(n in r for n in needles)


# --- Pause / override / empty-tank ---


def load_pause(path):
    data = load_json_file(path, None)
    if not data:
        return None
    if not isinstance(data.get("reason"), str):
        data["reason"] = data.get("reason") or "paused"
    return data


def save_pause(path, *, reason, source, until=None, extra=None):
    payload = {
        "until": until,
        "reason": reason,
        "source": source,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(path, payload)
    return payload


def clear_pause(path):
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


def parse_until_iso(until):
    if until is None or until in ("", "forever", "until-cleared"):
        return None
    if isinstance(until, datetime):
        return until
    text = str(until).strip()
    if text.endswith("Z"):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def pause_is_active(pause, now=None):
    if not pause:
        return False
    now = now or datetime.now()
    until = parse_until_iso(pause.get("until"))
    if until is None:
        return True
    return now < until


def pause_expiry_label(pause, now=None):
    if not pause:
        return "not paused"
    until = parse_until_iso(pause.get("until"))
    if until is None:
        return "until cleared (delete pause file or run resume)"
    now = now or datetime.now()
    if now >= until:
        return "expired"
    return f"until {until.isoformat(timespec='minutes')}"


def parse_pause_spec(spec, now=None, start_hour=7):
    """Return (until_iso_or_None, label). None until means until-cleared."""
    now = now or datetime.now()
    raw = (spec or "").strip().lower()
    if raw in ("until-cleared", "until_cleared", "forever", "clear", "cleared"):
        return None, "until cleared"
    if raw in ("tomorrow", "until-tomorrow", "until_tomorrow"):
        tomorrow = (now + timedelta(days=1)).date()
        until = datetime(tomorrow.year, tomorrow.month, tomorrow.day, int(start_hour), 0, 0)
        return until.isoformat(timespec="seconds"), f"until {until.isoformat(timespec='minutes')}"

    match = _DURATION_RE.match(raw.replace(" ", ""))
    if match and any(match.group(i) for i in (1, 2, 3)):
        hours = float(match.group(1) or 0)
        minutes = float(match.group(2) or 0)
        seconds = float(match.group(3) or 0)
        delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
        if delta.total_seconds() <= 0:
            raise ValueError(f"pause duration must be positive: {spec!r}")
        until = now + delta
        return until.isoformat(timespec="seconds"), f"until {until.isoformat(timespec='minutes')}"

    # Plain hours, e.g. "3"
    try:
        hours = float(raw)
        if hours <= 0:
            raise ValueError
        until = now + timedelta(hours=hours)
        return until.isoformat(timespec="seconds"), f"until {until.isoformat(timespec='minutes')}"
    except ValueError:
        pass

    # ISO timestamp
    parsed = parse_until_iso(spec)
    if parsed is not None:
        return parsed.isoformat(timespec="seconds"), f"until {parsed.isoformat(timespec='minutes')}"

    raise ValueError(
        f"unrecognized pause spec {spec!r}; use 3h, 90m, tomorrow, until-cleared, or an ISO time"
    )


def empty_tank_should_pause(consecutive_failed_verifies, threshold):
    return consecutive_failed_verifies >= max(1, int(threshold))


# --- Daily stats / digest ---


def day_key(when=None):
    when = when or datetime.now()
    if isinstance(when, datetime):
        return when.date().isoformat()
    return str(when)[:10]


def ensure_daily_bucket(history, day=None):
    day = day or day_key()
    daily = history.setdefault("daily", {})
    bucket = daily.setdefault(
        day,
        {
            "failed_reads": 0,
            "force_offs": 0,
            "empty_tank": 0,
            "on_seconds": 0,
            "crop_drift_alerts": 0,
            "host_down": 0,
            "force_sync_offs": 0,
        },
    )
    if len(daily) > 21:
        for old in sorted(daily)[:-21]:
            del daily[old]
    return bucket


def bump_stat(history, key, amount=1, when=None):
    bucket = ensure_daily_bucket(history, day_key(when))
    bucket[key] = bucket.get(key, 0) + amount
    return bucket[key]


def rh_stats_for_day(readings, day):
    values = []
    for entry in readings or []:
        t = entry.get("time") or ""
        if not str(t).startswith(day):
            continue
        rh = entry.get("remote_humidity")
        if isinstance(rh, (int, float)):
            values.append(float(rh))
    if not values:
        return None
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def format_hours(seconds):
    seconds = max(0, int(seconds or 0))
    h, rem = divmod(seconds, 3600)
    m = rem // 60
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"


def build_digest_message(
    history,
    *,
    when=None,
    kind="morning",
    pause=None,
    extra_on_seconds=0,
):
    """Build a short ntfy body. Thin history still yields a useful line."""
    when = when or datetime.now()
    day = day_key(when)
    bucket = ensure_daily_bucket(history, day)
    rh = rh_stats_for_day(history.get("readings"), day)
    on_secs = int(bucket.get("on_seconds") or 0) + int(extra_on_seconds or 0)
    pause_line = "none"
    if pause_is_active(pause, when):
        src = pause.get("source") or "manual"
        pause_line = f"{src} ({pause_expiry_label(pause, when)}): {pause.get('reason')}"

    title = "Morning digest" if kind == "morning" else "Evening digest"
    lines = [f"{title} {day}"]
    lines.append(f"ON time: {format_hours(on_secs)}")
    if rh:
        lines.append(
            f"RH (n={rh['count']}): avg {rh['avg']:.0f}%  min {rh['min']:.0f}%  max {rh['max']:.0f}%"
        )
    else:
        n = len(history.get("readings") or [])
        lines.append(f"RH: no readings stored for today (history has {n} total)")
    lines.append(
        f"Failed reads: {bucket.get('failed_reads', 0)}  "
        f"force-OFF: {bucket.get('force_offs', 0)}  "
        f"empty-tank: {bucket.get('empty_tank', 0)}  "
        f"host-down: {bucket.get('host_down', 0)}"
    )
    lines.append(f"Pause: {pause_line}")
    return title, "\n".join(lines)


def digest_due(last_sent, *, kind, day, hour, target_hour):
    """True once per calendar day when current hour has reached the slot."""
    if target_hour is None or int(target_hour) < 0:
        return False
    if hour < int(target_hour):
        return False
    sent = (last_sent or {}).get(kind)
    return sent != day


def mark_digest_sent(last_sent, kind, day):
    data = dict(last_sent or {})
    data[kind] = day
    if len(data) > 8:
        # keep only known keys
        data = {k: v for k, v in data.items() if k in ("morning", "evening") or k.endswith("_note")}
    return data


# --- Force-sync / state drift ---


def should_force_off_sync(
    *,
    desired_state,
    current_state,
    humidity,
    humidity_high,
    margin,
    now_ts,
    last_force_ts,
    interval,
):
    if desired_state != "OFF" or current_state != "OFF":
        return False
    if not isinstance(humidity, (int, float)):
        return False
    if humidity < humidity_high + margin:
        return False
    last = last_force_ts or 0
    return (now_ts - last) >= interval


# --- Rate limits ---


class RateLimiter:
    def __init__(self):
        self._last = {}

    def allow(self, key, interval_secs, now_ts=None):
        now_ts = time_now(now_ts)
        last = self._last.get(key)
        if last is not None and (now_ts - last) < interval_secs:
            return False
        self._last[key] = now_ts
        return True

    def peek(self, key):
        return self._last.get(key)


def time_now(now_ts=None):
    if now_ts is not None:
        return now_ts
    import time as _time

    return _time.time()


# --- CLI (Shortcuts-friendly) ---


def cli(argv, *, pause_path, start_hour=7, stdout=None):
    """pause / resume / status. Returns process exit code."""
    import sys

    out = stdout if stdout is not None else sys.stdout
    if not argv:
        out.write("usage: humidifier_bot_v2.py pause <3h|90m|tomorrow|until-cleared>\n")
        out.write("       humidifier_bot_v2.py resume\n")
        out.write("       humidifier_bot_v2.py status\n")
        return 2
    cmd = argv[0].lower()
    if cmd in ("resume", "unpause", "clear"):
        existed = clear_pause(pause_path)
        out.write(f"pause cleared ({pause_path})\n" if existed else "no pause file\n")
        return 0
    if cmd == "status":
        pause = load_pause(pause_path)
        if pause_is_active(pause):
            out.write(f"PAUSED {pause_expiry_label(pause)} source={pause.get('source')} reason={pause.get('reason')}\n")
        elif pause:
            out.write("pause file present but expired — will be ignored\n")
        else:
            out.write("not paused\n")
        return 0
    if cmd == "pause":
        spec = argv[1] if len(argv) > 1 else "until-cleared"
        reason = " ".join(argv[2:]) if len(argv) > 2 else "manual pause"
        try:
            until, label = parse_pause_spec(spec, start_hour=start_hour)
        except ValueError as e:
            out.write(str(e) + "\n")
            return 2
        save_pause(pause_path, reason=reason, source="manual", until=until)
        out.write(f"paused {label}: {reason}\n")
        out.write(f"file: {pause_path}\n")
        return 0
    out.write(f"unknown command {cmd!r}\n")
    return 2
