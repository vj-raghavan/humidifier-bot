#!/usr/bin/env python3
"""Unit tests for humidifier_ops (no network, no shortcuts)."""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
import humidifier_ops as ops


class PauseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "humidifier_pause.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_until_cleared_stays_active(self):
        ops.save_pause(self.path, reason="refill tank", source="empty_tank", until=None)
        pause = ops.load_pause(self.path)
        self.assertTrue(ops.pause_is_active(pause, datetime(2026, 9, 6, 12, 0)))
        self.assertIn("until cleared", ops.pause_expiry_label(pause))

    def test_expires_at_until(self):
        until = datetime(2026, 9, 6, 15, 0)
        ops.save_pause(
            self.path,
            reason="hold off",
            source="manual",
            until=until.isoformat(timespec="seconds"),
        )
        pause = ops.load_pause(self.path)
        self.assertTrue(ops.pause_is_active(pause, datetime(2026, 9, 6, 14, 59)))
        self.assertFalse(ops.pause_is_active(pause, datetime(2026, 9, 6, 15, 0)))

    def test_clear_pause(self):
        ops.save_pause(self.path, reason="x", source="manual", until=None)
        self.assertTrue(ops.clear_pause(self.path))
        self.assertFalse(ops.pause_is_active(ops.load_pause(self.path)))

    def test_parse_specs(self):
        now = datetime(2026, 9, 6, 10, 0, 0)
        until, _ = ops.parse_pause_spec("3h", now=now)
        self.assertEqual(until, "2026-09-06T13:00:00")
        until, _ = ops.parse_pause_spec("90m", now=now)
        self.assertEqual(until, "2026-09-06T11:30:00")
        until, _ = ops.parse_pause_spec("until-cleared", now=now)
        self.assertIsNone(until)
        until, _ = ops.parse_pause_spec("tomorrow", now=now, start_hour=7)
        self.assertEqual(until, "2026-09-07T07:00:00")

    def test_cli_pause_resume(self):
        buf = []
        class W:
            def write(self, s):
                buf.append(s)
        rc = ops.cli(["pause", "2h", "bedtime"], pause_path=self.path, stdout=W())
        self.assertEqual(rc, 0)
        self.assertTrue(ops.pause_is_active(ops.load_pause(self.path), datetime.now()))
        rc = ops.cli(["resume"], pause_path=self.path, stdout=W())
        self.assertEqual(rc, 0)
        self.assertFalse(ops.pause_is_active(ops.load_pause(self.path)))

    def test_empty_tank_threshold(self):
        self.assertFalse(ops.empty_tank_should_pause(2, 3))
        self.assertTrue(ops.empty_tank_should_pause(3, 3))


class VisionClassifyTests(unittest.TestCase):
    def test_http(self):
        self.assertEqual(ops.classify_http_status(503), ops.VISION_HOST_DOWN)
        self.assertEqual(ops.classify_http_status(404), ops.VISION_SOFT)
        self.assertEqual(ops.classify_http_status(200), ops.VISION_SOFT)

    def test_exceptions(self):
        self.assertEqual(ops.classify_vision_exception(TimeoutError("timed out")), ops.VISION_HOST_DOWN)
        self.assertEqual(ops.classify_vision_exception(ConnectionRefusedError()), ops.VISION_HOST_DOWN)
        self.assertEqual(ops.classify_vision_exception(ValueError("bad json")), ops.VISION_SOFT)

    def test_crop_reasons(self):
        self.assertTrue(ops.is_crop_like_reason("remote 12% vs local 80% delta > 40"))
        self.assertTrue(ops.is_crop_like_reason("remote_humidity 3% outside 10-95%"))
        self.assertTrue(ops.is_crop_like_reason("Could not parse vision response"))
        self.assertFalse(ops.is_crop_like_reason("ffmpeg timed out"))
        self.assertFalse(ops.is_crop_like_reason("host unreachable"))


class DigestTests(unittest.TestCase):
    def test_thin_history_still_useful(self):
        history = {"readings": [], "last_state": "OFF"}
        title, body = ops.build_digest_message(history, when=datetime(2026, 9, 6, 7), kind="morning")
        self.assertEqual(title, "Morning digest")
        self.assertIn("no readings stored", body)
        self.assertIn("ON time: 0m", body)
        self.assertIn("Pause: none", body)

    def test_stats_and_rh(self):
        history = {
            "readings": [
                {"time": "2026-09-06T08:00:00", "remote_humidity": 40},
                {"time": "2026-09-06T09:00:00", "remote_humidity": 50},
                {"time": "2026-09-05T09:00:00", "remote_humidity": 99},
            ]
        }
        ops.bump_stat(history, "failed_reads", 4, when=datetime(2026, 9, 6))
        ops.bump_stat(history, "on_seconds", 5400, when=datetime(2026, 9, 6))
        ops.bump_stat(history, "empty_tank", 1, when=datetime(2026, 9, 6))
        _, body = ops.build_digest_message(history, when=datetime(2026, 9, 6, 20), kind="evening")
        self.assertIn("ON time: 1h 30m", body)
        self.assertIn("avg 45%", body)
        self.assertIn("empty-tank: 1", body)

    def test_idempotent_digest_due(self):
        self.assertTrue(ops.digest_due({}, kind="morning", day="2026-09-06", hour=7, target_hour=7))
        self.assertFalse(ops.digest_due({}, kind="morning", day="2026-09-06", hour=6, target_hour=7))
        sent = ops.mark_digest_sent({}, "morning", "2026-09-06")
        self.assertFalse(ops.digest_due(sent, kind="morning", day="2026-09-06", hour=8, target_hour=7))
        self.assertTrue(ops.digest_due(sent, kind="evening", day="2026-09-06", hour=20, target_hour=20))
        self.assertFalse(ops.digest_due({}, kind="morning", day="2026-09-06", hour=7, target_hour=-1))


class ForceSyncTests(unittest.TestCase):
    def test_triggers_when_off_but_humidity_high(self):
        self.assertTrue(
            ops.should_force_off_sync(
                desired_state="OFF",
                current_state="OFF",
                humidity=66,
                humidity_high=60,
                margin=5,
                now_ts=1000,
                last_force_ts=0,
                interval=900,
            )
        )

    def test_respects_interval_and_state(self):
        self.assertFalse(
            ops.should_force_off_sync(
                desired_state="OFF",
                current_state="OFF",
                humidity=66,
                humidity_high=60,
                margin=5,
                now_ts=1000,
                last_force_ts=200,
                interval=900,
            )
        )
        self.assertFalse(
            ops.should_force_off_sync(
                desired_state="ON",
                current_state="OFF",
                humidity=80,
                humidity_high=60,
                margin=5,
                now_ts=10_000,
                last_force_ts=0,
                interval=900,
            )
        )
        self.assertFalse(
            ops.should_force_off_sync(
                desired_state="OFF",
                current_state="OFF",
                humidity=64,
                humidity_high=60,
                margin=5,
                now_ts=10_000,
                last_force_ts=0,
                interval=900,
            )
        )


class RateLimiterTests(unittest.TestCase):
    def test_interval(self):
        rl = ops.RateLimiter()
        self.assertTrue(rl.allow("host", 60, now_ts=100))
        self.assertFalse(rl.allow("host", 60, now_ts=150))
        self.assertTrue(rl.allow("host", 60, now_ts=160))
        self.assertTrue(rl.allow("other", 60, now_ts=161))


class CropDriftLoopSignals(unittest.TestCase):
    """Sanity that crop-like streaks would fire, host-down would not."""

    def test_streak_policy(self):
        reasons = [
            "remote 11% vs local 70% delta > 40",
            "No valid humidity in response",
            "remote_humidity 120% outside 10-95%",
        ]
        streak = 0
        for r in reasons:
            if ops.is_crop_like_reason(r):
                streak += 1
            else:
                streak = 0
        self.assertEqual(streak, 3)
        self.assertFalse(ops.is_crop_like_reason("Vision host unreachable"))


if __name__ == "__main__":
    unittest.main()
