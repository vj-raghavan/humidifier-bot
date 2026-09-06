#!/usr/bin/env python3
"""Mocked tests for humidifier_bot_v2 control helpers (no Shortcuts, no camera)."""

import unittest
from unittest import mock

import humidifier_bot_v2 as bot
import humidifier_ops as ops


class SetHumidifierForceSyncTests(unittest.TestCase):
    def test_skip_when_already_off_without_force(self):
        with mock.patch.object(bot, "run_shortcut") as run, mock.patch.object(bot, "send_ntfy") as ntfy:
            self.assertEqual(bot.set_humidifier("OFF", "OFF"), "OFF")
            run.assert_not_called()
            ntfy.assert_not_called()

    def test_force_sync_always_sends_off(self):
        with mock.patch.object(bot, "run_shortcut", return_value=True) as run, mock.patch.object(
            bot, "send_ntfy"
        ) as ntfy:
            self.assertEqual(bot.set_humidifier("OFF", "OFF", force=True), "OFF")
            run.assert_called_once_with(bot.SHORTCUT_OFF_NAME)
            ntfy.assert_not_called()

    def test_real_transition_notifies(self):
        with mock.patch.object(bot, "run_shortcut", return_value=True) as run, mock.patch.object(
            bot, "send_ntfy"
        ) as ntfy:
            self.assertEqual(bot.set_humidifier("OFF", "ON"), "OFF")
            run.assert_called_once_with(bot.SHORTCUT_OFF_NAME)
            ntfy.assert_called()

    def test_fail_safe_off_force_syncs(self):
        bot._runtime["state"] = "OFF"
        bot._runtime["history"] = {"readings": [], "last_state": "OFF"}
        bot._runtime["on_since"] = None
        with mock.patch.object(bot, "set_humidifier", return_value="OFF") as setter, mock.patch.object(
            bot, "persist_state"
        ):
            bot.fail_safe_off("test")
            setter.assert_called_once()
            args, kwargs = setter.call_args
            self.assertEqual(args[0], "OFF")
            self.assertTrue(kwargs.get("force") or (len(args) > 3 and args[3] is True))


class VisionReadTests(unittest.TestCase):
    def test_connection_error_is_host_down_without_soft_retries(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", side_effect=ConnectionError("refused")
        ) as post, mock.patch.object(bot.time, "sleep") as sleep:
            reading, kind = bot.read_humidity_from_image("/tmp/fake.jpg")
            self.assertIsNone(reading)
            self.assertEqual(kind, ops.VISION_HOST_DOWN)
            self.assertEqual(post.call_count, max(1, bot.VISION_HOST_DOWN_RETRIES))
            self.assertEqual(sleep.call_count, 0)

    def test_bad_json_is_soft(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {"choices": [{"message": {"content": "not json"}}]}
        with mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ), mock.patch.object(bot.time, "sleep"):
            reading, kind = bot.read_humidity_from_image("/tmp/fake.jpg")
            self.assertIsNone(reading)
            self.assertEqual(kind, ops.VISION_SOFT)

    def test_ok_reading(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"remote_humidity": 55, "remote_temp": 21, "local_humidity": null, "local_temp": null}'}}]
        }
        with mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ), mock.patch.object(bot.time, "sleep"):
            reading, kind = bot.read_humidity_from_image("/tmp/fake.jpg")
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["remote_humidity"], 55)


if __name__ == "__main__":
    unittest.main()
