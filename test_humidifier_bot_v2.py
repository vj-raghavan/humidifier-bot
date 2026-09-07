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


class FailedReadWaitTests(unittest.TestCase):
    def test_defaults_and_host_down_floor(self):
        with mock.patch.object(bot, "CHECK_INTERVAL", 60), mock.patch.object(
            bot, "FAILED_READ_RETRY_SECS", 0
        ), mock.patch.object(bot, "FAILED_READ_BACKOFF_SECS", 0), mock.patch.object(
            bot, "VISION_HOST_DOWN_BACKOFF", 180
        ):
            self.assertEqual(bot._failed_read_wait_secs(backoff=False), 60)
            self.assertEqual(bot._failed_read_wait_secs(backoff=True), 120)
            self.assertEqual(bot._failed_read_wait_secs(backoff=False, host_down=True), 180)
            self.assertEqual(bot._failed_read_wait_secs(backoff=True, host_down=True), 180)


class VisionReadTests(unittest.TestCase):
    def test_connection_error_is_host_down_without_soft_retries(self):
        with mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", side_effect=ConnectionError("refused")
        ) as post, mock.patch.object(bot, "sleep_seconds") as sleep:
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
        ), mock.patch.object(bot, "sleep_seconds") as sleep:
            reading, kind = bot.read_humidity_from_image("/tmp/fake.jpg")
            self.assertIsNone(reading)
            self.assertEqual(kind, ops.VISION_SOFT)
            self.assertEqual(sleep.call_count, 2)

    def test_ok_reading(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"remote_humidity": 55, "remote_temp": 21, "local_humidity": null, "local_temp": null}'}}]
        }
        with mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ), mock.patch.object(bot, "sleep_seconds"):
            reading, kind = bot.read_humidity_from_image("/tmp/fake.jpg")
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["remote_humidity"], 55)


class ResolveHumidityReadTests(unittest.TestCase):
    def test_ocr_first_skips_llm_when_parse_is_confident(self):
        ocr_result = {
            "text": "21.5 55%",
            "observations": [],
            "backend": "tesseract",
            "detail": {"bin": "tesseract"},
            "error": None,
        }
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "ocr_first"), mock.patch.object(
            bot.ocr, "ocr_image", return_value=ocr_result
        ), mock.patch.object(bot.requests, "post") as post:
            reading, kind = bot.resolve_humidity_reading("/tmp/fake.jpg", prev_humidity=50)
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["source"], "ocr")
            self.assertEqual(reading["remote_humidity"], 55)
            self.assertEqual(reading["remote_temp"], 21.5)
            post.assert_not_called()

    def test_ocr_first_falls_back_to_llm(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"remote_humidity": 44, "remote_temp": 20, "local_humidity": null, "local_temp": null}'}}]
        }
        ocr_result = {
            "text": "blur",
            "observations": [],
            "backend": "tesseract",
            "detail": {},
            "error": None,
        }
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "ocr_first"), mock.patch.object(
            bot.ocr, "ocr_image", return_value=ocr_result
        ), mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ) as post, mock.patch.object(bot, "sleep_seconds"):
            reading, kind = bot.resolve_humidity_reading("/tmp/fake.jpg", prev_humidity=50)
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["source"], "llm")
            self.assertEqual(reading["remote_humidity"], 44)
            post.assert_called()

    def test_ocr_only_does_not_call_llm(self):
        ocr_result = {
            "text": "nope",
            "observations": [],
            "backend": None,
            "detail": {},
            "error": "no local OCR backend",
        }
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "ocr_only"), mock.patch.object(
            bot.ocr, "ocr_image", return_value=ocr_result
        ), mock.patch.object(bot.requests, "post") as post:
            reading, kind = bot.resolve_humidity_reading("/tmp/fake.jpg")
            self.assertIsNone(reading)
            self.assertEqual(kind, ops.VISION_SOFT)
            post.assert_not_called()

    def test_llm_only_skips_ocr(self):
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": '{"remote_humidity": 41, "remote_temp": 19, "local_humidity": null, "local_temp": null}'}}]
        }
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "llm_only"), mock.patch.object(
            bot.ocr, "ocr_image"
        ) as ocr_image, mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ), mock.patch.object(bot, "sleep_seconds"):
            reading, kind = bot.resolve_humidity_reading("/tmp/fake.jpg")
            self.assertEqual(reading["source"], "llm")
            self.assertEqual(reading["remote_humidity"], 41)
            ocr_image.assert_not_called()
            self.assertEqual(kind, ops.VISION_OK)


class OcrConsensusResolveTests(unittest.TestCase):
    def test_two_of_three_agree_skips_llm(self):
        results = [
            {"text": "55%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
            {"text": "56%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
            {"text": "80%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
        ]
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "ocr_first"), mock.patch.object(
            bot, "OCR_CONSENSUS_MIN_AGREE", 2
        ), mock.patch.object(bot, "OCR_CONSENSUS_MAX_DELTA", 2), mock.patch.object(
            bot.ocr, "ocr_image", side_effect=results
        ), mock.patch.object(bot.requests, "post") as post:
            reading, kind = bot.resolve_humidity_reading(
                "/tmp/a.jpg",
                prev_humidity=54,
                ocr_frame_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"],
            )
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["source"], "ocr")
            self.assertIn(reading["remote_humidity"], (55, 56))
            post.assert_not_called()

    def test_disagreement_falls_back_to_llm(self):
        results = [
            {"text": "40%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
            {"text": "70%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
            {"text": "90%", "observations": [], "backend": "tesseract", "detail": {}, "error": None},
        ]
        resp = mock.Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"remote_humidity": 55, "remote_temp": 21, "local_humidity": null, "local_temp": null}'
                    }
                }
            ]
        }
        with mock.patch.object(bot, "HUMIDITY_READ_MODE", "ocr_first"), mock.patch.object(
            bot, "OCR_CONSENSUS_MIN_AGREE", 2
        ), mock.patch.object(bot, "OCR_CONSENSUS_MAX_DELTA", 2), mock.patch.object(
            bot.ocr, "ocr_image", side_effect=results
        ), mock.patch("builtins.open", mock.mock_open(read_data=b"jpeg")), mock.patch.object(
            bot.requests, "post", return_value=resp
        ) as post, mock.patch.object(bot, "sleep_seconds"):
            reading, kind = bot.resolve_humidity_reading(
                "/tmp/a.jpg",
                prev_humidity=54,
                ocr_frame_paths=["/tmp/a.jpg", "/tmp/b.jpg", "/tmp/c.jpg"],
            )
            self.assertEqual(kind, ops.VISION_OK)
            self.assertEqual(reading["source"], "llm")
            self.assertEqual(reading["remote_humidity"], 55)
            post.assert_called()


class CaptureCycleTests(unittest.TestCase):
    def test_n_frames_and_gaps(self):
        with mock.patch.object(bot, "OCR_CONSENSUS_FRAMES", 3), mock.patch.object(
            bot, "OCR_CONSENSUS_GAP_SECS", 0.5
        ), mock.patch.object(
            bot, "capture_frame", side_effect=["/a.jpg", "/b.jpg", "/c.jpg"]
        ) as cap, mock.patch.object(bot, "sleep_seconds") as sleep:
            paths = bot.capture_cycle_frames()
            self.assertEqual(paths, ["/a.jpg", "/b.jpg", "/c.jpg"])
            self.assertEqual(cap.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            sleep.assert_called_with(0.5)


if __name__ == "__main__":
    unittest.main()
