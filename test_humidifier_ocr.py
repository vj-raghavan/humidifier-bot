#!/usr/bin/env python3
"""Unit tests for ThermoPro OCR parse helpers (no camera, no Vision/Tesseract)."""

import unittest
from unittest import mock

import humidifier_ocr as ocr
import humidifier_ops as ops


class ParseThermoproOcrTests(unittest.TestCase):
    def test_percent_token(self):
        reading, why = ocr.parse_thermopro_ocr("55%")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertIsNone(reading["remote_temp"])
        self.assertEqual(reading["ocr_method"], "percent")

    def test_temp_and_humidity_text(self):
        reading, why = ocr.parse_thermopro_ocr("21.5\n55 %\nOUT CH1")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(reading["remote_temp"], 21.5)

    def test_two_integers_layout_last_is_humidity(self):
        reading, why = ocr.parse_thermopro_ocr("22 58")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 58)
        self.assertEqual(reading["remote_temp"], 22)

    def test_lone_temperature_rejected(self):
        reading, why = ocr.parse_thermopro_ocr("21.5")
        self.assertIsNone(reading)
        self.assertIn("temperature", why)

        reading, why = ocr.parse_thermopro_ocr("21")
        self.assertIsNone(reading)
        self.assertIn("temperature", why)

    def test_lone_humidity_integer_accepted(self):
        reading, why = ocr.parse_thermopro_ocr("55")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(reading["ocr_method"], "single_rh")

    def test_no_digits(self):
        reading, why = ocr.parse_thermopro_ocr("OUT CH1")
        self.assertIsNone(reading)
        self.assertIn("no numeric", why)

    def test_conflicting_percents_ambiguous(self):
        reading, why = ocr.parse_thermopro_ocr("40% 70%")
        self.assertIsNone(reading)
        self.assertIn("multiple %", why)

    def test_vision_boxes_humidity_is_lower_row(self):
        observations = [
            {"text": "22.4", "confidence": 0.92, "x": 0.2, "y": 0.65, "w": 0.4, "h": 0.2, "origin": "vision"},
            {"text": "47", "confidence": 0.88, "x": 0.2, "y": 0.15, "w": 0.3, "h": 0.25, "origin": "vision"},
            {"text": "%", "confidence": 0.8, "x": 0.55, "y": 0.18, "w": 0.1, "h": 0.1, "origin": "vision"},
        ]
        reading, why = ocr.parse_thermopro_ocr("", observations)
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 47)
        self.assertEqual(reading["remote_temp"], 22.4)
        self.assertGreaterEqual(reading["ocr_confidence"], 0.8)

    def test_plausibility_and_confidence_gate(self):
        reading, _ = ocr.parse_thermopro_ocr("55%")
        ok, reason = ocr.accept_ocr_reading(
            reading,
            min_confidence=0.5,
            prev_humidity=50,
            plausible_min=10,
            plausible_max=95,
            max_jump=15,
            remote_local_max_delta=40,
            plausibility_fn=ops.reading_is_plausible,
        )
        self.assertTrue(ok, reason)

        reading["ocr_confidence"] = 0.2
        ok, reason = ocr.accept_ocr_reading(
            reading,
            min_confidence=0.5,
            prev_humidity=50,
            plausible_min=10,
            plausible_max=95,
            max_jump=15,
            remote_local_max_delta=40,
            plausibility_fn=ops.reading_is_plausible,
        )
        self.assertFalse(ok)
        self.assertIn("confidence", reason)

        reading, _ = ocr.parse_thermopro_ocr("90%")
        ok, reason = ocr.accept_ocr_reading(
            reading,
            min_confidence=0.1,
            prev_humidity=50,
            plausible_min=10,
            plausible_max=95,
            max_jump=15,
            remote_local_max_delta=40,
            plausibility_fn=ops.reading_is_plausible,
        )
        self.assertFalse(ok)
        self.assertIn("jumped", reason)

    def test_out_of_range_percent_not_used_as_humidity(self):
        reading, why = ocr.parse_thermopro_ocr("99%")
        # 99 is within default 10-95? 99 > 95 so % filter excludes it.
        self.assertIsNone(reading)
        self.assertTrue(why.startswith("ocr:"))

    def test_tesseract_tsv_words(self):
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t10\t10\t40\t20\t90\t21.5\n"
            "5\t1\t1\t1\t2\t1\t10\t80\t30\t20\t85\t55\n"
            "5\t1\t1\t1\t2\t2\t45\t80\t12\t18\t80\t%\n"
        )
        obs = ocr._parse_tesseract_tsv(tsv)
        self.assertEqual(len(obs), 3)
        reading, why = ocr.parse_thermopro_ocr("", obs)
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(reading["remote_temp"], 21.5)

    def test_normalize_read_mode(self):
        self.assertEqual(ocr.normalize_read_mode("OCR-FIRST"), "ocr_first")
        self.assertEqual(ocr.normalize_read_mode("nope", default="ocr_first"), "ocr_first")


class DetectBackendTests(unittest.TestCase):
    def test_none_on_linux_without_tesseract(self):
        with mock.patch.object(ocr, "_pyobjc_vision_available", return_value=False), mock.patch.object(
            ocr, "_swift_available", return_value=False
        ), mock.patch.object(ocr, "_which", return_value=None):
            backend, detail = ocr.detect_ocr_backend()
            self.assertIsNone(backend)
            self.assertEqual(ocr.backend_label(backend, detail), "none")

    def test_prefer_tesseract(self):
        with mock.patch.object(ocr, "_pyobjc_vision_available", return_value=True), mock.patch.object(
            ocr, "_which", return_value="/opt/homebrew/bin/tesseract"
        ):
            backend, detail = ocr.detect_ocr_backend(prefer="tesseract")
            self.assertEqual(backend, "tesseract")
            self.assertEqual(detail["bin"], "/opt/homebrew/bin/tesseract")


if __name__ == "__main__":
    unittest.main()
