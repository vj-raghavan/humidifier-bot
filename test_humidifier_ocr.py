#!/usr/bin/env python3
"""Unit tests for ThermoPro OCR parse helpers (no camera, no Vision/Tesseract)."""

import os
import tempfile
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

    def test_colon_token_is_humidity(self):
        reading, why = ocr.parse_thermopro_ocr("79:")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 79)
        self.assertEqual(reading["ocr_method"], "percent")

    def test_percent_seventy_nine(self):
        reading, why = ocr.parse_thermopro_ocr("79%")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 79)
        self.assertEqual(reading["ocr_method"], "percent")

    def test_vision_colon_over_digit_soup(self):
        reading, why = ocr.parse_thermopro_ocr("I0 79: 16956")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 79)

        observations = [
            {
                "text": "I0",
                "confidence": 0.4,
                "x": 0.05,
                "y": 0.72,
                "w": 0.15,
                "h": 0.12,
                "origin": "vision",
            },
            {
                "text": "79:",
                "confidence": 0.94,
                "x": 0.25,
                "y": 0.42,
                "w": 0.28,
                "h": 0.18,
                "origin": "vision",
            },
            {
                "text": "16956",
                "confidence": 0.35,
                "x": 0.15,
                "y": 0.08,
                "w": 0.55,
                "h": 0.22,
                "origin": "vision",
            },
        ]
        reading, why = ocr.parse_thermopro_ocr("I0 79: 16956", observations)
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 79)
        self.assertGreaterEqual(reading["ocr_confidence"], 0.9)

    def test_four_number_layout_prefers_out_humidity(self):
        reading, why = ocr.parse_thermopro_ocr("22.5 55 21.0 48")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(reading["remote_temp"], 22.5)
        self.assertEqual(reading["ocr_method"], "layout")

        reading, why = ocr.parse_thermopro_ocr("22 55 21 48")
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(reading["remote_temp"], 22)

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


class CropAndPreprocessTests(unittest.TestCase):
    def test_parse_and_default_rh_crop(self):
        self.assertEqual(ocr.parse_crop_spec("360:140:160:150"), (360, 140, 160, 150))
        self.assertIsNone(ocr.parse_crop_spec(""))
        self.assertIsNone(ocr.parse_crop_spec("nope"))
        spec = ocr.default_rh_crop(640, 420)
        w, h, x, y = ocr.parse_crop_spec(spec)
        self.assertEqual((w, h, x, y), (461, 143, 102, 151))
        self.assertEqual(ocr.resolve_rh_crop("360:140:160:150", 640, 420), "360:140:160:150")
        self.assertEqual(ocr.resolve_rh_crop("", 640, 420), spec)
        # Clamp a box that hangs off the right/bottom edge.
        self.assertEqual(ocr.resolve_rh_crop("400:200:400:300", 640, 420), "240:120:400:300")

    def test_vf_filter_order(self):
        filters = ocr.ocr_vf_filters(
            crop="360:140:160:150",
            preprocess=True,
            upscale=2,
            contrast=1.6,
            threshold=160,
            invert=True,
        )
        self.assertEqual(filters[0], "crop=360:140:160:150")
        self.assertTrue(filters[1].startswith("scale=iw*2"))
        self.assertEqual(filters[2], "format=gray")
        self.assertEqual(filters[3], "eq=contrast=1.6")
        self.assertEqual(filters[4], "negate")
        self.assertIn("160", filters[5])
        self.assertEqual(
            ocr.ocr_vf_filters(crop="10:10:0:0", preprocess=False),
            ["crop=10:10:0:0"],
        )

    def test_pillow_preprocess_upscale_gray(self):
        from PIL import Image

        fd, src = tempfile.mkstemp(prefix="ocr_src_", suffix=".png")
        os.close(fd)
        fd, dest = tempfile.mkstemp(prefix="ocr_dst_", suffix=".png")
        os.close(fd)
        try:
            Image.new("RGB", (40, 20), color=(200, 30, 30)).save(src)
            path, is_temp, err = ocr.prepare_ocr_frame(
                src,
                dest,
                crop="20:10:10:5",
                preprocess=True,
                upscale=2,
                contrast=1.0,
                threshold=0,
                invert=False,
                impl="pillow",
            )
            self.assertIsNone(err)
            self.assertEqual(path, dest)
            with Image.open(dest) as out:
                self.assertEqual(out.size, (40, 20))  # 20x10 crop, then 2×
                self.assertEqual(out.mode, "L")
        finally:
            for p in (src, dest):
                try:
                    os.remove(p)
                except OSError:
                    pass


class ConsensusHumidityTests(unittest.TestCase):
    def _rh(self, value, conf=0.9):
        return {
            "remote_humidity": value,
            "remote_temp": 21,
            "ocr_confidence": conf,
            "ocr_method": "percent",
        }

    def test_median_when_all_close(self):
        reading, why, detail = ocr.consensus_humidity(
            [self._rh(54), self._rh(55), self._rh(56)],
            min_agree=2,
            max_delta=2,
        )
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 55)
        self.assertEqual(detail["method"], "median")
        self.assertEqual(detail["agree"], 3)

    def test_cluster_ignores_outlier(self):
        reading, why, detail = ocr.consensus_humidity(
            [self._rh(55), self._rh(56), self._rh(80)],
            min_agree=2,
            max_delta=2,
        )
        self.assertEqual(why, "ok")
        self.assertIn(reading["remote_humidity"], (55, 56))
        self.assertEqual(detail["method"], "cluster")
        self.assertEqual(detail["agree"], 2)

    def test_failed_when_spread_too_wide(self):
        reading, why, detail = ocr.consensus_humidity(
            [self._rh(40), self._rh(70), self._rh(90)],
            min_agree=2,
            max_delta=2,
        )
        self.assertIsNone(reading)
        self.assertIn("ocr consensus", why)
        self.assertEqual(detail["agree"], 1)

    def test_none_frames_and_partial(self):
        reading, why, _ = ocr.consensus_humidity([None, None, None], min_agree=2, max_delta=2)
        self.assertIsNone(reading)
        self.assertIn("no humidity", why)

        reading, why, detail = ocr.consensus_humidity(
            [None, self._rh(55), self._rh(56)],
            min_agree=2,
            max_delta=2,
        )
        self.assertEqual(why, "ok")
        self.assertIn(reading["remote_humidity"], (55, 56))
        self.assertEqual(detail["agree"], 2)

    def test_single_reading_min_agree_clamped(self):
        reading, why, detail = ocr.consensus_humidity([self._rh(58)], min_agree=2, max_delta=2)
        self.assertEqual(why, "ok")
        self.assertEqual(reading["remote_humidity"], 58)
        self.assertEqual(detail["min_agree"], 1)


if __name__ == "__main__":
    unittest.main()
