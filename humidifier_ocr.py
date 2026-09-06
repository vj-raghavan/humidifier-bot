#!/usr/bin/env python3
"""Local OCR for cropped ThermoPro LCD frames (Apple Vision, then Tesseract).

Import-safe: no dotenv, no logging handlers. The live bot detects backends at
runtime and falls back to the vision LLM when OCR is missing or low-confidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

READ_MODE_OCR_FIRST = "ocr_first"
READ_MODE_OCR_ONLY = "ocr_only"
READ_MODE_LLM_ONLY = "llm_only"
READ_MODES = (READ_MODE_OCR_FIRST, READ_MODE_OCR_ONLY, READ_MODE_LLM_ONLY)

BACKEND_VISION = "vision"
BACKEND_TESSERACT = "tesseract"

# 1–2 digit LCD values only. Do not take prefixes of digit soup like "16956" → 16.
# Vision often reads the humidity "%" as a trailing colon ("79:").
_NUM_TOKEN_RE = re.compile(
    r"(?<![A-Za-z\d.])(-?\d{1,2}(?:\.\d{1,2})?)(?!\d)\s*(%|°|:|deg(?:rees?)?|c)?",
    re.I,
)
_SWIFT_CACHE_NAME = "humidifier-ocr-vision"

# VNRecognizeTextRequest helper; compiled once per source hash on macOS.
_SWIFT_SOURCE = r"""
import Foundation
import Vision
import AppKit

if CommandLine.arguments.count < 2 {
    fputs("usage: humidifier-ocr-vision [--fast] <image>\n", stderr)
    exit(2)
}
var fast = false
var path = ""
for arg in CommandLine.arguments.dropFirst() {
    if arg == "--fast" { fast = true; continue }
    path = arg
}
if path.isEmpty {
    fputs("missing image path\n", stderr)
    exit(2)
}
let url = URL(fileURLWithPath: path)
guard let nsImage = NSImage(contentsOf: url) else {
    fputs("load failed\n", stderr)
    exit(1)
}
var rect = NSRect(origin: .zero, size: nsImage.size)
guard let cgImage = nsImage.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
    fputs("cgimage failed\n", stderr)
    exit(1)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = fast ? .fast : .accurate
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("vision failed: \(error)\n", stderr)
    exit(1)
}
struct Row: Codable {
    let text: String
    let confidence: Float
    let x: CGFloat
    let y: CGFloat
    let w: CGFloat
    let h: CGFloat
}
var rows: [Row] = []
if let results = request.results {
    for obs in results {
        let candidate = obs.topCandidates(1).first
        let text = candidate?.string ?? ""
        let conf = candidate?.confidence ?? obs.confidence
        let b = obs.boundingBox
        rows.append(Row(text: text, confidence: conf, x: b.origin.x, y: b.origin.y, w: b.size.width, h: b.size.height))
    }
}
let encoder = JSONEncoder()
FileHandle.standardOutput.write(try encoder.encode(rows))
"""


def normalize_read_mode(raw, default=READ_MODE_OCR_FIRST):
    text = (raw or default or READ_MODE_OCR_FIRST).strip().lower().replace("-", "_")
    if text in READ_MODES:
        return text
    return default if default in READ_MODES else READ_MODE_OCR_FIRST


def _which(name):
    if not name:
        return None
    if os.path.sep in name and os.path.isfile(name) and os.access(name, os.X_OK):
        return name
    found = shutil.which(name)
    if found:
        return found
    for prefix in ("/opt/homebrew/bin", "/usr/local/bin"):
        candidate = os.path.join(prefix, os.path.basename(name))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _pyobjc_vision_available():
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        return hasattr(Vision, "VNRecognizeTextRequest")
    except Exception:
        return False


def _swift_available():
    return sys.platform == "darwin" and bool(_which("swiftc"))


def detect_ocr_backend(prefer="auto", tesseract_bin="tesseract"):
    """Return (backend, detail) or (None, {}). prefer: auto|vision|tesseract."""
    prefer = (prefer or "auto").strip().lower()
    vision_how = None
    if _pyobjc_vision_available():
        vision_how = "pyobjc"
    elif _swift_available():
        vision_how = "swift"

    tess = _which(tesseract_bin) or _which("tesseract")

    if prefer == BACKEND_VISION:
        if vision_how:
            return BACKEND_VISION, {"how": vision_how, "tesseract": tess}
        return None, {"tried": "vision"}
    if prefer == BACKEND_TESSERACT:
        if tess:
            return BACKEND_TESSERACT, {"bin": tess}
        return None, {"tried": "tesseract"}

    if vision_how:
        return BACKEND_VISION, {"how": vision_how, "tesseract": tess}
    if tess:
        return BACKEND_TESSERACT, {"bin": tess}
    return None, {}


def backend_label(backend, detail=None):
    detail = detail or {}
    if backend == BACKEND_VISION:
        how = detail.get("how") or "macos"
        return f"vision/{how}"
    if backend == BACKEND_TESSERACT:
        return f"tesseract:{detail.get('bin') or 'tesseract'}"
    return "none"


def _swift_binary_path():
    digest = hashlib.sha256(_SWIFT_SOURCE.encode()).hexdigest()[:12]
    return os.path.join(tempfile.gettempdir(), f"{_SWIFT_CACHE_NAME}-{digest}")


def _ensure_swift_helper(timeout=60):
    dest = _swift_binary_path()
    if os.path.isfile(dest) and os.access(dest, os.X_OK):
        return dest
    swiftc = _which("swiftc")
    if not swiftc:
        raise FileNotFoundError("swiftc not found")
    src_path = dest + ".swift"
    with open(src_path, "w", encoding="utf-8") as f:
        f.write(_SWIFT_SOURCE)
    try:
        subprocess.run(
            [swiftc, "-O", "-o", dest, src_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass
    return dest


def _run_captured(argv, timeout):
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ocr_pyobjc(image_path, vision_level="accurate"):
    import Vision
    from Foundation import NSURL

    url = NSURL.fileURLWithPath_(os.path.abspath(image_path))
    request = Vision.VNRecognizeTextRequest.alloc().init()
    level = (vision_level or "accurate").strip().lower()
    accurate = getattr(Vision, "VNRequestTextRecognitionLevelAccurate", 0)
    fast = getattr(Vision, "VNRequestTextRecognitionLevelFast", 1)
    request.setRecognitionLevel_(fast if level == "fast" else accurate)
    if hasattr(request, "setUsesLanguageCorrection_"):
        request.setUsesLanguageCorrection_(False)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    ok, err = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"VNRecognizeTextRequest failed: {err}")
    observations = []
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        cand = candidates[0] if candidates else None
        text = cand.string() if cand is not None else ""
        conf = float(cand.confidence()) if cand is not None else float(obs.confidence())
        box = obs.boundingBox()
        observations.append(
            {
                "text": text,
                "confidence": conf,
                "x": float(box.origin.x),
                "y": float(box.origin.y),
                "w": float(box.size.width),
                "h": float(box.size.height),
                "origin": "vision",
            }
        )
    return observations


def _ocr_swift(image_path, timeout, vision_level="accurate"):
    helper = _ensure_swift_helper(timeout=max(30, int(timeout)))
    argv = [helper]
    if (vision_level or "").strip().lower() == "fast":
        argv.append("--fast")
    argv.append(os.path.abspath(image_path))
    result = _run_captured(argv, timeout)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[:300]
        raise RuntimeError(f"vision helper rc={result.returncode}: {err or 'no output'}")
    rows = json.loads(result.stdout or "[]")
    observations = []
    for row in rows:
        observations.append(
            {
                "text": row.get("text") or "",
                "confidence": float(row.get("confidence") or 0),
                "x": float(row.get("x") or 0),
                "y": float(row.get("y") or 0),
                "w": float(row.get("w") or 0),
                "h": float(row.get("h") or 0),
                "origin": "vision",
            }
        )
    return observations


def _ocr_tesseract(
    image_path,
    *,
    bin_path,
    timeout,
    lang,
    psm,
    whitelist,
):
    argv = [bin_path, os.path.abspath(image_path), "stdout", "tsv", "-l", lang, "--psm", str(psm)]
    if whitelist:
        argv.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
    result = _run_captured(argv, timeout)
    if result.returncode != 0 and not (result.stdout or "").strip():
        err = (result.stderr or "").strip()[:300]
        raise RuntimeError(f"tesseract rc={result.returncode}: {err or 'no output'}")
    return _parse_tesseract_tsv(result.stdout or "")


def _parse_tesseract_tsv(tsv_text):
    observations = []
    lines = [ln for ln in (tsv_text or "").splitlines() if ln.strip()]
    if not lines:
        return observations
    header = lines[0].split("\t")
    try:
        idx_conf = header.index("conf")
        idx_text = header.index("text")
        idx_left = header.index("left")
        idx_top = header.index("top")
        idx_w = header.index("width")
        idx_h = header.index("height")
        idx_level = header.index("level") if "level" in header else None
    except ValueError:
        return observations
    max_bottom = 1.0
    words = []
    for line in lines[1:]:
        cols = line.split("\t")
        if len(cols) <= max(idx_conf, idx_text, idx_left, idx_top, idx_w, idx_h):
            continue
        if idx_level is not None:
            try:
                if int(cols[idx_level]) != 5:
                    continue
            except ValueError:
                continue
        raw_text = cols[idx_text].strip()
        if not raw_text:
            continue
        try:
            conf = float(cols[idx_conf])
            left = float(cols[idx_left])
            top = float(cols[idx_top])
            w = float(cols[idx_w])
            h = float(cols[idx_h])
        except ValueError:
            continue
        max_bottom = max(max_bottom, top + h)
        words.append((raw_text, conf, left, top, w, h))
    page_h = max_bottom if max_bottom > 1 else 1.0
    page_w = 1.0
    for raw_text, conf, left, top, w, h in words:
        page_w = max(page_w, left + w)
    for raw_text, conf, left, top, w, h in words:
        # Tesseract origin is top-left; store Vision-like bottom-left y for sorting.
        y_bl = 1.0 - ((top + h) / page_h) if page_h else 0.0
        observations.append(
            {
                "text": raw_text,
                "confidence": (conf / 100.0) if conf >= 0 else 0.0,
                "x": left / page_w if page_w else 0.0,
                "y": y_bl,
                "w": w / page_w if page_w else 0.0,
                "h": h / page_h if page_h else 0.0,
                "origin": "tesseract",
            }
        )
    return observations


def ocr_image(
    image_path,
    *,
    timeout=8,
    prefer="auto",
    tesseract_bin="tesseract",
    tesseract_lang="eng",
    tesseract_psm="6",
    tesseract_whitelist="0123456789.%C",
    vision_level="accurate",
    backend=None,
    detail=None,
):
    """Run local OCR. Returns dict with text, observations, backend, error."""
    if backend is None:
        backend, detail = detect_ocr_backend(prefer=prefer, tesseract_bin=tesseract_bin)
    detail = detail or {}
    if not backend:
        return {
            "text": "",
            "observations": [],
            "backend": None,
            "error": "no local OCR backend (need macOS Vision or tesseract CLI)",
        }
    try:
        if backend == BACKEND_VISION:
            how = detail.get("how")
            if how == "pyobjc":
                observations = _ocr_pyobjc(image_path, vision_level=vision_level)
            else:
                observations = _ocr_swift(image_path, timeout=timeout, vision_level=vision_level)
        else:
            bin_path = detail.get("bin") or _which(tesseract_bin) or tesseract_bin
            observations = _ocr_tesseract(
                image_path,
                bin_path=bin_path,
                timeout=timeout,
                lang=tesseract_lang,
                psm=tesseract_psm,
                whitelist=tesseract_whitelist,
            )
        text = " ".join(o["text"] for o in observations if o.get("text")).strip()
        return {
            "text": text,
            "observations": observations,
            "backend": backend,
            "detail": detail,
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "text": "",
            "observations": [],
            "backend": backend,
            "detail": detail,
            "error": f"ocr timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "text": "",
            "observations": [],
            "backend": backend,
            "detail": detail,
            "error": f"ocr error: {e}",
        }


def _maybe_float(token):
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _is_intish(value):
    return abs(value - round(value)) < 0.05


def _candidates_from_text(text):
    candidates = []
    for match in _NUM_TOKEN_RE.finditer(text or ""):
        value = _maybe_float(match.group(1))
        if value is None:
            continue
        suffix = (match.group(2) or "").lower()
        has_percent = suffix in ("%", ":")
        has_degree = suffix in ("°", "c") or suffix.startswith("deg")
        candidates.append(
            {
                "value": value,
                "is_int": _is_intish(value),
                "has_percent": has_percent,
                "has_degree": has_degree,
                "confidence": None,
                "y_from_top": None,
                "raw": match.group(0),
            }
        )
    return candidates


def _candidates_from_observations(observations):
    candidates = []
    texts = []
    for obs in observations or []:
        raw = (obs.get("text") or "").strip()
        if not raw:
            continue
        texts.append(raw)
        y = obs.get("y")
        # Vision: y is bottom-left origin (image top ≈ 1). Convert to from-top.
        y_from_top = (1.0 - (float(y) + float(obs.get("h") or 0) / 2.0)) if y is not None else None
        conf = obs.get("confidence")
        for match in _NUM_TOKEN_RE.finditer(raw):
            value = _maybe_float(match.group(1))
            if value is None:
                continue
            suffix = (match.group(2) or "").lower()
            candidates.append(
                {
                    "value": value,
                    "is_int": _is_intish(value),
                    "has_percent": suffix in ("%", ":"),
                    "has_degree": suffix in ("°", "c") or suffix.startswith("deg"),
                    "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                    "y_from_top": y_from_top,
                    "raw": match.group(0),
                }
            )
    # Attach a nearby lone "%" observation to the preceding number.
    joined = " ".join(texts)
    if "%" in joined:
        for cand in candidates:
            if not cand["has_percent"] and cand["is_int"]:
                # If the full OCR string has % and this is the last int, mark later.
                pass
        if not any(c["has_percent"] for c in candidates):
            ints = [c for c in candidates if c["is_int"]]
            if ints:
                # Prefer the lowest-on-screen integer (humidity row).
                pick = max(ints, key=lambda c: c["y_from_top"] if c["y_from_top"] is not None else -1)
                pick["has_percent"] = True
    return candidates or _candidates_from_text(joined)


def _in_rh(value, plausible_min, plausible_max):
    return plausible_min <= value <= plausible_max and _is_intish(value)


def _pick_lower(cands):
    if any(c.get("y_from_top") is not None for c in cands):
        return max(cands, key=lambda c: (c.get("y_from_top") is not None, c.get("y_from_top") or 0))
    return cands[-1]


def _pick_upper(cands):
    if any(c.get("y_from_top") is not None for c in cands):
        return min(cands, key=lambda c: (c.get("y_from_top") is None, c.get("y_from_top") or 0))
    return cands[0]


def _ordered_for_layout(cands):
    """Top-to-bottom (Vision y) or left-to-right document order."""
    if any(c.get("y_from_top") is not None for c in cands):
        return sorted(
            cands,
            key=lambda c: (
                c.get("y_from_top") is None,
                c.get("y_from_top") if c.get("y_from_top") is not None else 0,
            ),
        )
    return list(cands)


def _confidence_of(item, fallback):
    if item and isinstance(item.get("confidence"), (int, float)):
        return float(item["confidence"])
    return fallback


def parse_thermopro_ocr(
    text,
    observations=None,
    *,
    plausible_min=10,
    plausible_max=95,
):
    """Parse remote humidity (and temp if present) from OCR text/boxes.

    Returns (reading_dict_or_None, reason). Does not apply jump/delta checks.
    """
    observations = observations or []
    if observations:
        candidates = _candidates_from_observations(observations)
    else:
        candidates = _candidates_from_text(text)

    if not candidates:
        snippet = (text or "").strip().replace("\n", " ")[:80]
        return None, f"ocr: no numeric tokens{f' in {snippet!r}' if snippet else ''}"

    humidity = None
    temp = None
    method = None

    pct = [c for c in candidates if c["has_percent"] and _in_rh(c["value"], plausible_min, plausible_max)]
    if len(pct) == 1:
        humidity = pct[0]
        method = "percent"
    elif len(pct) > 1:
        values = sorted({round(c["value"]) for c in pct})
        if len(values) > 1 and (max(values) - min(values)) > 2:
            # Two bare % tokens with no layout → ambiguous. OUT-above-IN
            # (four numbers or vertical boxes) → remote/OUT is the first / upper RH.
            if len(candidates) >= 4 or any(c.get("y_from_top") is not None for c in pct):
                humidity = _pick_upper(pct)
                method = "percent"
            else:
                return None, "ocr: multiple % humidity values"
        else:
            humidity = _pick_lower(pct)
            method = "percent"

    others = [c for c in candidates if c is not humidity]
    degree_temps = [c for c in others if c["has_degree"] or not c["is_int"]]
    if degree_temps:
        temp = _pick_upper(degree_temps)

    if humidity is None:
        nums = _ordered_for_layout(candidates)
        rh_ints = [c for c in nums if _in_rh(c["value"], plausible_min, plausible_max)]
        if len(nums) >= 4:
            out_temp, out_rh = nums[0], nums[1]
            if _in_rh(out_rh["value"], plausible_min, plausible_max):
                humidity = out_rh
                if temp is None and out_temp is not humidity:
                    temp = out_temp
                method = "layout"
            elif len(rh_ints) == 1:
                humidity = rh_ints[0]
                method = "single_rh"
                rest = [c for c in nums if c is not humidity]
                if rest and temp is None:
                    temp = _pick_upper(rest)
        elif len(nums) >= 2:
            upper = _pick_upper(nums)
            lower = _pick_lower(nums)
            if lower is not upper and _in_rh(lower["value"], plausible_min, plausible_max):
                humidity = lower
                if temp is None and upper is not humidity:
                    temp = upper
                method = "layout"
            elif len(rh_ints) == 1:
                humidity = rh_ints[0]
                method = "single_rh"
                rest = [c for c in nums if c is not humidity]
                if rest and temp is None:
                    temp = _pick_upper(rest)
        elif len(nums) == 1:
            only = nums[0]
            if _in_rh(only["value"], plausible_min, plausible_max) and (
                only["value"] >= 30 or only["has_percent"]
            ):
                humidity = only
                method = "single_rh"
            else:
                return None, f"ocr: single number {only['value']} looks like temperature, not humidity"

    if humidity is None:
        return None, "ocr: could not identify humidity"

    conf_fallback = {"percent": 0.75, "layout": 0.62, "single_rh": 0.55}.get(method, 0.5)
    confidence = _confidence_of(humidity, conf_fallback)
    if method == "single_rh" and not humidity.get("has_percent"):
        confidence = min(confidence, 0.55)

    rh = int(round(humidity["value"]))
    reading = {
        "remote_humidity": rh,
        "remote_temp": None,
        "local_humidity": None,
        "local_temp": None,
        "ocr_confidence": round(confidence, 3),
        "ocr_method": method,
    }
    if temp is not None and temp is not humidity:
        tv = temp["value"]
        if -20 <= tv <= 60:
            reading["remote_temp"] = int(tv) if _is_intish(tv) else round(tv, 1)
    return reading, "ok"


def accept_ocr_reading(
    reading,
    *,
    min_confidence,
    prev_humidity,
    plausible_min,
    plausible_max,
    max_jump,
    remote_local_max_delta,
    plausibility_fn,
):
    """Confidence + shared plausibility. Returns (ok, reason)."""
    if not reading:
        return False, "ocr: empty reading"
    conf = reading.get("ocr_confidence")
    if isinstance(conf, (int, float)) and conf < min_confidence:
        return False, f"ocr: confidence {conf:.2f} < {min_confidence}"
    return plausibility_fn(
        reading,
        prev_humidity,
        plausible_min=plausible_min,
        plausible_max=plausible_max,
        max_jump=max_jump,
        remote_local_max_delta=remote_local_max_delta,
    )
