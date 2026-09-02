"""Offline tests: image preprocessing, OCR scoring, diary parsing.
Run: .venv/bin/python -m pytest tests/test_offline.py -v
No ClickHouse / claude CLI / Telegram required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from healthbot import diary
from healthbot import ocr as image_ocr


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: synthesize a lab-report image and degrade it
# ─────────────────────────────────────────────────────────────────────────────
LAB_LINES = [
    "Общий анализ крови",
    "Гемоглобин      145   г/л      130-160",
    "Эритроциты      4.52  млн/мкл  4.0-5.1",
    "Лейкоциты       6.1   тыс/мкл  4.0-9.0",
    "Тромбоциты      248   тыс/мкл  150-400",
    "Глюкоза         5.4   ммоль/л  3.9-6.1",
    "Креатинин       88    мкмоль/л 62-106",
    "АЛТ             25    Ед/л     0-41",
    "АСТ             22    Ед/л     0-40",
    "Ферритин        95    мкг/л    30-400",
]


def render_lab_image() -> np.ndarray:
    img = np.full((900, 1400, 3), 255, np.uint8)
    y = 70
    for line in LAB_LINES:
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 80
    return img


def degrade(img: np.ndarray) -> np.ndarray:
    """Extreme degradation for quality-assessment tests (beyond rescue)."""
    out = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (7, 7), 0)
    out = cv2.convertScaleAbs(out, alpha=0.55, beta=-30)
    noise = np.random.default_rng(42).normal(0, 12, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def degrade_realistic(img: np.ndarray) -> np.ndarray:
    """Realistic bad phone shot: half resolution, moderate blur, dim light, noise."""
    out = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (5, 5), 0)
    out = cv2.convertScaleAbs(out, alpha=0.62, beta=-15)
    noise = np.random.default_rng(7).normal(0, 8, out.shape).astype(np.int16)
    return np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Quality assessment
# ─────────────────────────────────────────────────────────────────────────────
class TestQualityAssessment:
    def test_clean_image_has_no_issues(self):
        info = image_ocr.assess_quality(render_lab_image())
        assert "размытое" not in info.issues
        assert "тёмное" not in info.issues
        assert "пересвеченное" not in info.issues

    def test_degraded_image_flagged(self):
        info = image_ocr.assess_quality(degrade(render_lab_image()))
        assert "тёмное" in info.issues
        assert "низкое разрешение" in info.issues
        assert info.poor

    def test_blurry_bright_image_flagged(self):
        img = cv2.GaussianBlur(render_lab_image(), (11, 11), 0)
        info = image_ocr.assess_quality(img)
        assert "размытое" in info.issues

    def test_noise_not_confused_with_blur(self):
        img = render_lab_image()
        noise = np.random.default_rng(42).normal(0, 25, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        info = image_ocr.assess_quality(noisy)
        assert "размытое" not in info.issues

    def test_overexposed_flagged(self):
        img = np.full((800, 1200, 3), 245, np.uint8)
        info = image_ocr.assess_quality(img)
        assert "пересвеченное" in info.issues


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing variants
# ─────────────────────────────────────────────────────────────────────────────
class TestVariants:
    def test_variants_created_and_readable(self, tmp_path):
        src = tmp_path / "photo.jpg"
        cv2.imwrite(str(src), degrade(render_lab_image()))
        info = image_ocr.assess_quality(cv2.imread(str(src)))
        variants = image_ocr.build_variants(src, tmp_path, info)
        names = [v.name for v in variants]
        assert names[0] == "prepared"
        assert "prepared_180" in names
        assert "enhanced" in names
        assert "binary" in names
        for v in variants:
            img = cv2.imread(str(v.path), cv2.IMREAD_UNCHANGED)
            assert img is not None and img.size > 0

    def test_enhanced_improves_contrast(self, tmp_path):
        src = tmp_path / "photo.jpg"
        degraded = degrade(render_lab_image())
        cv2.imwrite(str(src), degraded)
        info = image_ocr.assess_quality(cv2.imread(str(src)))
        variants = image_ocr.build_variants(src, tmp_path, info)
        prepared = cv2.imread(str(variants[0].path))
        orig_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        prep_gray = cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY)
        assert prep_gray.std() > orig_gray.std() * 1.05

    def test_binary_is_dark_text_on_light_bg(self, tmp_path):
        src = tmp_path / "photo.jpg"
        cv2.imwrite(str(src), degrade(render_lab_image()))
        info = image_ocr.assess_quality(cv2.imread(str(src)))
        variants = image_ocr.build_variants(src, tmp_path, info)
        binary = next(v for v in variants if v.name == "binary")
        img = cv2.imread(str(binary.path), cv2.IMREAD_GRAYSCALE)
        assert img.mean() > 127  # mostly light background

    def test_deskew_corrects_rotation(self):
        img = cv2.cvtColor(render_lab_image(), cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w // 2, h // 2), 6, 1.0)
        rotated = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        fixed = image_ocr._deskew(rotated)
        coords_orig = np.column_stack(np.where(
            cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0))
        coords_fixed = np.column_stack(np.where(
            cv2.threshold(fixed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0))
        angle_orig = cv2.minAreaRect(coords_orig)[-1]
        angle_fixed = cv2.minAreaRect(coords_fixed)[-1]
        norm = lambda a: -(90 + a) if a < -45 else -a
        assert abs(norm(angle_fixed)) < abs(norm(angle_orig))


class TestOrientationAndPrepare:
    def test_upright_keeps_no_rotation(self):
        assert image_ocr.choose_rotation(render_lab_image()) is None

    def test_90_cw_is_corrected(self):
        lab = render_lab_image()
        sideways = cv2.rotate(lab, cv2.ROTATE_90_CLOCKWISE)
        code = image_ocr.choose_rotation(sideways)
        assert code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fixed = cv2.rotate(sideways, code)
        assert fixed.shape[1] > fixed.shape[0]
        assert image_ocr._horizontal_line_var(
            cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
        ) > image_ocr._horizontal_line_var(
            cv2.cvtColor(sideways, cv2.COLOR_BGR2GRAY)
        )

    def test_90_ccw_is_corrected(self):
        lab = render_lab_image()
        sideways = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
        code = image_ocr.choose_rotation(sideways)
        assert code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        fixed = cv2.rotate(sideways, code)
        assert fixed.shape[1] > fixed.shape[0]

    def test_prepare_upscales_tiny_photo(self):
        tiny = cv2.resize(render_lab_image(), (320, 206), interpolation=cv2.INTER_AREA)
        out = image_ocr.prepare_for_ocr(tiny)
        assert max(out.shape[:2]) >= 1990

    def test_prepare_fixes_sideways_tiny_photo(self):
        tiny = cv2.resize(render_lab_image(), (400, 257), interpolation=cv2.INTER_AREA)
        sideways = cv2.rotate(tiny, cv2.ROTATE_90_CLOCKWISE)
        out = image_ocr.prepare_for_ocr(sideways)
        assert out.shape[1] > out.shape[0]
        assert max(out.shape[:2]) >= 1990

    def test_exif_orientation_applied(self, tmp_path):
        from PIL import Image
        rgb = np.zeros((40, 80, 3), np.uint8)
        rgb[:, :40] = (255, 0, 0)
        im = Image.fromarray(rgb)
        exif = im.getexif()
        exif[274] = 6  # rotate 90 CW for display → 80x40
        path = tmp_path / "exif6.jpg"
        im.save(path, format="JPEG", quality=95, exif=exif)
        loaded = image_ocr.load_image_bgr(path)
        assert loaded.shape[0] == 80 and loaded.shape[1] == 40


# ─────────────────────────────────────────────────────────────────────────────
# OCR output scoring
# ─────────────────────────────────────────────────────────────────────────────
class TestOcrScoring:
    def test_good_lab_text_scores_high(self):
        text = "\n".join(LAB_LINES)
        assert image_ocr.ocr_quality_score(text) >= 0.55

    def test_empty_and_short_score_zero(self):
        assert image_ocr.ocr_quality_score("") == 0.0
        assert image_ocr.ocr_quality_score("abc") == 0.0

    def test_refusal_scores_low(self):
        assert image_ocr.ocr_quality_score(
            "К сожалению, не удалось распознать текст на этом изображении. " * 3) <= 0.10

    def test_transliterated_russian_penalized(self):
        ru = "\n".join(LAB_LINES)
        latin = (
            "Gemoglobin 145 g/l 130-160\nEritrotsity 4.52\nLeykotsity 6.1\n"
            "Trombotsity 248\nGlyukoza 5.4 mmol/l\nKreatinin 88 umol/l\n"
            "ALT 25 U/l AST 22 Ferritin 95\nBilirubin cholesterol glucose"
        )
        assert image_ocr.ocr_quality_score(latin) < image_ocr.ocr_quality_score(ru)

    def test_question_marks_penalized(self):
        clean = "\n".join(LAB_LINES)
        noisy = clean.replace("5", "?")
        assert image_ocr.ocr_quality_score(noisy) < image_ocr.ocr_quality_score(clean)
        clean = "\n".join(LAB_LINES)
        noisy = clean.replace("5", "?")
        assert image_ocr.ocr_quality_score(noisy) < image_ocr.ocr_quality_score(clean)

    def test_strips_leaked_think_tags(self):
        assert image_ocr._strip_model_noise("</think>Гемоглобин 145") == "Гемоглобин 145"
        assert image_ocr._strip_model_noise("<think>reason</think>\nГемоглобин 145") == "Гемоглобин 145"

    def test_result_ok_thresholds(self):
        good = image_ocr.OCRResult(
            text="\n".join(LAB_LINES), variant="enhanced", score=0.6,
            attempts=2, quality=image_ocr.QualityInfo())
        bad = image_ocr.OCRResult(
            text="???", variant="original", score=0.1,
            attempts=3, quality=image_ocr.QualityInfo())
        assert good.ok
        assert not bad.ok


# ─────────────────────────────────────────────────────────────────────────────
# Vision API client (mocked OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────
class TestVisionApiClient:
    def _run_mock_server(self, captured: dict):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json as _json
        import threading

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                captured["request"] = _json.loads(body)
                captured["path"] = self.path
                payload = _json.dumps({
                    "choices": [{"message": {"content": "Гемоглобин 145 г/л 130-160"}}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    def test_request_shape_and_parsing(self, tmp_path, monkeypatch):
        captured: dict = {}
        server = self._run_mock_server(captured)
        try:
            monkeypatch.setattr(image_ocr, "VISION_API_BASE",
                                f"http://127.0.0.1:{server.server_port}")
            monkeypatch.setattr(image_ocr, "VISION_API_KEY", "test-key")
            src = tmp_path / "photo.jpg"
            cv2.imwrite(str(src), render_lab_image())

            text = image_ocr._vision_api_read(src, "gemini-2.5-flash-lite")

            assert text == "Гемоглобин 145 г/л 130-160"
            req = captured["request"]
            assert captured["path"] == "/chat/completions"
            assert req["model"] == "gemini-2.5-flash-lite"
            assert req["temperature"] == 0.0
            parts = req["messages"][0]["content"]
            assert parts[0]["type"] == "text" and "медицинского" in parts[0]["text"]
            assert parts[1]["type"] == "image_url"
            assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            server.shutdown()

    def test_backend_auto_prefers_vision_api(self, monkeypatch):
        monkeypatch.delenv("OCR_BACKEND", raising=False)
        monkeypatch.setattr(image_ocr, "VISION_API_KEY", "k")
        assert image_ocr.ocr_backend() == "vision_api"
        # Local Tesseract is permanently rejected
        monkeypatch.setattr(image_ocr, "VISION_API_KEY", "")
        monkeypatch.setenv("OCR_BACKEND", "tesseract")
        with pytest.raises(RuntimeError, match="локальный OCR запрещён"):
            image_ocr.ocr_backend()
        monkeypatch.setenv("OCR_TESSERACT", "1")
        with pytest.raises(RuntimeError, match="локальный OCR запрещён"):
            image_ocr.ocr_backend()

    def test_error_body_raises(self, tmp_path, monkeypatch):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(429)
                self.end_headers()
                self.wfile.write(b"rate limited")

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            monkeypatch.setattr(image_ocr, "VISION_API_BASE",
                                f"http://127.0.0.1:{server.server_port}")
            monkeypatch.setattr(image_ocr, "VISION_API_KEY", "k")
            src = tmp_path / "photo.jpg"
            cv2.imwrite(str(src), render_lab_image())
            with pytest.raises(RuntimeError, match="429"):
                image_ocr._vision_api_read(src)
        finally:
            server.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Diary parsing
# ─────────────────────────────────────────────────────────────────────────────
class TestDiaryParsing:
    def test_hypothesis_entry(self):
        msg = diary.detect_diary_message("гипотеза: ферритин падает из-за донорства")
        assert msg["kind"] == "entry"
        assert msg["entry"]["entry_type"] == "hypothesis"
        assert msg["entry"]["text"] == "ферритин падает из-за донорства"
        assert msg["entry"]["status"] == "active"

    def test_wellbeing_full_parse(self):
        msg = diary.detect_diary_message(
            "самочувствие 7/10 сон 6.5ч энергия 6 настроение 7 симптомы: лёгкая головная боль")
        e = msg["entry"]
        assert e["entry_type"] == "wellbeing"
        assert e["wellbeing_score"] == 7
        assert e["sleep_hours"] == 6.5
        assert e["energy_score"] == 6
        assert e["mood_score"] == 7
        assert "головная боль" in e["symptoms"]

    def test_symptom_entry(self):
        msg = diary.detect_diary_message("симптом: тянет правый бок после еды")
        assert msg["entry"]["entry_type"] == "symptom"
        assert "тянет правый бок" in msg["entry"]["symptoms"]

    def test_note_entry(self):
        msg = diary.detect_diary_message("дневник: начал принимать магний вечером")
        assert msg["entry"]["entry_type"] == "note"

    def test_hypothesis_status_confirmed(self):
        msg = diary.detect_diary_message("гипотеза подтвердилась: донорство")
        assert msg["kind"] == "status"
        assert msg["status"] == "confirmed"
        assert msg["match"] == "донорство"

    def test_hypothesis_status_rejected(self):
        msg = diary.detect_diary_message("гипотеза опровергнута")
        assert msg["kind"] == "status"
        assert msg["status"] == "rejected"
        assert msg["match"] == ""

    def test_regular_question_not_diary(self):
        assert diary.detect_diary_message("что у меня с ферритином?") is None
        assert diary.detect_diary_message("/summary") is None

    def test_out_of_range_scores_ignored(self):
        msg = diary.detect_diary_message("самочувствие 25/10")
        assert msg["entry"]["wellbeing_score"] is None


class TestDiaryCommands:
    def test_parse_diary_command(self):
        assert diary.parse_diary_command("") == {"limit": 10, "days": None}
        assert diary.parse_diary_command("25") == {"limit": 25, "days": None}

    def test_format_empty_diary(self):
        text = diary.format_diary_list([])
        assert "Дневник пуст" in text

    def test_format_empty_hypotheses(self):
        assert "Активных гипотез нет" in diary.format_hypotheses([])
