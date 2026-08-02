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
import image_ocr
import diary


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
    """Blur + darken + downscale + noise — a 'bad phone photo'."""
    out = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
    out = cv2.GaussianBlur(out, (7, 7), 0)
    out = cv2.convertScaleAbs(out, alpha=0.55, beta=-30)
    noise = np.random.default_rng(42).normal(0, 12, out.shape).astype(np.int16)
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
        assert names == ["original", "enhanced", "binary"]
        for v in variants[1:]:
            img = cv2.imread(str(v.path), cv2.IMREAD_GRAYSCALE)
            assert img is not None and img.size > 0

    def test_enhanced_improves_contrast(self, tmp_path):
        src = tmp_path / "photo.jpg"
        degraded = degrade(render_lab_image())
        cv2.imwrite(str(src), degraded)
        info = image_ocr.assess_quality(cv2.imread(str(src)))
        variants = image_ocr.build_variants(src, tmp_path, info)
        enhanced = cv2.imread(str(variants[1].path), cv2.IMREAD_GRAYSCALE)
        orig_gray = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        # CLAHE + unsharp should increase local contrast (std)
        assert enhanced.std() > orig_gray.std() * 1.1

    def test_binary_is_dark_text_on_light_bg(self, tmp_path):
        src = tmp_path / "photo.jpg"
        cv2.imwrite(str(src), degrade(render_lab_image()))
        info = image_ocr.assess_quality(cv2.imread(str(src)))
        variants = image_ocr.build_variants(src, tmp_path, info)
        binary = cv2.imread(str(variants[2].path), cv2.IMREAD_GRAYSCALE)
        assert binary.mean() > 127  # mostly light background

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

    def test_question_marks_penalized(self):
        clean = "\n".join(LAB_LINES)
        noisy = clean.replace("5", "?")
        assert image_ocr.ocr_quality_score(noisy) < image_ocr.ocr_quality_score(clean)

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
