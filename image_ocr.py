"""Robust OCR for low-quality photos of medical documents.

Pipeline: quality assessment → OpenCV preprocessing variants (upscale,
denoise, CLAHE contrast, unsharp mask, deskew, adaptive binarization) →
multi-pass Claude Vision with per-candidate quality scoring → optional
Tesseract cross-check. The best-scoring transcription wins.

Env knobs:
    OCR_BACKEND        auto|vision|tesseract (default auto: vision when the
                       Claude CLI is the LLM provider, tesseract otherwise —
                       DeepSeek API is text-only, so photos go through Tesseract)
    OCR_VISION_MODEL   vision model for attempts (default claude-sonnet-4-6)
    OCR_FINAL_MODEL    stronger model for the last attempt (default = OCR_VISION_MODEL)
    OCR_MAX_ATTEMPTS   max vision attempts across variants (default 3)
    OCR_TIMEOUT        per-attempt timeout seconds (default 180)
    OCR_TESSERACT      1/0 force tesseract cross-check, default auto (on if rus data present)
    OCR_CLEANUP        1/0 LLM post-correction of OCR text (default 1)
"""
from __future__ import annotations

import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("health-bot")

VISION_MODEL = os.getenv("OCR_VISION_MODEL", "claude-sonnet-4-6")
FINAL_MODEL = os.getenv("OCR_FINAL_MODEL", VISION_MODEL)
MAX_ATTEMPTS = int(os.getenv("OCR_MAX_ATTEMPTS", "3"))
OCR_TIMEOUT = int(os.getenv("OCR_TIMEOUT", "180"))

# Quality assessment thresholds (calibrated on synthetic lab-report images)
_BLUR_THRESHOLD = 40.0        # Laplacian var (after 3x3 denoise) below this = blurry
_DARK_THRESHOLD = 115.0       # document photos should be bright (white paper)
_BRIGHT_MEAN = 235.0          # overexposed only if bright AND washed-out ...
_BRIGHT_MAX_STD = 15.0        # ... i.e. contrast nearly gone
_MIN_LONG_SIDE = 1600         # upscale if long side is smaller
_MAX_LONG_SIDE = 3000         # downscale huge photos for OCR speed

_OCR_PROMPT = """Это фото медицинского документа (анализы, заключение, назначение).
Фото может быть низкого качества: размытое, тёмное, под углом, с бликами.

ЗАДАЧА: извлеки весь видимый текст максимально точно.

ПРАВИЛА:
1. Переписывай текст КАК ЕСТЬ, построчно, сохраняя структуру таблицы: показатель — значение — единицы — норма
2. НЕ ПРИДУМЫВАЙ значения. Если символ или число не читается — ставь знак ?
3. Если строка частично читается — перепиши читаемое, нечитаемое замени на ?
4. Числа критичны: перепроверь каждое. Десятичный разделитель — как в документе (точка или запятая)
5. Никаких комментариев, интерпретаций и пояснений — только извлечённый текст
6. Если это не медицинский документ — всё равно перепиши видимый текст"""

_UNITS_RE = re.compile(
    r"(?:г/л|г/дл|мг/л|мг/дл|ммоль/л|мкмоль/л|нмоль/л|пмоль/л|мкг/л|нг/мл|пг/мл|"
    r"мЕд/мл|Ед/л|Ед/мл|МЕ/мл|МЕ/л|мМЕ/л|мкМЕ/мл|тыс/мкл|млн/мкл|фл|пг|%|"
    r"mg/dl|mmol/l|umol/l|nmol/l|ng/ml|pg/ml|u/l|iu/ml|miu/l|g/l|мм/ч|сек)",
    re.IGNORECASE,
)
_KEYWORDS_RE = re.compile(
    r"(?:гемоглобин|эритроцит|лейкоцит|тромбоцит|глюкоз|холестерин|билирубин|"
    r"креатинин|мочевин|АЛТ|АСТ|ГГТ|ферритин|железо|ТТГ|Т3|Т4|тестостерон|"
    r"кортизол|инсулин|витамин|СОЭ|СРБ|фибриноген|протромбин|антител|"
    r"результат|референс|норма|показатель|исследование|анализ|пациент|"
    r"hemoglobin|glucose|cholesterol|creatinin|bilirubin|TSH)",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"(?:не\s+(?:удалось|могу)\s+(?:прочитать|распознать|разобрать)|"
    r"невозможно\s+(?:прочитать|распознать)|текст\s+нечитаем|"
    r"cannot\s+(?:read|recognize)|unable\s+to\s+read)",
    re.IGNORECASE,
)

_CLEANUP_PROMPT = """Ниже — результат OCR медицинского документа. В нём есть ошибки распознавания.

ЗАДАЧА: исправь очевидные OCR-ошибки и верни чистый текст.

ПРАВИЛА:
1. Исправляй типичные OCR-подмены: О↔0, л↔1, З↔3, В↔8, |↔1, разорванные слова
2. Восстанови структуру строк: показатель — значение — единицы — норма
3. НЕ ПРИДУМЫВАЙ значения и не угадывай числа: если число явно повреждено и не восстанавливается из контекста — оставь ?
4. Сохрани все показатели, ничего не удаляй
5. Верни ТОЛЬКО исправленный текст, без комментариев

=== OCR-ТЕКСТ ===
"""


def ocr_backend() -> str:
    """Which OCR engine to use: vision LLM or local Tesseract."""
    forced = os.getenv("OCR_BACKEND", "auto").strip().lower()
    if forced in ("vision", "tesseract"):
        return forced
    try:
        import llm
        if llm.provider() == "claude" and shutil.which("claude"):
            return "vision"
    except Exception:
        pass
    return "tesseract" if _tesseract_available() else "vision"


@dataclass
class QualityInfo:
    blur_var: float = 0.0
    brightness: float = 0.0
    long_side: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def poor(self) -> bool:
        return bool(self.issues)


@dataclass
class Variant:
    name: str
    path: Path


@dataclass
class OCRResult:
    text: str
    variant: str
    score: float
    attempts: int
    quality: QualityInfo
    scores: dict[str, float] = field(default_factory=dict)
    backend: str = "vision"

    @property
    def ok(self) -> bool:
        return self.score >= 0.30 and len(self.text.strip()) >= 20

    @property
    def user_note(self) -> str:
        """Short human-readable note about what it took to read the photo."""
        parts = []
        if self.quality.poor:
            parts.append("📸 Фото низкого качества ({})".format(", ".join(self.quality.issues)))
        if self.variant != "original":
            parts.append(f"применил улучшение изображения ({self.variant})")
        if self.attempts > 1:
            parts.append(f"попыток распознавания: {self.attempts}")
        if self.backend == "tesseract":
            parts.append("⚠️ распознано локальным OCR без визуальной проверки — "
                         "цифры могли исказиться, сверь ключевые значения с оригиналом")
        elif self.score < 0.55:
            parts.append("⚠️ часть текста могла быть нечитаема — проверь значения")
        return " — ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Quality assessment
# ─────────────────────────────────────────────────────────────────────────────
def assess_quality(img: np.ndarray) -> QualityInfo:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    brightness = float(gray.mean())
    # Sensor noise inflates raw Laplacian variance and masks blur; measure
    # after a light denoise. In dark images edge contrast is intrinsically
    # weak, so the blur check is unreliable there — skip it.
    blur_var = 0.0
    if brightness >= _DARK_THRESHOLD:
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        blur_var = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    info = QualityInfo(
        blur_var=blur_var,
        brightness=brightness,
        long_side=max(h, w),
    )
    if brightness >= _DARK_THRESHOLD and info.blur_var < _BLUR_THRESHOLD:
        info.issues.append("размытое")
    if brightness < _DARK_THRESHOLD:
        info.issues.append("тёмное")
    elif brightness > _BRIGHT_MEAN and float(gray.std()) < _BRIGHT_MAX_STD:
        info.issues.append("пересвеченное")
    if info.long_side < _MIN_LONG_SIDE:
        info.issues.append("низкое разрешение")
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def _resize_to(img: np.ndarray, long_side: int, interp: int) -> np.ndarray:
    h, w = img.shape[:2]
    cur = max(h, w)
    if cur == long_side:
        return img
    scale = long_side / cur
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=interp)


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Rotate to correct small skew (0.5°–15°); skip if angle implausible."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 200:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _enhanced(gray: np.ndarray) -> np.ndarray:
    """Denoise → CLAHE contrast → unsharp mask → deskew."""
    out = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    out = clahe.apply(out)
    blur = cv2.GaussianBlur(out, (0, 0), 3)
    out = cv2.addWeighted(out, 1.6, blur, -0.6, 0)
    return _deskew(out)


def _binarized(enhanced_gray: np.ndarray) -> np.ndarray:
    out = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=35, C=11,
    )
    # Text should be dark on light background
    if out.mean() < 127:
        out = cv2.bitwise_not(out)
    return out


def build_variants(image_path: Path, work_dir: Path, quality: QualityInfo) -> list[Variant]:
    """Produce OCR candidates. Original always first; enhanced variants added
    when quality is poor (or always, cheap insurance — attempts are capped)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")

    variants = [Variant("original", image_path)]

    # Normalize resolution for preprocessing variants
    if max(img.shape[:2]) > _MAX_LONG_SIDE:
        img = _resize_to(img, _MAX_LONG_SIDE, cv2.INTER_AREA)
    if max(img.shape[:2]) < _MIN_LONG_SIDE:
        img = _resize_to(img, _MIN_LONG_SIDE, cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enh = _enhanced(gray)
    enh_path = work_dir / f"{image_path.stem}_enhanced.png"
    cv2.imwrite(str(enh_path), enh)
    variants.append(Variant("enhanced", enh_path))

    bin_img = _binarized(enh)
    bin_path = work_dir / f"{image_path.stem}_binary.png"
    cv2.imwrite(str(bin_path), bin_img)
    variants.append(Variant("binary", bin_path))

    return variants


# ─────────────────────────────────────────────────────────────────────────────
# OCR output scoring — picks the most plausible transcription
# ─────────────────────────────────────────────────────────────────────────────
def ocr_quality_score(text: str) -> float:
    """Heuristic 0..1 score: rewards length, digits, lab units/keywords;
    penalizes refusals and high '?' density."""
    t = (text or "").strip()
    if len(t) < 20:
        return 0.0
    if _REFUSAL_RE.search(t[:400]):
        return min(0.10, 0.10 * (len(t) / 200))

    digits = sum(c.isdigit() for c in t)
    units = len(_UNITS_RE.findall(t))
    keywords = len(_KEYWORDS_RE.findall(t))
    q_density = t.count("?") / max(len(t), 1)

    score = (
        min(len(t) / 800, 1.0) * 0.35
        + min(digits / 60, 1.0) * 0.25
        + min(units / 5, 1.0) * 0.20
        + min(keywords / 6, 1.0) * 0.20
    )
    if q_density > 0.05:
        score *= 0.5
    elif q_density > 0.02:
        score *= 0.8
    return round(score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Vision backends
# ─────────────────────────────────────────────────────────────────────────────
def _vision_read_file(image_path: Path, model: str) -> str:
    """File-based vision call: claude reads the image with its Read tool.
    Handles real-size photos without ARG_MAX limits of inline base64."""
    prompt = (
        f"Используй инструмент Read и открой файл изображения: {image_path}\n\n"
        + _OCR_PROMPT
    )
    result = subprocess.run(
        ["claude", "-p", "--model", model,
         "--add-dir", str(image_path.parent),
         "--permission-mode", "default",
         "--", prompt],
        capture_output=True, text=True, timeout=OCR_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI rc={result.returncode}: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def _vision_read_inline(image_path: Path, model: str) -> str:
    """Fallback: base64 image inline in the prompt (legacy path)."""
    img_b64 = base64.b64encode(image_path.read_bytes()).decode()
    suffix = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix, "image/jpeg")
    result = subprocess.run(
        ["claude", "-p", "--model", model,
         "--", f"[image: data:{mime};base64,{img_b64}] {_OCR_PROMPT}"],
        capture_output=True, text=True, timeout=OCR_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude CLI rc={result.returncode}: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def _vision_read(image_path: Path, model: str) -> str:
    """File-based first; inline base64 fallback for CLI versions without
    Read-tool image support in pipe mode."""
    try:
        text = _vision_read_file(image_path, model)
        if text:
            return text
    except (RuntimeError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.info("File-based vision failed (%s), trying inline base64", exc)
    return _vision_read_inline(image_path, model)


def _tesseract_available() -> bool:
    flag = os.getenv("OCR_TESSERACT", "auto").strip().lower()
    if flag in ("0", "no", "false", "off"):
        return False
    exe = shutil.which("tesseract")
    if not exe:
        return False
    try:
        langs = subprocess.run([exe, "--list-langs"], capture_output=True,
                               text=True, timeout=10).stdout
        return "rus" in langs
    except Exception:
        return False


def _tesseract_ocr(image_path: Path) -> str:
    exe = shutil.which("tesseract")
    result = subprocess.run(
        [exe, str(image_path), "stdout", "-l", "rus+eng", "--psm", "6"],
        capture_output=True, text=True, timeout=120,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def llm_cleanup_ocr(text: str) -> str:
    """Post-correct OCR garbage with the text LLM. Returns cleaned text or,
    on failure, the original."""
    if os.getenv("OCR_CLEANUP", "1").strip().lower() in ("0", "no", "false", "off"):
        return text
    try:
        import llm
        if not llm.available():
            return text
        cleaned = llm.chat(_CLEANUP_PROMPT + text[:12000], tier="fast",
                           timeout=90, temperature=0.0)
        if cleaned and len(cleaned) >= len(text) * 0.5:
            log.info("OCR cleanup: %d → %d chars", len(text), len(cleaned))
            return cleaned
    except Exception as exc:
        log.warning("OCR cleanup failed, keeping raw OCR: %s", exc)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def ocr_image(image_path: str | Path) -> OCRResult:
    """OCR a (possibly low-quality) photo of a medical document.

    Tries the original first, then preprocessed variants until a candidate
    scores well; returns the best-scoring transcription overall.
    Backend: vision LLM (Claude) or Tesseract (DeepSeek provider) with
    optional LLM post-correction.
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {image_path}")
    quality = assess_quality(img)
    backend = ocr_backend()
    log.info("Photo quality: blur=%.0f brightness=%.0f long_side=%d issues=%s | backend=%s",
             quality.blur_var, quality.brightness, quality.long_side,
             quality.issues or "none", backend)

    scores: dict[str, float] = {}
    best_text, best_variant, best_score = "", "original", 0.0
    attempts = 0

    with tempfile.TemporaryDirectory(dir=str(image_path.parent)) as tmp:
        try:
            variants = build_variants(image_path, Path(tmp), quality)
        except Exception as exc:
            log.warning("Preprocessing failed, OCR on original only: %s", exc)
            variants = [Variant("original", image_path)]

        if backend == "tesseract":
            # Local OCR: every variant is cheap, try all, best score wins
            for variant in variants:
                attempts += 1
                try:
                    text = _tesseract_ocr(variant.path)
                except Exception as exc:
                    log.warning("Tesseract failed on %s: %s", variant.name, exc)
                    scores[variant.name] = 0.0
                    continue
                score = ocr_quality_score(text)
                scores[variant.name] = score
                log.info("Tesseract variant=%s score=%.3f chars=%d",
                         variant.name, score, len(text))
                if score > best_score:
                    best_text, best_variant, best_score = text, variant.name, score
            if best_text:
                best_text = llm_cleanup_ocr(best_text)
                best_score = max(best_score, ocr_quality_score(best_text))
        else:
            # Vision LLM: attempts cost tokens — stop at first good candidate
            for i, variant in enumerate(variants):
                if attempts >= MAX_ATTEMPTS:
                    break
                model = FINAL_MODEL if i == len(variants) - 1 and attempts > 0 else VISION_MODEL
                attempts += 1
                try:
                    text = _vision_read(variant.path, model)
                except (RuntimeError, subprocess.TimeoutExpired) as exc:
                    log.warning("Vision OCR failed on %s: %s", variant.name, exc)
                    scores[variant.name] = 0.0
                    continue
                score = ocr_quality_score(text)
                scores[variant.name] = score
                log.info("OCR variant=%s model=%s score=%.3f chars=%d",
                         variant.name, model, score, len(text))
                if score > best_score:
                    best_text, best_variant, best_score = text, variant.name, score
                if best_score >= 0.55:
                    break  # good enough, stop spending tokens

            # Tesseract cross-check on the binarized variant (opt-in / auto)
            if best_score < 0.55 and _tesseract_available():
                tess_src = variants[-1].path if len(variants) > 1 else image_path
                try:
                    tess_text = _tesseract_ocr(tess_src)
                    tess_score = ocr_quality_score(tess_text) * 0.8  # discount: no layout reasoning
                    scores["tesseract"] = round(tess_score, 3)
                    log.info("Tesseract score=%.3f chars=%d", tess_score, len(tess_text))
                    if tess_score > best_score:
                        best_text, best_variant, best_score = tess_text, "tesseract", tess_score
                except Exception as exc:
                    log.warning("Tesseract failed: %s", exc)

    return OCRResult(
        text=best_text, variant=best_variant, score=best_score,
        attempts=attempts, quality=quality, scores=scores, backend=backend,
    )
