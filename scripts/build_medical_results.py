#!/usr/bin/env python3
"""Build medical_results*.csv and review markdowns from history_bundle.

Writes (gitignored under data/ + copy to /opt/cursor/artifacts):
  medical_results.csv
  medical_results_latest.csv
  medical_results_summary.md
  missing_tests.md
  ocr_review.md
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from healthbot.extractor import extract_biomarkers, validate_results  # noqa: E402

log = logging.getLogger("build_medical_results")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BUNDLE = Path("data/history_bundle")
OUT = Path("data/analysis")
ARTIFACTS = Path("/opt/cursor/artifacts")

# Canonical names we care about most (for summary / missing)
PRIORITY = [
    "Холестерин общий", "ЛПНП", "ЛПВП", "Триглицериды", "non-HDL", "ApoB", "Lp(a)",
    "Медь", "Церулоплазмин", "Цинк",
    "Витамин B12", "Витамин B12 активный",
    "Общий белок", "Альбумин",
    "Гемоглобин", "Эритроциты", "Лейкоциты", "Тромбоциты", "СОЭ",
    "АЛТ", "АСТ", "ГГТ", "Щелочная фосфатаза", "Билирубин общий",
    "Креатинин", "eGFR", "Мочевина",
    "Глюкоза", "HbA1c",
    "СРБ", "Ферритин", "Железо", "Трансферрин", "ОЖСС",
    "Витамин D", "Витамин B6", "Фолиевая кислота",
    "ТТГ", "Т4 свободный",
]

# More specific CBC *indices* first — otherwise "гемоглобин" inside MCHC
# aliases 334 г/л as hemoglobin, MPV 9.9 фл as platelets, etc.
INDEX_ALIASES = {
    r"mchc|сред\.?\s*конц|средн[аяейи].{0,30}концентрац|концентраци[яи].{0,20}гемогл": "MCHC",
    r"(?<![a-z])mch(?![ac])|сред\.?\s*сод|средн[аяейи].{0,30}(содерж|масс)|содержани[ея].{0,20}гемогл": "MCH",
    r"(?<![a-z])mcv(?![a-z])|средний объем эритроцит": "MCV",
    r"(?<![a-z])rdw|анизоцит|анизот\s*эритр": "RDW",
    r"(?<![a-z])mpv(?![a-z])|средний объем тромбоцит": "MPV",
    r"(?<![a-z])pdw|ширина распред.*тромбоц": "PDW",
}

NAME_ALIASES = {
    **INDEX_ALIASES,
    r"холестерин\s*общ": "Холестерин общий",
    r"общий\s*холестерин": "Холестерин общий",
    r"cholesterol\s*total|total\s*cholesterol": "Холестерин общий",
    r"лпнп|ldl": "ЛПНП",
    r"лпвп|hdl": "ЛПВП",
    r"триглицерид": "Триглицериды",
    r"non[\s\-]?hdl|не[\s\-]?лпвп": "non-HDL",
    r"апо\s*b|apo\s*b": "ApoB",
    r"lp\s*\(?a\)?|лп\s*\(?а\)?|липопротеин\s*\(а\)": "Lp(a)",
    r"^медь$|медь\s*в\s*сыворот|copper": "Медь",
    r"церулоплазмин|ceruloplasmin": "Церулоплазмин",
    r"^цинк$|цинк\s*в\s*сыворот|zinc": "Цинк",
    r"активн.*b12|b12\s*актив|holotranscobalamin|голотранскобаламин": "Витамин B12 активный",
    r"b12|цианокобаламин|кобаламин": "Витамин B12",
    r"общий\s*белок|total\s*protein": "Общий белок",
    r"альбумин": "Альбумин",
    r"гемоглобин(?!\s*a1)": "Гемоглобин",
    r"эритроцит": "Эритроциты",
    r"лейкоцит": "Лейкоциты",
    r"тромбоцит": "Тромбоциты",
    r"^соэ|скорость\s*оседания": "СОЭ",
    r"^алт$|аланинаминотрансфер|alanine\s*amino": "АЛТ",
    r"^аст$|аспартатаминотрансфер|aspartate\s*amino": "АСТ",
    r"^ггт$|гамма[\s\-]?глутамил": "ГГТ",
    r"щелочн.*фосфат|щф|alp": "Щелочная фосфатаза",
    r"билирубин\s*общ": "Билирубин общий",
    r"креатинин": "Креатинин",
    r"egfr|скф|скорость\s*клубочков": "eGFR",
    r"мочевин": "Мочевина",
    r"^глюкоз": "Глюкоза",
    r"hba1c|гликир.*гемогл": "HbA1c",
    r"^срб$|c[\s\-]?reactive|с[\s\-]реактав|с[\s\-]реактивн": "СРБ",
    r"ферритин": "Ферритин",
    r"^железо$|сыворот.*желез": "Железо",
    r"трансферрин": "Трансферрин",
    r"ожсс|tibc": "ОЖСС",
    r"25[\s\-]?oh|витамин\s*d|25\(oh\)": "Витамин D",
    r"витамин\s*b6|пиридокс": "Витамин B6",
    r"фолат|фолиев|витамин\s*b9": "Фолиевая кислота",
    r"^ттг$|тиреотроп|tsh": "ТТГ",
    r"т4\s*своб|свободн.*т4|free\s*t4|ft4": "Т4 свободный",
}

DEMO_RE = re.compile(
    r"(demo|synthetic|тест\s*пациент|sample\s*patient|john\s*doe|иванов\s*иван\s*иванович\s*01\.01\.1990)",
    re.I,
)
JUNK_FILE_RE = re.compile(r"(^back(@2x)?\.png$|media_|section_|style\.css|\.js$)", re.I)


@dataclass
class Row:
    date: str
    test_name: str
    value: float
    unit: str
    reference_low: float | None
    reference_high: float | None
    flag: str
    laboratory: str
    source_file: str
    confidence: str
    notes: str


def normalize_name(name: str) -> str:
    raw = (name or "").strip()
    low = raw.lower().replace("ё", "е")
    for pat, canon in NAME_ALIASES.items():
        if re.search(pat, low, re.I):
            return canon
    return raw


def flag_for(value: float, lo: float | None, hi: float | None) -> str:
    if lo is not None and value < lo:
        return "low"
    if hi is not None and value > hi:
        return "high"
    if lo is None and hi is None:
        return ""
    return "normal"


def load_text(bundle: Path, file_hash: str, filename: str) -> str:
    text_dir = bundle / "text"
    for p in text_dir.glob(f"{file_hash[:8]}_*.txt"):
        return p.read_text(encoding="utf-8", errors="replace")
    cache = bundle / ".textcache" / f"{file_hash}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    # fallback by filename stem
    safe = re.sub(r"[^\wа-яА-ЯёЁ.-]+", "_", filename)[:80]
    for p in text_dir.glob(f"*_{Path(safe).stem}.txt"):
        return p.read_text(encoding="utf-8", errors="replace")
    return ""


def is_demo(text: str, title: str, filename: str) -> bool:
    blob = f"{filename}\n{title}\n{text[:2000]}"
    return bool(DEMO_RE.search(blob))


def dedupe_key(r: Row) -> str:
    # Round value to reduce near-duplicates from OCR noise on same day/source
    return f"{r.date}|{r.test_name.lower()}|{round(r.value, 3)}|{r.unit.lower()}|{r.source_file}"


def extract_from_doc(rec: dict, text: str) -> tuple[list[Row], list[str]]:
    notes_global: list[str] = []
    if not text or len(text.strip()) < 40:
        return [], ["empty/short text"]
    if is_demo(text, rec.get("title") or "", rec.get("filename") or ""):
        return [], ["demo/synthetic skipped"]

    fallback = None
    if rec.get("doc_date"):
        try:
            fallback = date.fromisoformat(rec["doc_date"])
        except ValueError:
            pass

    try:
        extracted = extract_biomarkers(text)
        rows, warnings = validate_results(extracted, fallback_date=fallback)
    except Exception as exc:
        return [], [f"extract failed: {exc}"]

    notes_global.extend(warnings)
    lab = (extracted.get("lab_name") or rec.get("summary") or "")[:80]
    # Prefer explicit lab from extraction
    lab_name = extracted.get("lab_name") or ""
    out: list[Row] = []
    for r in rows:
        name = normalize_name(r["biomarker"])
        lo, hi = r.get("ref_low"), r.get("ref_high")
        collected = r["collected_at"]
        if isinstance(collected, date):
            d = collected.isoformat()
        else:
            d = str(collected)[:10]
        conf = "medium"
        # Higher confidence for text PDFs from known labs
        if lab_name and any(x in lab_name.lower() for x in ("helix", "хеликс", "инвитро", "invitro", "kdl", "гемотест")):
            conf = "high"
        if rec.get("filename", "").lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            conf = "low" if conf != "high" else "medium"
            notes = "OCR photo source — verify doubtful digits"
        else:
            notes = ""
        out.append(Row(
            date=d,
            test_name=name,
            value=float(r["value"]),
            unit=r.get("unit") or "",
            reference_low=lo,
            reference_high=hi,
            flag=flag_for(float(r["value"]), lo, hi),
            laboratory=lab_name or "unknown",
            source_file=rec.get("filename") or "",
            confidence=conf,
            notes=notes,
        ))
    return out, notes_global


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date", "test_name", "value", "unit", "reference_low", "reference_high",
            "flag", "laboratory", "source_file", "confidence", "notes",
        ])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x.date, x.test_name, x.source_file)):
            d = asdict(r)
            d["reference_low"] = "" if d["reference_low"] is None else d["reference_low"]
            d["reference_high"] = "" if d["reference_high"] is None else d["reference_high"]
            w.writerow(d)


def latest_rows(rows: list[Row]) -> list[Row]:
    best: dict[str, Row] = {}
    for r in rows:
        k = r.test_name.lower()
        if k not in best or r.date > best[k].date:
            best[k] = r
    return sorted(best.values(), key=lambda x: x.test_name)


def build_summary(rows: list[Row], docs_n: int, issues: list[str]) -> str:
    latest = {r.test_name: r for r in latest_rows(rows)}
    by_name: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_name[r.test_name].append(r)

    lines = [
        "# Medical results summary",
        "",
        f"- Documents processed (lab_results medical): **{docs_n}**",
        f"- Unique biomarker rows (after dedupe): **{len(rows)}**",
        f"- Unique tests: **{len(by_name)}**",
        f"- Date range: **{min((r.date for r in rows), default='—')}** → **{max((r.date for r in rows), default='—')}**",
        "",
        "## Priority panel (latest)",
        "",
        "| Test | Date | Value | Unit | Ref | Flag | Lab | Source |",
        "|---|---|---:|---|---|---|---|---|",
    ]
    for name in PRIORITY:
        r = latest.get(name)
        if not r:
            lines.append(f"| {name} | — | — | — | — | missing | — | — |")
            continue
        ref = ""
        if r.reference_low is not None or r.reference_high is not None:
            ref = f"{r.reference_low if r.reference_low is not None else '…'}–{r.reference_high if r.reference_high is not None else '…'}"
        lines.append(
            f"| {r.test_name} | {r.date} | {r.value} | {r.unit} | {ref} | {r.flag or '—'} | {r.laboratory} | {r.source_file} |"
        )

    # Explicit lipids / Cu / B12 block
    lines += ["", "## Focus values (all dated points)", ""]
    for name in ["Холестерин общий", "ЛПНП", "ЛПВП", "Триглицериды", "non-HDL", "ApoB", "Lp(a)",
                 "Медь", "Церулоплазмин", "Цинк", "Витамин B12", "Витамин B12 активный"]:
        pts = sorted(by_name.get(name, []), key=lambda x: x.date)
        if not pts:
            lines.append(f"- **{name}**: нет извлечённых значений")
            continue
        series = "; ".join(f"{p.date}: {p.value} {p.unit} [{p.flag or 'ok'}] ({p.source_file})" for p in pts)
        lines.append(f"- **{name}**: {series}")

    # Persistent abnormalities: same flag high/low on ≥2 distinct dates
    lines += ["", "## Persistent abnormalities (≥2 dates, same direction)", ""]
    found = False
    for name, pts in sorted(by_name.items()):
        highs = {p.date for p in pts if p.flag == "high"}
        lows = {p.date for p in pts if p.flag == "low"}
        if len(highs) >= 2:
            found = True
            lines.append(f"- ↑ **{name}** high on {sorted(highs)} (latest {latest[name].value} {latest[name].unit} on {latest[name].date})")
        if len(lows) >= 2:
            found = True
            lines.append(f"- ↓ **{name}** low on {sorted(lows)} (latest {latest[name].value} {latest[name].unit} on {latest[name].date})")
    if not found:
        lines.append("- нет устойчивых отклонений по текущим правилам (≥2 даты)")

    lines += ["", "## Import / extraction issues", ""]
    if issues:
        for i in issues[:80]:
            lines.append(f"- {i}")
        if len(issues) > 80:
            lines.append(f"- … +{len(issues)-80} more")
    else:
        lines.append("- none recorded")
    lines.append("")
    return "\n".join(lines)


def build_missing(rows: list[Row]) -> str:
    latest = {r.test_name: r for r in latest_rows(rows)}
    today = date.today()
    lines = [
        "# Missing / outdated tests",
        "",
        "Правила: missing = нет в извлечённых результатах; outdated = последнее значение старше 12 месяцев (биохимия/липиды/ОАК) или 6 месяцев (глюкоза/СРБ) или 18 месяцев (витамины/гормоны).",
        "",
        "## Missing priority tests",
        "",
    ]
    missing = [n for n in PRIORITY if n not in latest]
    if missing:
        for n in missing:
            lines.append(f"- {n}")
    else:
        lines.append("- нет — все приоритетные тесты представлены хотя бы одной точкой")

    lines += ["", "## Outdated (present but old)", ""]
    windows = {
        "Глюкоза": 180, "HbA1c": 180, "СРБ": 180, "СОЭ": 180,
        "ТТГ": 365, "Т4 свободный": 365, "Витамин D": 540, "Витамин B12": 540,
        "Витамин B12 активный": 540, "Витамин B6": 540, "Фолиевая кислота": 540,
        "Ферритин": 365, "Медь": 540, "Церулоплазмин": 540, "Цинк": 540,
    }
    any_old = False
    for name in PRIORITY:
        r = latest.get(name)
        if not r:
            continue
        try:
            d = date.fromisoformat(r.date)
        except ValueError:
            continue
        days = windows.get(name, 365)
        age = (today - d).days
        if age > days:
            any_old = True
            lines.append(f"- **{name}**: последнее {r.date} ({age} дн. назад; порог {days}) = {r.value} {r.unit}")
    if not any_old:
        lines.append("- нет устаревших по текущим порогам")
    lines.append("")
    return "\n".join(lines)


def build_ocr_review(rows: list[Row], issues: list[str], manifest: list[dict]) -> str:
    lines = [
        "# OCR / classification review",
        "",
        "## Low-confidence numeric results (verify against raw)",
        "",
    ]
    low = [r for r in rows if r.confidence == "low"]
    # also medium with extreme flags on priority
    suspect = low + [
        r for r in rows
        if r.confidence == "medium" and r.test_name in PRIORITY and r.flag in ("low", "high")
        and r.source_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".pdf"))
    ]
    # unique
    seen = set()
    uniq = []
    for r in suspect:
        k = dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    if not uniq:
        lines.append("- нет явных low-confidence кандидатов")
    else:
        lines.append("| date | test | value | unit | flag | conf | source | notes |")
        lines.append("|---|---|---:|---|---|---|---|---|")
        for r in sorted(uniq, key=lambda x: (x.date, x.test_name))[:120]:
            lines.append(
                f"| {r.date} | {r.test_name} | {r.value} | {r.unit} | {r.flag} | {r.confidence} | {r.source_file} | {r.notes} |"
            )

    lines += ["", "## Manifest anomalies", ""]
    for rec in manifest:
        fn = rec.get("filename") or ""
        err = rec.get("error") or ""
        if err and err != "duplicate" and not str(err).startswith("skip"):
            lines.append(f"- ERROR `{fn}`: {err[:200]}")
        if rec.get("medical") and (rec.get("text_chars") or 0) < 50:
            lines.append(f"- SHORT TEXT medical `{fn}` ({rec.get('text_chars')} chars) — possible OCR failure")
        if JUNK_FILE_RE.search(fn) and rec.get("medical"):
            lines.append(f"- MISCLASSIFIED junk marked medical: `{fn}`")
        title = (rec.get("title") or "").lower()
        if rec.get("medical") and any(x in title for x in ("неизвестный", "пустой документ")):
            lines.append(f"- WEAK classification `{fn}`: {rec.get('title')}")

    lines += ["", "## Extraction issues", ""]
    for i in issues[:100]:
        lines.append(f"- {i}")
    lines.append("")
    return "\n".join(lines)


def publish(path: Path) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, ARTIFACTS / path.name)


def main() -> int:
    manifest_path = BUNDLE / "manifest.json"
    if not manifest_path.exists():
        log.error("Missing %s — restore import first", manifest_path)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    # Prefer lab_results; also try research/consultation only if they look like labs
    candidates = [
        r for r in manifest
        if r.get("medical") and not (r.get("error") and r.get("error") not in ("", "duplicate"))
        and (r.get("kind") in ("lab_results", "") or "анализ" in (r.get("title") or "").lower()
             or "результат" in (r.get("title") or "").lower())
        and not JUNK_FILE_RE.search(r.get("filename") or "")
    ]
    # Always include explicit lab_results
    lab_docs = [r for r in manifest if r.get("medical") and r.get("kind") == "lab_results"
                and not JUNK_FILE_RE.search(r.get("filename") or "")]
    # merge unique by hash
    by_hash = {r["file_hash"]: r for r in lab_docs}
    for r in candidates:
        by_hash.setdefault(r["file_hash"], r)
    docs = list(by_hash.values())
    log.info("Extracting from %d medical docs (lab-focused)", len(docs))

    all_rows: list[Row] = []
    issues: list[str] = []
    cache_path = OUT / "extraction_cache.json"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    for i, rec in enumerate(docs, 1):
        h = rec["file_hash"]
        fn = rec.get("filename") or h
        text = load_text(BUNDLE, h, fn)
        if h in cache and cache[h].get("text_sha") == hashlib.sha256(text.encode()).hexdigest():
            rows = [Row(**x) for x in cache[h]["rows"]]
            issues.extend(cache[h].get("issues", []))
            all_rows.extend(rows)
            log.info("[%d/%d] CACHE %s → %d rows", i, len(docs), fn[:60], len(rows))
            continue
        log.info("[%d/%d] EXTRACT %s (%d chars)", i, len(docs), fn[:60], len(text))
        rows, doc_issues = extract_from_doc(rec, text)
        for msg in doc_issues:
            issues.append(f"{fn}: {msg}")
        all_rows.extend(rows)
        cache[h] = {
            "text_sha": hashlib.sha256(text.encode()).hexdigest(),
            "rows": [asdict(r) for r in rows],
            "issues": [f"{fn}: {m}" for m in doc_issues],
        }
        if i % 5 == 0:
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Dedupe
    seen = set()
    deduped: list[Row] = []
    for r in all_rows:
        k = dedupe_key(r)
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    log.info("Rows %d → %d after dedupe", len(all_rows), len(deduped))

    # Write artifacts
    csv_all = OUT / "medical_results.csv"
    csv_latest = OUT / "medical_results_latest.csv"
    md_sum = OUT / "medical_results_summary.md"
    md_miss = OUT / "missing_tests.md"
    md_ocr = OUT / "ocr_review.md"

    write_csv(csv_all, deduped)
    write_csv(csv_latest, latest_rows(deduped))
    md_sum.write_text(build_summary(deduped, len(docs), issues), encoding="utf-8")
    md_miss.write_text(build_missing(deduped), encoding="utf-8")
    md_ocr.write_text(build_ocr_review(deduped, issues, manifest), encoding="utf-8")

    for p in (csv_all, csv_latest, md_sum, md_miss, md_ocr):
        publish(p)
        log.info("Wrote %s (%d bytes)", p, p.stat().st_size)

    # Also symlink-friendly copies under bundle for user path expectation
    for p in (csv_all, csv_latest, md_sum, md_miss, md_ocr):
        shutil.copy2(p, BUNDLE / p.name)

    print(json.dumps({
        "docs": len(docs),
        "rows": len(deduped),
        "latest": len(latest_rows(deduped)),
        "out": str(OUT.resolve()),
        "artifacts": str(ARTIFACTS.resolve()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
