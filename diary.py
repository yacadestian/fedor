"""Health diary: timestamped entries, wellbeing scores, symptoms,
personal thoughts and hypotheses with a lifecycle (active → confirmed/rejected).

Entries are detected by natural prefixes in chat ("гипотеза: ...",
"самочувствие 7/10 ...", "дневник: ...") or via /diary commands.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

log = logging.getLogger("health-bot")

ENTRY_LABELS = {
    "wellbeing": "Самочувствие",
    "symptom": "Симптом",
    "note": "Заметка",
    "hypothesis": "Гипотеза",
    "measurement": "Измерение",
}

_STATUS_LABELS = {
    "active": "🔬 активна",
    "confirmed": "✅ подтвердилась",
    "rejected": "❌ опровергнута",
}

# ─────────────────────────────────────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────────────────────────────────────
_HYPO_STATUS_RES = [
    (re.compile(r"^гипотеза\s+(?:подтвердилась|верна|сработала)\b[:\s]*(.*)", re.I | re.S), "confirmed"),
    (re.compile(r"^гипотеза\s+(?:опровергнута|отвергнута|неверна|не\s+подтвердилась|снята)\b[:\s]*(.*)", re.I | re.S), "rejected"),
]

_TYPE_PREFIX_RES = [
    (re.compile(r"^гипотеза\b[:\s–-]*", re.I), "hypothesis"),
    (re.compile(r"^(?:самочувствие|состояние)\b[:\s–-]*", re.I), "wellbeing"),
    (re.compile(r"^симптомы?\b[:\s–-]*", re.I), "symptom"),
    (re.compile(r"^(?:дневник|заметка|мысли?|наблюдение)\b[:\s–-]*", re.I), "note"),
]

_SCORE_10_RE = re.compile(r"(\d{1,2})\s*/\s*10")
_FIELD_RES = {
    "sleep_hours": re.compile(r"сон[:\s]*(\d+(?:[.,]\d+)?)\s*(?:ч|час|h)?", re.I),
    "energy_score": re.compile(r"энергия[:\s]*(\d{1,2})", re.I),
    "mood_score": re.compile(r"настроение[:\s]*(\d{1,2})", re.I),
    "wellbeing_score": re.compile(r"самочувствие[:\s]*(\d{1,2})(?!\s*/\s*10)?", re.I),
}
_SYMPTOMS_RE = re.compile(r"симптомы?[:\s]+(.+?)(?:$|\n)", re.I | re.S)


def _to_int(text: str | None) -> int | None:
    if not text:
        return None
    try:
        v = int(text)
        return v if 0 <= v <= 10 else None
    except ValueError:
        return None


def _to_float(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return None


def detect_diary_message(text: str) -> dict | None:
    """Classify an incoming chat text as a diary action.

    Returns:
        {"kind": "entry", "entry": {...}}        — new diary entry
        {"kind": "status", "status": ..., "match": str} — hypothesis lifecycle update
        None — not a diary message
    """
    stripped = text.strip()

    for rx, status in _HYPO_STATUS_RES:
        m = rx.match(stripped)
        if m:
            return {"kind": "status", "status": status, "match": (m.group(1) or "").strip()}

    for rx, entry_type in _TYPE_PREFIX_RES:
        m = rx.match(stripped)
        if not m:
            continue
        body = stripped[m.end():].strip() or stripped
        entry = {
            "entry_type": entry_type,
            "text": body,
            "wellbeing_score": None,
            "sleep_hours": None,
            "energy_score": None,
            "mood_score": None,
            "symptoms": "",
            "status": "active",
        }
        if entry_type == "wellbeing":
            m10 = _SCORE_10_RE.search(body)
            entry["wellbeing_score"] = _to_int(m10.group(1)) if m10 else None
            for field_name, frx in _FIELD_RES.items():
                fm = frx.search(body)
                if not fm:
                    continue
                if field_name == "sleep_hours":
                    entry["sleep_hours"] = _to_float(fm.group(1))
                elif entry[field_name] is None:
                    entry[field_name] = _to_int(fm.group(1))
            sm = _SYMPTOMS_RE.search(body)
            if sm:
                entry["symptoms"] = sm.group(1).strip()[:300]
        elif entry_type == "symptom":
            entry["symptoms"] = body[:300]
        return {"kind": "entry", "entry": entry}

    return None


def parse_diary_command(arg: str) -> dict:
    """Parse /diary argument: empty | number | 'N дней'."""
    arg = (arg or "").strip().lower()
    if not arg:
        return {"limit": 10, "days": None}
    m = re.match(r"^(\d{1,3})\s*(?:д|дн|дней|день|days?)?$", arg)
    if m:
        return {"limit": min(int(m.group(1)), 50), "days": None}
    m = re.match(r"^(?:за\s+)?(\d{1,3})\s*(?:дн|дней)\s*(?:назад)?$", arg)
    if m:
        return {"limit": 50, "days": int(m.group(1))}
    return {"limit": 10, "days": None}


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────
def _fmt_ts(ts) -> str:
    return ts.strftime("%d.%m %H:%M") if isinstance(ts, datetime) else str(ts)[:16]


def format_entry(row: dict) -> str:
    label = ENTRY_LABELS.get(row.get("entry_type", "note"), "Запись")
    line = f"{_fmt_ts(row.get('ts'))} | <b>{label}</b>: {row.get('text', '')[:200]}"
    extras = []
    if row.get("wellbeing_score") is not None:
        extras.append(f"самочувствие {row['wellbeing_score']}/10")
    if row.get("sleep_hours") is not None:
        extras.append(f"сон {row['sleep_hours']}ч")
    if row.get("energy_score") is not None:
        extras.append(f"энергия {row['energy_score']}/10")
    if row.get("mood_score") is not None:
        extras.append(f"настроение {row['mood_score']}/10")
    if row.get("symptoms"):
        extras.append(f"симптомы: {row['symptoms']}")
    if row.get("entry_type") == "hypothesis":
        extras.append(_STATUS_LABELS.get(row.get("status", "active"), row.get("status", "")))
    if extras:
        line += "\n    ↳ " + " | ".join(extras)
    return line


def format_entry_saved(entry: dict) -> str:
    label = ENTRY_LABELS.get(entry["entry_type"], "Запись")
    parts = [f"📔 <b>{label} записана</b> ({datetime.now().strftime('%d.%m %H:%M')})", entry["text"][:300]]
    scores = []
    if entry.get("wellbeing_score") is not None:
        scores.append(f"самочувствие {entry['wellbeing_score']}/10")
    if entry.get("sleep_hours") is not None:
        scores.append(f"сон {entry['sleep_hours']}ч")
    if entry.get("energy_score") is not None:
        scores.append(f"энергия {entry['energy_score']}/10")
    if entry.get("mood_score") is not None:
        scores.append(f"настроение {entry['mood_score']}/10")
    if entry.get("symptoms"):
        scores.append(f"симптомы: {entry['symptoms']}")
    if scores:
        parts.append(" | ".join(scores))
    if entry["entry_type"] == "hypothesis":
        parts.append("🔬 Буду учитывать её в ответах и сверять с новыми данными. "
                     "Скажи <i>гипотеза подтвердилась</i> или <i>гипотеза опровергнута</i>, когда станет ясно.")
    return "\n".join(parts)


def format_diary_list(rows: list[dict]) -> str:
    if not rows:
        return ("📔 Дневник пуст. Просто напиши, например:\n"
                "<i>самочувствие 7/10 сон 6.5ч энергия 6 симптомы: лёгкая головная боль</i>\n"
                "<i>гипотеза: ферритин падает из-за частой сдачи крови</i>")
    lines = ["📔 <b>Дневник здоровья</b>\n"]
    lines.extend(format_entry(r) for r in rows)
    return "\n".join(lines)


def format_hypotheses(rows: list[dict]) -> str:
    if not rows:
        return ("🔬 Активных гипотез нет. Добавь так:\n"
                "<i>гипотеза: инсулинорезистентность из-за недосыпа</i>")
    lines = ["🔬 <b>Твои гипотезы</b>\n"]
    for r in rows:
        status = _STATUS_LABELS.get(r.get("status", "active"), r.get("status", ""))
        lines.append(f"{_fmt_ts(r.get('ts'))} | {r.get('text', '')[:250]}\n    ↳ {status}")
    return "\n".join(lines)
