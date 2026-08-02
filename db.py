"""ClickHouse interface for health_analytics — multi-tenant by owner_id."""
from __future__ import annotations

import os
import uuid
from datetime import date, datetime
from typing import Any

import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

_client = None
_schema_ready = False


def ensure_schema() -> None:
    """Run schema.sql (idempotent CREATE IF NOT EXISTS) once per process.
    Uses a separate client on the default DB because the target database
    itself is created by the schema."""
    global _schema_ready
    if _schema_ready:
        return
    schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")
    try:
        ddl_client = clickhouse_connect.get_client(
            host=os.getenv("CH_HOST", "localhost"),
            port=int(os.getenv("CH_PORT", "8123")),
            database="default",
            username=os.getenv("CH_USER", "default"),
            password=os.getenv("CH_PASSWORD", ""),
        )
        with open(schema_path, encoding="utf-8") as f:
            for statement in f.read().split(";"):
                if statement.strip():
                    ddl_client.command(statement)
        _schema_ready = True
    except Exception as exc:
        import logging
        logging.getLogger("health-bot").warning("ensure_schema failed: %s", exc)


def get_client() -> clickhouse_connect.driver.Client:
    global _client
    if _client is None:
        ensure_schema()
        _client = clickhouse_connect.get_client(
            host=os.getenv("CH_HOST", "localhost"),
            port=int(os.getenv("CH_PORT", "8123")),
            database=os.getenv("CH_DATABASE", "health_analytics"),
            username=os.getenv("CH_USER", "default"),
            password=os.getenv("CH_PASSWORD", ""),
        )
    return _client


# ─── owner_id filter helper ─────────────────────────────────────────────────
def _own(owner_id: str) -> str:
    """WHERE clause fragment for owner isolation."""
    return f"owner_id = '{owner_id}'"


# ─── Inserts ─────────────────────────────────────────────────────────────────
def insert_lab_results(rows: list[dict], owner_id: str = "") -> int:
    if not rows:
        return 0
    client = get_client()
    columns = [
        "id", "collected_at", "category", "biomarker", "biomarker_original",
        "value", "unit", "ref_low", "ref_high", "is_abnormal",
        "lab_name", "source_file", "raw_text", "notes", "owner_id",
    ]
    data = []
    for r in rows:
        val = float(r["value"])
        ref_low = r.get("ref_low")
        ref_high = r.get("ref_high")
        is_abnormal = False
        if ref_low is not None and val < ref_low:
            is_abnormal = True
        if ref_high is not None and val > ref_high:
            is_abnormal = True
        data.append([
            str(uuid.uuid4()), r["collected_at"], r.get("category", "other"),
            r["biomarker"], r.get("biomarker_original", r["biomarker"]),
            val, r.get("unit", ""), ref_low, ref_high, is_abnormal,
            r.get("lab_name", ""), r.get("source_file", ""),
            r.get("raw_text", ""), r.get("notes", ""), owner_id,
        ])
    client.insert("lab_results", data, column_names=columns)
    return len(data)


def insert_document(
    collected_at: date, doc_type: str, title: str, source_file: str,
    full_text: str, lab_name: str = "", summary: str = "",
    owner_id: str = "",
) -> None:
    client = get_client()
    client.insert("documents", [[
        str(uuid.uuid4()), datetime.now(), collected_at, doc_type, title,
        lab_name, source_file, full_text, summary, owner_id,
    ]], column_names=[
        "id", "uploaded_at", "collected_at", "doc_type", "title",
        "lab_name", "source_file", "full_text", "summary", "owner_id",
    ])


def insert_chat_message(role: str, text: str, message_id: int = 0,
                        owner_id: str = "") -> None:
    client = get_client()
    client.insert("chat_log", [[
        datetime.now(), role, text, message_id, 0, owner_id,
    ]], column_names=["ts", "role", "text", "message_id", "tokens_used", "owner_id"])


def insert_upload_log(
    source_file: str, file_size: int, pages: int, biomarkers_extracted: int,
    lab_name: str, collected_at: date, status: str = "ok",
    error_message: str = "", raw_text: str = "",
    owner_id: str = "",
) -> None:
    client = get_client()
    client.insert("upload_log", [[
        str(uuid.uuid4()), datetime.now(), source_file, file_size, pages,
        biomarkers_extracted, lab_name, collected_at, status,
        error_message, raw_text, owner_id,
    ]], column_names=[
        "id", "uploaded_at", "source_file", "file_size_bytes", "pages",
        "biomarkers_extracted", "lab_name", "collected_at", "status",
        "error_message", "raw_text", "owner_id",
    ])


# ─── Diary ───────────────────────────────────────────────────────────────────
def insert_diary_entry(entry: dict, owner_id: str = "") -> str:
    """Insert a diary entry; returns its id (str)."""
    client = get_client()
    entry_id = entry.get("id") or str(uuid.uuid4())
    client.insert("diary_entries", [[
        entry_id, owner_id, entry.get("ts") or datetime.now(),
        entry["entry_type"], entry["text"],
        entry.get("wellbeing_score"), entry.get("sleep_hours"),
        entry.get("energy_score"), entry.get("mood_score"),
        entry.get("symptoms", ""), entry.get("tags", []),
        entry.get("status", "active"), entry.get("source", "telegram"),
    ]], column_names=[
        "id", "owner_id", "ts", "entry_type", "text",
        "wellbeing_score", "sleep_hours", "energy_score", "mood_score",
        "symptoms", "tags", "status", "source",
    ])
    return entry_id


def query_diary_entries(owner_id: str = "", limit: int = 10,
                        entry_type: str | None = None,
                        days: int | None = None) -> list[dict]:
    client = get_client()
    where = f"WHERE {_own(owner_id)}"
    params: dict = {"lim": limit}
    if entry_type:
        where += " AND entry_type = {et:String}"
        params["et"] = entry_type
    if days:
        where += " AND ts >= now() - INTERVAL {d:UInt32} DAY"
        params["d"] = days
    result = client.query(
        f"SELECT id, ts, entry_type, text, wellbeing_score, sleep_hours, "
        f"energy_score, mood_score, symptoms, status FROM diary_entries FINAL "
        f"{where} ORDER BY ts DESC LIMIT {{lim:UInt32}}",
        parameters=params,
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_hypotheses(owner_id: str = "", status: str | None = None,
                     limit: int = 20) -> list[dict]:
    client = get_client()
    where = f"WHERE {_own(owner_id)} AND entry_type = 'hypothesis'"
    params: dict = {"lim": limit}
    if status:
        where += " AND status = {st:String}"
        params["st"] = status
    result = client.query(
        f"SELECT id, ts, text, status FROM diary_entries FINAL "
        f"{where} ORDER BY ts DESC LIMIT {{lim:UInt32}}",
        parameters=params,
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def set_hypothesis_status(owner_id: str, entry_id: str, status: str,
                          match_text: str = "") -> dict | None:
    """Re-insert the hypothesis with a new status (ReplacingMergeTree swap).
    If match_text given, the row's text must contain it (case-insensitive).
    Returns the updated row or None if not found."""
    client = get_client()
    result = client.query(
        f"SELECT id, ts, entry_type, text, wellbeing_score, sleep_hours, "
        f"energy_score, mood_score, symptoms, tags, status, source "
        f"FROM diary_entries FINAL "
        f"WHERE {_own(owner_id)} AND id = {{eid:String}} AND entry_type = 'hypothesis' "
        f"ORDER BY ts DESC LIMIT 1",
        parameters={"eid": entry_id},
    )
    if not result.result_rows:
        return None
    row = dict(zip(result.column_names, result.result_rows[0]))
    if match_text and match_text.lower() not in row["text"].lower():
        return None
    row["status"] = status
    insert_diary_entry(row, owner_id)
    return row


def find_latest_active_hypothesis(owner_id: str, match_text: str = "") -> dict | None:
    """Latest active hypothesis, optionally filtered by substring."""
    hypotheses = query_hypotheses(owner_id, status="active", limit=50)
    if match_text:
        hypotheses = [h for h in hypotheses if match_text.lower() in h["text"].lower()]
    return hypotheses[0] if hypotheses else None


# ─── Queries (all filtered by owner_id) ─────────────────────────────────────
def query_recent_chat(limit: int = 10, owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT ts, role, text FROM chat_log "
        f"WHERE {_own(owner_id)} ORDER BY ts DESC LIMIT {{lim:UInt32}}",
        parameters={"lim": limit},
    )
    rows = [dict(zip(result.column_names, row)) for row in result.result_rows]
    return list(reversed(rows))


def query_biomarker_trend(biomarker: str, limit: int = 50,
                          owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, value, unit, ref_low, ref_high, is_abnormal, lab_name "
        f"FROM lab_results WHERE {_own(owner_id)} AND biomarker ILIKE {{bm:String}} "
        f"ORDER BY collected_at",
        parameters={"bm": f"%{biomarker}%"},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows[:limit]]


def query_latest_results(limit: int = 30, owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, category, biomarker, value, unit, ref_low, ref_high, "
        f"is_abnormal, lab_name FROM lab_results WHERE {_own(owner_id)} "
        f"ORDER BY collected_at DESC, biomarker LIMIT {{lim:UInt32}}",
        parameters={"lim": limit},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_abnormal(limit: int = 50, owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, category, biomarker, value, unit, ref_low, ref_high, "
        f"lab_name FROM lab_results WHERE {_own(owner_id)} AND is_abnormal = true "
        f"ORDER BY collected_at DESC LIMIT {{lim:UInt32}}",
        parameters={"lim": limit},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_all_biomarkers(owner_id: str = "") -> list[str]:
    client = get_client()
    result = client.query(
        f"SELECT DISTINCT biomarker FROM lab_results WHERE {_own(owner_id)} ORDER BY biomarker"
    )
    return [row[0] for row in result.result_rows]


def query_summary_stats(owner_id: str = "") -> dict:
    client = get_client()
    r = client.query(
        f"SELECT count(), min(collected_at), max(collected_at), uniqExact(biomarker), "
        f"uniqExact(source_file) FROM lab_results WHERE {_own(owner_id)}"
    )
    row = r.result_rows[0] if r.result_rows else (0, None, None, 0, 0)
    has_data = bool(row[0])
    return {
        "total_records": row[0],
        "earliest_date": str(row[1]) if has_data and row[1] else "N/A",
        "latest_date": str(row[2]) if has_data and row[2] else "N/A",
        "unique_biomarkers": row[3],
        "unique_files": row[4],
    }


def query_fulltext_search(query: str, limit: int = 30,
                          owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, category, biomarker, value, unit, ref_low, ref_high, "
        f"is_abnormal, lab_name, source_file FROM lab_results "
        f"WHERE {_own(owner_id)} AND ("
        f"  positionCaseInsensitiveUTF8(biomarker, {{q:String}}) > 0 "
        f"  OR positionCaseInsensitiveUTF8(biomarker_original, {{q:String}}) > 0 "
        f"  OR positionCaseInsensitiveUTF8(raw_text, {{q:String}}) > 0) "
        f"ORDER BY collected_at DESC LIMIT {{lim:UInt32}}",
        parameters={"q": query, "lim": limit},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_documents_search(query: str, limit: int = 10,
                           owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, doc_type, title, lab_name, "
        f"substring(full_text, "
        f"  greatest(1, positionCaseInsensitiveUTF8(full_text, {{q:String}}) - 100), "
        f"  300) as context_snippet "
        f"FROM documents WHERE {_own(owner_id)} AND ("
        f"  positionCaseInsensitiveUTF8(full_text, {{q:String}}) > 0 "
        f"  OR positionCaseInsensitiveUTF8(title, {{q:String}}) > 0) "
        f"ORDER BY collected_at DESC LIMIT {{lim:UInt32}}",
        parameters={"q": query, "lim": limit},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_all_documents(limit: int = 20, owner_id: str = "") -> list[dict]:
    client = get_client()
    result = client.query(
        f"SELECT collected_at, doc_type, title, lab_name, length(full_text) as text_len "
        f"FROM documents WHERE {_own(owner_id)} "
        f"ORDER BY collected_at DESC LIMIT {{lim:UInt32}}",
        parameters={"lim": limit},
    )
    return [dict(zip(result.column_names, row)) for row in result.result_rows]


def query_spc_data(owner_id: str = "") -> dict[str, list]:
    """Get biomarker time series for SPC (only those with ≥2 points)."""
    client = get_client()
    result = client.query(
        f"SELECT biomarker, collected_at, value, unit, ref_low, ref_high "
        f"FROM lab_results WHERE {_own(owner_id)} "
        f"AND biomarker IN ("
        f"  SELECT biomarker FROM lab_results WHERE {_own(owner_id)} "
        f"  GROUP BY biomarker HAVING count() >= 2"
        f") ORDER BY biomarker, collected_at"
    )
    from spc import SPCPoint
    series: dict[str, list] = {}
    for row in result.result_rows:
        bm = row[0]
        series.setdefault(bm, []).append(SPCPoint(
            date=row[1], value=row[2], unit=row[3],
            ref_low=row[4], ref_high=row[5],
        ))
    return series


def query_health_profile(owner_id: str = "") -> str | None:
    client = get_client()
    result = client.query(
        f"SELECT date, profile_text, overall_status, key_findings, watchlist, "
        f"correlations, missing_data, alerts "
        f"FROM health_profile WHERE {_own(owner_id)} ORDER BY date DESC LIMIT 1"
    )
    if not result.result_rows:
        return None
    r = result.result_rows[0]
    parts = [f"Дата профиля: {r[0]}", f"Статус: {r[2]}", f"\n{r[1]}"]
    if r[3] and r[3] != "[]":
        parts.append(f"\nКлючевые находки: {r[3]}")
    if r[4] and r[4] != "[]":
        parts.append(f"\nПод наблюдением: {r[4]}")
    if r[5] and r[5] != "[]":
        parts.append(f"\nКорреляции: {r[5]}")
    if r[6] and r[6] != "[]":
        parts.append(f"\nНе хватает: {r[6]}")
    if r[7] and r[7] != "[]":
        parts.append(f"\nАлерты: {r[7]}")
    return "\n".join(parts)


def query_recent_digests(days: int = 7, owner_id: str = "") -> str:
    client = get_client()
    result = client.query(
        f"SELECT date, digest, user_concerns, new_info FROM daily_digest "
        f"WHERE {_own(owner_id)} ORDER BY date DESC LIMIT {{d:UInt32}}",
        parameters={"d": days},
    )
    if not result.result_rows:
        return ""
    lines = []
    for r in result.result_rows:
        line = f"{r[0]}: {r[1]}"
        if r[2]:
            line += f" | Беспокоит: {r[2]}"
        if r[3]:
            line += f" | Новое: {r[3]}"
        lines.append(line)
    return "\n".join(lines)


def query_diary_for_context(owner_id: str = "", days: int = 14,
                            limit: int = 30) -> str:
    """Compact diary block for the LLM context."""
    rows = query_diary_entries(owner_id, limit=limit, days=days)
    if not rows:
        return ""
    lines = []
    for r in reversed(rows):
        ts = r["ts"].strftime("%d.%m %H:%M") if hasattr(r["ts"], "strftime") else str(r["ts"])[:16]
        line = f"  {ts} | {r['entry_type']}: {r['text'][:250]}"
        extras = []
        if r.get("wellbeing_score") is not None:
            extras.append(f"самочувствие {r['wellbeing_score']}/10")
        if r.get("sleep_hours") is not None:
            extras.append(f"сон {r['sleep_hours']}ч")
        if r.get("symptoms"):
            extras.append(f"симптомы: {r['symptoms']}")
        if r["entry_type"] == "hypothesis":
            extras.append(f"статус: {r.get('status', 'active')}")
        if extras:
            line += " (" + "; ".join(extras) + ")"
        lines.append(line)
    return "\n".join(lines)


def query_for_llm_context(question: str, owner_id: str = "") -> str:
    stats = query_summary_stats(owner_id)
    profile = query_health_profile(owner_id)
    digests = query_recent_digests(7, owner_id)
    diary_block = query_diary_for_context(owner_id)
    hypotheses = query_hypotheses(owner_id, status="active")

    if stats["total_records"] == 0 and not profile and not diary_block:
        return "(Нет данных в базе. Загрузите анализы через PDF или ведите дневник.)"

    client = get_client()

    latest = client.query(
        f"SELECT collected_at, category, biomarker, value, unit, ref_low, ref_high, is_abnormal "
        f"FROM lab_results WHERE {_own(owner_id)} "
        f"ORDER BY collected_at DESC, biomarker LIMIT 60"
    )
    latest_lines = []
    for row in latest.result_rows:
        flag = " [!!!]" if row[7] else ""
        ref = ""
        if row[5] is not None or row[6] is not None:
            ref = f" (норма: {row[5] or '?'}–{row[6] or '?'})"
        latest_lines.append(f"  {row[0]} | {row[1]} | {row[2]}: {row[3]} {row[4]}{ref}{flag}")

    abnormal = client.query(
        f"SELECT collected_at, biomarker, value, unit, ref_low, ref_high "
        f"FROM lab_results WHERE {_own(owner_id)} AND is_abnormal = true "
        f"ORDER BY collected_at DESC LIMIT 30"
    )
    abnormal_lines = []
    for row in abnormal.result_rows:
        abnormal_lines.append(
            f"  {row[0]} | {row[1]}: {row[2]} {row[3]} (норма: {row[4] or '?'}–{row[5] or '?'})")

    docs = client.query(
        f"SELECT collected_at, doc_type, title, "
        f"substring(full_text, 1, 1500) as text_preview "
        f"FROM documents WHERE {_own(owner_id)} ORDER BY collected_at DESC LIMIT 10"
    )
    doc_lines = []
    for row in docs.result_rows:
        doc_lines.append(f"--- {row[2]} ({row[1]}, {row[0]}) ---\n{row[3]}")

    sections = []
    if profile:
        sections.append(f"=== HEALTH PROFILE ===\n{profile}")
    if digests:
        sections.append(f"=== ДАЙДЖЕСТЫ (7 дней) ===\n{digests}")
    if diary_block:
        sections.append(f"=== ДНЕВНИК ЗДОРОВЬЯ (14 дней) ===\n{diary_block}")
    if hypotheses:
        hypo_lines = "\n".join(f"  {h['ts'].strftime('%d.%m') if hasattr(h['ts'], 'strftime') else h['ts']}: {h['text'][:250]}"
                               for h in hypotheses)
        sections.append(
            f"=== АКТИВНЫЕ ГИПОТЕЗЫ ПОЛЬЗОВАТЕЛЯ ===\n{hypo_lines}\n"
            f"Учитывай эти гипотезы в рассуждениях: подтверждай или опровергай их "
            f"данными анализов и дневника, говори когда новые данные проливают на них свет.")

    sections.append(
        f"=== БАЗА ===\nЗаписей: {stats['total_records']}, "
        f"период: {stats['earliest_date']}—{stats['latest_date']}, "
        f"показателей: {stats['unique_biomarkers']}, файлов: {stats['unique_files']}")
    sections.append(f"=== РЕЗУЛЬТАТЫ ===\n{chr(10).join(latest_lines) or '(нет)'}")
    sections.append(f"=== ВНЕ НОРМЫ ===\n{chr(10).join(abnormal_lines) or '(в норме)'}")
    sections.append(f"=== ДОКУМЕНТЫ ===\n{chr(10).join(doc_lines) or '(нет)'}")

    return "\n\n".join(sections)
