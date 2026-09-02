"""Offline tests for the history-import tooling (no network, no LLM)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from healthbot import ingest, mailscan, yadisk


# ─────────────────────────────────────────────────────────────────────────────
# Yandex Disk PROPFIND parsing
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PROPFIND = """<?xml version="1.0" encoding="utf-8"?>
<d:multistatus xmlns:d="DAV:">
  <d:response>
    <d:href>/Здоровье/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype>
    <d:getcontentlength>0</d:getcontentlength></d:prop></d:propstat>
  </d:response>
  <d:response>
    <d:href>/Здоровье/2024/</d:href>
    <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype>
    <d:getcontentlength>0</d:getcontentlength></d:prop></d:propstat>
  </d:response>
  <d:response>
    <d:href>/%D0%97%D0%B4%D0%BE%D1%80%D0%BE%D0%B2%D1%8C%D0%B5/kdl_2024.pdf</d:href>
    <d:propstat><d:prop><d:resourcetype/>
    <d:getcontentlength>102400</d:getcontentlength></d:prop></d:propstat>
  </d:response>
</d:multistatus>"""


class TestYadiskParsing:
    def test_parse_propfind(self):
        entries = yadisk.parse_propfind(SAMPLE_PROPFIND, "/Здоровье")
        dirs = [e for e in entries if e[2]]
        files = [e for e in entries if not e[2]]
        assert dirs == [("/Здоровье/2024/", 0, True)]
        assert len(files) == 1
        assert files[0][0].endswith("kdl_2024.pdf")  # URL-decoded
        assert files[0][1] == 102400


# ─────────────────────────────────────────────────────────────────────────────
# Mail scanning helpers
# ─────────────────────────────────────────────────────────────────────────────
class TestMailHelpers:
    def test_mutf7_decode(self):
        # "Чеки" in modified UTF-7
        assert mailscan.decode_mutf7("&BCcENQQ6BDg-") == "Чеки"
        assert mailscan.decode_mutf7("INBOX") == "INBOX"
        assert mailscan.decode_mutf7("&BCcENQQ6BDg- 2024") == "Чеки 2024"

    def test_match_folders(self):
        all_f = ["INBOX", "&BCcENQQ6BDg-", "Sent", "Spam"]
        assert set(mailscan.match_folders(all_f, ["INBOX", "Чеки"])) == {"INBOX", "&BCcENQQ6BDg-"}
        assert mailscan.match_folders(all_f, ["неттакой"]) == []

    def test_bodystructure_pdf_parts(self):
        struct = ('("HEADER" NIL) (("text" "plain" NIL) '
                  '("application" "pdf" ("name" "result.pdf") NIL "base64" 50234) '
                  '("text" "html" NIL))')
        assert mailscan._message_has_pdf(struct) == [2]
        struct2 = '(("text" "plain" NIL) ("image" "jpeg" ("name" "photo.jpg")))'
        assert mailscan._message_has_pdf(struct2) == []

    def test_mime_header_decode(self):
        assert "Результаты" in mailscan.decode_mime_header(
            "=?UTF-8?B?0KDQtdC30YPQu9GM0YLQsNGC0Ys=?= test")


# ─────────────────────────────────────────────────────────────────────────────
# Ingest pipeline
# ─────────────────────────────────────────────────────────────────────────────
class TestIngest:
    def _fake_llm(self, monkeypatch, medical: bool, kind: str = "lab_results"):
        def fake_chat(prompt, tier="fast", timeout=60, temperature=0.0):
            return json.dumps({
                "medical": medical, "kind": kind,
                "title": "Тест", "date": "2024-03-15", "summary": "…"})
        monkeypatch.setattr("healthbot.llm.chat", fake_chat)

    def test_medical_filter_routes(self, tmp_path, monkeypatch):
        self._fake_llm(monkeypatch, medical=True)
        rec = ingest.process_file("note.txt", ("Гемоглобин 145 г/л норма 130-160 "
                                               "Эритроциты 4.5").encode(),
                                  "mail", "INBOX | test | today", tmp_path, set())
        assert rec.medical and rec.kind == "lab_results"
        assert (tmp_path / "text").exists()  # txt mirror saved for medical

        rec2 = ingest.process_file("receipt.txt", "Чек магазина, итого 500 рублей".encode(),
                                   "mail", "Чеки | чек | today", tmp_path, set())
        assert rec2.medical  # same fake LLM; content irrelevant here

    def test_dedup(self, tmp_path, monkeypatch):
        self._fake_llm(monkeypatch, medical=True)
        seen: set[str] = set()
        blob = b"%PDF fake but same content " * 10
        r1 = ingest.process_file("a.txt", blob, "mail", "s1", tmp_path, seen)
        r2 = ingest.process_file("b.txt", blob, "yadisk", "s2", tmp_path, seen)
        assert r1.error != "duplicate"
        assert r2.error == "duplicate"

    def test_consolidated_document(self, tmp_path, monkeypatch):
        recs = [
            ingest.IngestRecord("h1", "b.pdf", "mail", "INBOX | s | d", 100,
                                medical=True, kind="lab_results",
                                title="Биохимия", doc_date="2025-01-10"),
            ingest.IngestRecord("h2", "a.pdf", "yadisk", "/Здоровье/a.pdf", 100,
                                medical=True, kind="consultation",
                                title="Консультация", doc_date="2023-06-01"),
            ingest.IngestRecord("h3", "receipt.pdf", "mail", "Чеки | чек | d", 100,
                                medical=False),
        ]
        for r in recs[:2]:
            ingest._save_text_cache(tmp_path, r.file_hash, f"Текст {r.title}")
        doc = ingest.build_consolidated(recs, tmp_path)
        assert "2023" in doc and "2025" in doc
        assert doc.index("Консультация") < doc.index("Биохимия")  # sorted by date
        assert "чек" not in doc.lower()  # non-medical excluded
        assert "отфильтровано немедицинских: 1" in doc

    def test_manifest_roundtrip(self, tmp_path):
        recs = [ingest.IngestRecord("h1", "a.pdf", "mail", "ref", 5, medical=True)]
        ingest.save_manifest(recs, tmp_path)
        loaded = ingest.load_manifest(tmp_path)
        assert loaded[0].file_hash == "h1" and loaded[0].medical

    def test_cid_heavy_detects_glyph_soup(self):
        soup = "(cid:24)" * 30 + " padding"
        assert ingest._cid_heavy(soup)
        assert not ingest._cid_heavy("Гемоглобин 145 г/л норма 130-160")

    def test_pdf_to_text_prefers_ocr_over_longer_cid(self, tmp_path, monkeypatch):
        soup = "(cid:1)" * 80  # long unreadable layer
        monkeypatch.setattr(
            "healthbot.pdf_parser.extract_text",
            lambda p: (soup, 1),
        )
        monkeypatch.setattr(
            "healthbot.ingest.scanned_pdf_to_text",
            lambda p, max_pages=10: "Группа крови 0 (I)\nРезус-фактор Rh (+)",
        )
        pdf = tmp_path / "cid.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        text = ingest.pdf_to_text(pdf)
        assert "Группа крови" in text
        assert "(cid:" not in text


class TestBiomarkerAliases:
    def test_cbc_indices_not_aliased_to_hb_rbc_plt(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from build_medical_results import normalize_name
        assert normalize_name("Гемоглобин") == "Гемоглобин"
        assert normalize_name("Тромбоциты") == "Тромбоциты"
        assert normalize_name("Эритроциты") == "Эритроциты"
        assert normalize_name("Сред. конц. гемоглобина в эр") == "MCHC"
        assert normalize_name("Средняя концентрация гемоглобина в эритроците") == "MCHC"
        assert normalize_name("Сред. сод. гемоглобина") == "MCH"
        assert normalize_name("Средняя масса гемоглобина в эритроците") == "MCH"
        assert normalize_name("Средний объем тромбоцита") == "MPV"
        assert normalize_name("Коэфф анизот эритр") == "RDW"
