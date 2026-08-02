# Health Analytics Bot

Personal health analytics system via Telegram. Upload lab results (PDF, photo, text), get AI-powered analysis, track trends, manage protocols.

**Not a toy calorie counter.** A clinical-depth system that tracks amino acid profiles, drug-nutrient interactions, SPC control charts, and builds personalized health profiles.

> **This fork adds** (on top of [petrovich-health](https://github.com/petrovich-opendev/petrovich-health)):
> - **Robust OCR for low-quality photos** — OpenCV preprocessing (upscale, denoise, CLAHE, unsharp mask, deskew, adaptive binarization), quality assessment, multi-pass vision with per-candidate scoring, optional Tesseract cross-check (`image_ocr.py`)
> - **Health diary** — timestamped wellbeing/symptom/note entries with structured scores, and **personal hypotheses** with lifecycle (active → confirmed/rejected); diary flows into the LLM context (`diary.py`)
> - **Text mirroring** — every recognized PDF/photo is mirrored to a `.txt` file next to the source in `data/`
> - **Auto-schema** — ClickHouse tables are created automatically on first run

## Features

### Data Input
- **PDF** — auto-classifies (lab results / EEG / Holter / prescriptions / estimates), extracts biomarkers
- **Photo** — OCR via Claude Vision, same pipeline as PDF. **Low-quality shots are handled**: quality assessment (blur / exposure / resolution) → OpenCV enhancement (denoise, CLAHE, sharpening, deskew, binarization) → multi-pass recognition with best-candidate scoring → optional Tesseract cross-check. The bot tells you when it had to enhance the image.
- **Text** — paste lab results, auto-detected and parsed

### Health Diary
- **Timestamped entries** — `самочувствие 7/10 сон 6.5ч энергия 6 симптомы: ...` is parsed into structured scores
- **Hypotheses** — `гипотеза: ...` is tracked and injected into the LLM context; the bot cross-checks it against new lab data and diary entries. Close the loop with `гипотеза подтвердилась` / `гипотеза опровергнута`
- **Notes** — `дневник: ...` for free-form thoughts about your health
- `/diary` shows recent entries, `/hypotheses` lists active ones

### Analysis
- **Health Profile** — daily AI-generated cumulative profile (Opus), personalized to YOUR data
- **SPC Charts** — Statistical Process Control for each biomarker (personal control limits, not just lab norms)
- **Trend Graphs** — matplotlib charts with reference ranges and SPC limits
- **Correlations** — biomarkers grouped by organ system (Liver, Kidney, Iron, Hormones, etc.)
- **Amino Acid Tracking** — 9 essential amino acids per meal, DIAAS score, leucine threshold for mTOR

### Protocol Management
- **32 compounds** in knowledge base — testosterone, GH, peptides (BPC-157, TB-500, CJC-1295, etc.), AI, SERMs
- **Drug-nutrient interactions** — metformin→B12, GLP-1→GI protocol, Ca×Fe antagonism, Zn×Cu depletion
- **Optimal ranges** — not lab reference (population), but functional/optimal ranges per biomarker
- **Dosing calculators** — BMR/TDEE/macros, reconstitution, half-life clearance

### Smart Features  
- **Multi-user** — family access via `users.yaml`, fully isolated data per user
- **Chat memory** — L0 (raw log) → L1 (daily digest) → L2 (health profile), layered context
- **Self-learning** — clinical lessons accumulate from user corrections, apply to similar profiles
- **Reminders** — medication schedules via chat commands
- **PDF Report** — one-click report for your doctor with all data

### Evidence-Based Only
Every recommendation tagged with evidence level:
- **[A]** — meta-analysis / systematic review
- **[B]** — RCT on humans
- **[C]** — observational studies
- **[D]** — in vitro / early research

No supplements hype. No astrology. No "consult your doctor" disclaimers — direct conclusions.

## Architecture

```
Telegram Bot (long-polling)
  ├── PDF/Photo/Text → Classify (Haiku) → Extract (Sonnet) → ClickHouse
  ├── Q&A → Health Profile + Chat History + Lab Data → LLM (Opus) → Answer
  └── Reminders → Background thread (60s check)

Diagnostician (cron)
  ├── 23:55 — L1: Compress day's chat → daily_digest (Haiku)
  └── 07:00 — L2: Full health profile → health_profile (Opus) → Telegram
  
ClickHouse (11 tables)
  ├── lab_results, documents, upload_log
  ├── chat_log, daily_digest, health_profile
  ├── goals, nutrition_log, body_log
  ├── clinical_lessons, reminders, diary_entries
  └── All queries filtered by owner_id
```

## Requirements

- Python 3.11+
- ClickHouse (local or remote)
- Claude CLI (`claude` command) with active subscription (MAX or API)
- Telegram Bot (create via @BotFather)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/youruser/health-bot.git
cd health-bot

# 2. Setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 3. Create .env from example
cp .env.example .env
# Edit .env: add your Telegram bot token, CH credentials

# 4. Create ClickHouse schema (optional — created automatically on first run)
clickhouse-client < schema.sql

# 5. (Optional) Create dedicated CH user
clickhouse-client -q "CREATE USER health_bot IDENTIFIED BY 'your_password'"
clickhouse-client -q "GRANT ALL ON health_analytics.* TO health_bot"

# 6. Configure access
# Edit users.yaml: add your Telegram username

# 7. Run
.venv/bin/python3 tg_listener.py

# 8. (Optional) Setup cron for diagnostician
# Add to crontab:
# 55 20 * * * /path/to/.venv/bin/python3 /path/to/diagnostician.py --digest
# 0  4  * * * /path/to/.venv/bin/python3 /path/to/diagnostician.py --profile
```

## Telegram Commands

| Command | Description |
|---|---|
| `/goal` | Set health goal (muscle gain / fat loss / etc.) |
| `/eat` | Log meal with amino acid profile |
| `/weight` | Log body weight |
| `/week` | Weekly review |
| `/last` | Latest lab results |
| `/trend` | Biomarker trend with chart |
| `/search` | Full-text search across all data |
| `/abnormal` | Out-of-range biomarkers |
| `/correlations` | Cross-system analysis |
| `/spc` | SPC control charts |
| `/remind` | Medication reminders |
| `/diary` | Health diary entries (`/diary 25` for more) |
| `/hypotheses` | Active hypotheses (`/hypotheses все` — incl. closed) |
| `/report` | PDF report for doctor |
| `/summary` | AI health assessment |

### Diary — no commands needed

| You write | What happens |
|---|---|
| `самочувствие 7/10 сон 6.5ч энергия 6 симптомы: ...` | Structured wellbeing entry |
| `симптом: тянет правый бок` | Symptom entry |
| `гипотеза: ферритин падает из-за донорства` | Hypothesis, tracked & cross-checked |
| `гипотеза подтвердилась: донорство` | Hypothesis marked confirmed |
| `гипотеза опровергнута` | Latest hypothesis marked rejected |
| `дневник: начал принимать магний` | Free-form note |

## OCR quality knobs (.env)

| Variable | Default | Purpose |
|---|---|---|
| `OCR_VISION_MODEL` | `claude-sonnet-4-6` | Model for vision attempts |
| `OCR_FINAL_MODEL` | = `OCR_VISION_MODEL` | Stronger model for the last attempt (e.g. `claude-opus-4-6`) |
| `OCR_MAX_ATTEMPTS` | `3` | Max vision attempts across image variants |
| `OCR_TIMEOUT` | `180` | Per-attempt timeout, seconds |
| `OCR_TESSERACT` | `auto` | `1`/`0` force Tesseract cross-check; auto = on if `rus` language data installed |

## Knowledge Bases

| File | Contents |
|---|---|
| `protocols.yaml` | 32 compounds: dosing, pharmacokinetics, monitoring |
| `optimal_ranges.yaml` | 25+ biomarkers: optimal vs lab ranges, interactions |
| `nutrient_antagonists.yaml` | Mineral antagonisms, drug depletions, GI effects |

## Data Privacy

- All data stored locally in your ClickHouse instance
- Each user's data isolated by `owner_id` — no cross-user access
- Bot ignores unauthorized Telegram users silently
- No external data sharing — all processing on your server

## License

Apache License 2.0 — see [LICENSE](LICENSE)

## Author

**Ruslan Karimov** — Industrial AI architect, building production multi-agent systems for mining & health analytics.

- LinkedIn: [linkedin.com/in/karimov-ruslan-69790b7](https://www.linkedin.com/in/karimov-ruslan-69790b7/)
- Telegram: [@petrovich_mobile](https://t.me/petrovich_mobile)
- Platform: [agentdata.pro](https://agentdata.pro)

Built with [Claude Code](https://claude.ai/code) — AI-powered development.

## Contributing

1. Fork the repo
2. Create a feature branch (`feat/your-feature`)
3. Send a PR

Found a bug? Open an [Issue](https://github.com/petrovich-opendev/petrovich-health/issues) or use `/feedback` command in the bot.

## Star History

If this project helped you — give it a ⭐ on GitHub. It helps others find it.
