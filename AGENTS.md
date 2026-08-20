# AGENTS.md

Правила для любого агента в этом репозитории (Cursor Cloud Agent, локальный
Worker, IDE). Читать до начала работы.

## 1. Проект

- Локальный путь: `/home/idanilov/projs/fedor`.
- GitHub origin: `git@github.com:yacadestian/fedor.git`. Cursor Cloud Agents
  и PR идут сюда. Не менять `origin`.
- GitLab-зеркало: `git@gitlab.com:yacadestian/fedor.git` (remote `gitlab`).
- GitLab namespace: `yacadestian` (личный, не group).
- Это форк [petrovich-health](https://github.com/petrovich-opendev/petrovich-health)
  с локальными доработками (LLM-абстракция, OCR, дневник, голос, Яндекс.Диск,
  импорт истории). Не подтягивать upstream и не переименовывать продукт
  без явного запроса.
- Назначение: личный Telegram-бот углублённой аналитики здоровья. Не
  калорийный трекер. Данные клинические: анализы, протоколы, дневник.
- Стек: Python 3.11+, Telegram long-polling (`tg_listener.py`), ClickHouse,
  пакет `healthbot/`, YAML-базы в `knowledge/`.
- Вход бота: `.venv/bin/python tg_listener.py`. Диагност:
  `.venv/bin/python diagnostician.py --digest` / `--profile`.

## 2. Режим работы

Отвечать на русском. Пользователь часто пишет с телефона.

- Не задавать лишних вопросов, если задачу можно закрыть по репо.
- Менять только то, что нужно задаче. Не выдумывать фичи, очереди и деплой.
- Перед правкой файла прочитать его целиком.
- Комментарии только для нетривиальных решений. Не дублировать очевидное.

## 3. Команды

Офлайн-тесты (без ClickHouse, Telegram и LLM):

```
.venv/bin/python -m pytest tests/test_offline.py tests/test_ingest.py -q
```

Если в venv нет pytest: `.venv/bin/pip install pytest`, в коммиты
`requirements.txt` не тащить без запроса.

Запуск бота — только если пользователь явно попросил и `.env` на месте:

```
pkill -f 'tg_listener.py' || true
nohup .venv/bin/python tg_listener.py >> logs/bot.log 2>&1 & disown
```

Не стартовать ClickHouse, cron диагноста и импорт истории без запроса.
`sudo` недоступно. Системные пакеты не ставить.

## 4. Git

- После содержательной правки: коммит с понятным сообщением, затем
  `git push -u origin HEAD` и `git push gitlab HEAD`.
- Не делать `git rebase -i`, `git push --force`, `--no-verify`,
  `--amend` поверх уже запушенных коммитов.
- Не коммитить `.env`, `data/`, `logs/`, сессии, ключи, медицинские файлы,
  `beobachten.txt`, `tlsfront/`.

## 5. Секреты и медданные

- `.env` в корне (gitignored): `TELEGRAM_BOT_TOKEN`, ClickHouse, LLM, OCR,
  Яндекс.Диск, IMAP. В чат, логи агента и коммиты ключи не копировать.
- `.env.example` — только пустые/фиктивные значения.
- `users.yaml` в репо — шаблон. Живые username/chat_id не коммитить.
- Любые анализы, дневник, гипотезы, PDF, фото, голосовые — персональные
  данные. Не вставлять реальные значения в тесты, примеры, PR и ответы.
- Каждый пользователь изолирован по `owner_id`. Не ломать этот инвариант.

## 6. Локальный Worker (телефон)

- Имя машины: `srv125304-fedor`.
- Юнит: `~/.config/systemd/user/cursor-agent-worker-fedor.service`
  (копия в `infra/cursor-agent-worker-fedor.service`).
- `--worker-dir` = `/home/idanilov/projs/fedor`. Worker регистрирует репо
  по git remote origin (GitHub `yacadestian/fedor`). Чужой worker-dir
  подключать нельзя.
- Этот worker уже обслуживает живые сессии. Не рестартовать юнит без
  явной просьбы — текущий агент умрёт.

Проверка:

```
systemctl --user status cursor-agent-worker-fedor.service
cd /home/idanilov/projs/fedor && agent worker debug
```

Прямая ссылка и перезапуск (только если попросили): `docs/worker.md`.

## 7. Продуктовые инварианты

Менять только по явному запросу:

- Уровни доказательности [A]–[D] в ответах бота.
- YAML в `knowledge/` (протоколы, оптимумы, антагонисты).
- Изоляция пользователей и тихий игнор чужих Telegram-аккаунтов.
- Не добавлять дисклеймеры «проконсультируйтесь с врачом» от себя.
- Не ослаблять OCR/LLM-маршрутизацию «для простоты».

## 8. Чего делать НЕ нужно

- Не коммитить и не пушить чужие ветки в GitLab как `main`.
- Не менять `origin` с GitHub на GitLab.
- Не генерировать картинки и не публиковать ничего в Telegram/YouTube
  из этой сессии, если задача про правила, git или воркер.
- Не удалять ClickHouse-данные, `data/` и пользовательские загрузки.

## 9. Ссылки

- `README.md` — продукт, запуск, команды бота.
- `docs/worker.md` — локальный Worker и телефон.
- `schema.sql` — ClickHouse.
- `tests/` — офлайн-тесты.
