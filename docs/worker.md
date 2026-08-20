# Локальный Cursor Worker (Fedor)

Нужен, чтобы агент с телефона выполнял tool calls на этой машине,
в чекауте `/home/idanilov/projs/fedor`.

## Имя и репо

- Worker name: `srv125304-fedor`
- worker-dir: `/home/idanilov/projs/fedor`
- Лейбл Cursor: `yacadestian/fedor` (берётся из git `origin`, это GitHub)

Чужой worker (telegram-line, tg-done, dating-bot) для этого репо не использовать.

Юнит уже запущен и держит живые сессии. Рестарт убивает текущих агентов.
Без явной просьбы `systemctl --user restart` не делать.

## systemd

Файл в репо: `infra/cursor-agent-worker-fedor.service`.
Боевой путь: `~/.config/systemd/user/cursor-agent-worker-fedor.service`.

Установка (если юнита ещё нет):

```
install -m 644 \
  /home/idanilov/projs/fedor/infra/cursor-agent-worker-fedor.service \
  "$HOME/.config/systemd/user/cursor-agent-worker-fedor.service"
systemctl --user daemon-reload
systemctl --user enable --now cursor-agent-worker-fedor.service
```

Статус:

```
systemctl --user status cursor-agent-worker-fedor.service --no-pager
```

Диагностика CLI:

```
cd /home/idanilov/projs/fedor
agent worker debug
```

## Телефон

Прямой старт на этом worker:

https://cursor.com/agents#workerId=20b05eec-fc69-4a88-9533-71b5adef206b

Либо вручную:

1. Cursor iOS / Android → New agent.
2. Репозиторий GitHub `yacadestian/fedor` (GitLab — зеркало; Cursor
   матчит worker по GitHub origin).
3. Среда: My Machines, машина `srv125304-fedor`.
4. Если в приложении пустой список веток — открыть ссылку выше.

Worker ходит исходящим HTTPS на `api2.cursor.sh` / `api2direct.cursor.sh`.
Входящие порты не нужны.
