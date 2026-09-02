#!/usr/bin/env bash
# Watch Чеки-only mail import; restart on hang; exit on Готово.
set -u
LOG=/workspace/logs/mail_5y_import.log
WDLOG=/workspace/logs/watchdog.log
STALE=${STALE:-240}
cd /workspace

start_import() {
  echo "$(date -u +%H:%M:%S) start" >>"$WDLOG"
  echo "===== WD START $(date -u -Iseconds) =====" >>"$LOG"
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
  nohup env MAIL_FOLDERS='Чеки' .venv/bin/python -u import_history.py --mail --out data/history_bundle >>"$LOG" 2>&1 &
  echo $! > /tmp/import_history.pid
}

import_alive() {
  local pid
  pid=$(pgrep -f '.venv/bin/python -u import_history.py' | head -1 || true)
  [[ -n "${pid}" ]]
}

finished() {
  # Look at last 40 lines for completion after a recent run
  tail -n 40 "$LOG" | grep -q 'Готово:'
}

echo "$(date -u +%H:%M:%S) watchdog_cheki stale=$STALE" >>"$WDLOG"
if ! import_alive; then
  start_import
fi

while true; do
  if finished && ! import_alive; then
    echo "$(date -u +%H:%M:%S) FINISHED" >>"$WDLOG"
    exit 0
  fi
  if ! import_alive; then
    if finished; then
      echo "$(date -u +%H:%M:%S) FINISHED" >>"$WDLOG"
      exit 0
    fi
    echo "$(date -u +%H:%M:%S) dead" >>"$WDLOG"
    start_import
    sleep 30
    continue
  fi
  now=$(date +%s)
  mtime=$(stat -c '%Y' "$LOG")
  age=$((now - mtime))
  if (( age > STALE )); then
    echo "$(date -u +%H:%M:%S) stale=$age" >>"$WDLOG"
    # kill by exact cmdline match via pgrep PIDs only
    pgrep -f '.venv/bin/python -u import_history.py' | xargs -r kill -9
    sleep 2
    start_import
    sleep 40
  else
    sleep 20
  fi
done
