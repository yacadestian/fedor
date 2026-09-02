#!/usr/bin/env bash
set -euo pipefail
LOG=/workspace/logs/mail_5y_import.log
STALE_SECS=${STALE_SECS:-240}
SESSION=mail-import
cd /workspace

restart_import() {
  echo "$(date -u +%H:%M:%S) WATCHDOG: restarting import (stale ${1}s)" | tee -a /workspace/logs/watchdog.log
  pkill -9 -f 'python.*import_history' 2>/dev/null || true
  sleep 2
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "=$SESSION" 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s "$SESSION" -c /workspace -- bash -l
  # append separator then restart
  echo "===== WATCHDOG RESTART $(date -u -Iseconds) =====" >> "$LOG"
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$SESSION:0.0" C-c
  sleep 1
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t "$SESSION:0.0" \
    'cd /workspace && set -a && . ./.env && set +a && .venv/bin/python -u import_history.py --yadisk-dir "/Анализы." --mail --out data/history_bundle 2>&1 | tee -a logs/mail_5y_import.log' Enter
}

done_marker() {
  rg -q 'Mail scan done|newly ingested this run' "$LOG" 2>/dev/null || return 1
  # process must have exited after done
  ! pgrep -f 'python.*import_history' >/dev/null
}

echo "$(date -u +%H:%M:%S) WATCHDOG started stale=${STALE_SECS}s" | tee -a /workspace/logs/watchdog.log
while true; do
  if done_marker; then
    echo "$(date -u +%H:%M:%S) WATCHDOG: import finished cleanly" | tee -a /workspace/logs/watchdog.log
    exit 0
  fi
  # if process dead without done marker — restart
  if ! pgrep -f 'python.*import_history' >/dev/null; then
    # allow a moment for final flush
    sleep 5
    if rg -q 'Mail scan done|newly ingested this run' "$LOG" 2>/dev/null; then
      echo "$(date -u +%H:%M:%S) WATCHDOG: finished after exit" | tee -a /workspace/logs/watchdog.log
      exit 0
    fi
    restart_import "process-dead"
    sleep 30
    continue
  fi
  now=$(date +%s)
  mtime=$(stat -c '%Y' "$LOG" 2>/dev/null || echo 0)
  age=$((now - mtime))
  if (( age > STALE_SECS )); then
    restart_import "$age"
    sleep 45
    continue
  fi
  sleep 20
done
