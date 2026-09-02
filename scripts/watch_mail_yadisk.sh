#!/usr/bin/env bash
# Restart mail cache / process workers if their logs go stale.
set -u
CACHE_LOG=/workspace/logs/sync_mail_yadisk.log
PROC_LOG=/workspace/logs/sync_mail_process.log
WD=/workspace/logs/watchdog.log
STALE=300
cd /workspace

start_cache() {
  echo "$(date -u +%H:%M:%S) start cache" >>"$WD"
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t mail-yadisk:0.0 \
    'cd /workspace && set -a && . ./.env && set +a && export OCR_BACKEND=vision_api OCR_TESSERACT=0 MAIL_SINCE_YEARS=0 MAIL_BATCH=50 && .venv/bin/python -u scripts/sync_mail_medical_to_yadisk.py --cache-only 2>&1 | tee -a logs/sync_mail_yadisk.log; echo EXIT:$?' Enter
}
start_proc() {
  echo "$(date -u +%H:%M:%S) start process" >>"$WD"
  tmux -f /exec-daemon/tmux.portal.conf has-session -t "=mail-process" 2>/dev/null \
    || tmux -f /exec-daemon/tmux.portal.conf new-session -d -s mail-process -c /workspace -- bash -l
  tmux -f /exec-daemon/tmux.portal.conf send-keys -t mail-process:0.0 \
    'cd /workspace && set -a && . ./.env && set +a && export OCR_BACKEND=vision_api OCR_TESSERACT=0 && .venv/bin/python -u scripts/sync_mail_medical_to_yadisk.py --process-only --loop 2>&1 | tee -a logs/sync_mail_process.log' Enter
}

alive_cache() { pgrep -f '.venv/bin/python -u scripts/sync_mail_medical_to_yadisk.py$' >/dev/null; }
alive_proc() { pgrep -f 'sync_mail_medical_to_yadisk.py --process-only' >/dev/null; }

echo "$(date -u +%H:%M:%S) watchdog start" >>"$WD"
while true; do
  now=$(date +%s)
  if ! alive_cache; then
    # finished cache-only is OK if log says Phase 1 done
    if ! tail -n 30 "$CACHE_LOG" | grep -q 'Phase 1 done'; then
      start_cache
      sleep 20
    fi
  else
    age=$(( now - $(stat -c %Y "$CACHE_LOG") ))
    if (( age > STALE )); then
      echo "$(date -u +%H:%M:%S) cache stale $age" >>"$WD"
      pgrep -f '.venv/bin/python -u scripts/sync_mail_medical_to_yadisk.py$' | xargs -r kill -9
      sleep 2
      start_cache
    fi
  fi
  if ! alive_proc; then
    start_proc
    sleep 15
  else
    age=$(( now - $(stat -c %Y "$PROC_LOG") ))
    if (( age > STALE )); then
      echo "$(date -u +%H:%M:%S) proc stale $age" >>"$WD"
      pgrep -f 'sync_mail_medical_to_yadisk.py --process-only' | xargs -r kill -9
      sleep 2
      start_proc
    fi
  fi
  sleep 30
done
