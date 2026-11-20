#!/usr/bin/env bash
# Unattended CooperBench experiment, start to finish, on the GPU machine:
#   1. the smoke pair in all three arms, and a sanity check of its logs;
#   2. the flash subset (50 pairs), one arm after another;
#   3. the report, then the experiment's containers are removed (GPU freed).
# A watchdog stops everything this experiment started (label cpc-cb=1, never anyone
# else's containers) once the LLM spend in the ledgers passes $SPEND_CAP.
#
#   VERTEX_PROJECT=... nohup bash scripts/cooperbench_queue.sh > queue_out/cb_queue.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CB="$ROOT/scripts/cooperbench.sh"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs_cooperbench}"
SPEND_CAP="${SPEND_CAP:-300}"
SUBSET="${SUBSET:-flash}"
SMOKE=(-r dottxt_ai_outlines_task -t 1655 -f 6,7)
ARMS=(none messages full)
STOP_FILE="$ROOT/queue_out/cb_queue.stop"  # touch it to stop after the current step
cd "$ROOT"
mkdir -p queue_out
rm -f "$STOP_FILE"

log() { echo "$(date '+%F %T') $*"; }
spent() { PYTHONPATH="$ROOT" python3 -m integrations.cooperbench.spend "$LOG_ROOT"; }
stop_requested() { [[ -f "$STOP_FILE" ]]; }

watchdog() {
  while sleep 120; do
    if python3 -c "import sys; sys.exit(0 if float('$(spent)') > $SPEND_CAP else 1)"; then
      log "WATCHDOG: spend \$$(spent) passed the cap \$$SPEND_CAP, stopping this experiment's containers"
      touch "$STOP_FILE"
      docker ps --filter label=cpc-cb=1 --format '{{.ID}} {{.Names}}' | awk '$2 != "cpc-compress" {print $1}' \
        | xargs -r docker kill >/dev/null
      return
    fi
  done
}

smoke_ok() {  # the smoke pair ran end to end: evaluated, ledgers with LLM calls, no compressor errors
  local dir
  dir=$(ls -d "$LOG_ROOT"/smoke_"$1"/coop/*/*/* 2>/dev/null | head -1)
  [[ -n "$dir" && -f "$dir/eval.json" ]] || { log "smoke $1: no eval.json"; return 1; }
  grep -q '"event": "llm"' "$dir"/*_ledger.jsonl 2>/dev/null || { log "smoke $1: no LLM calls in ledgers"; return 1; }
  if grep -q '"event": "compress_error"' "$dir"/*_ledger.jsonl; then log "smoke $1: compressor errors"; return 1; fi
  return 0
}

log "queue start, spend so far \$$(spent), cap \$$SPEND_CAP"
curl -sf http://127.0.0.1:8765/health >/dev/null || bash "$CB" server
watchdog &
WATCHDOG=$!

for arm in "${ARMS[@]}"; do
  stop_requested && break
  if ! smoke_ok "$arm" 2>/dev/null; then
    log "smoke $arm: running"
    bash "$CB" run "$arm" "smoke_$arm" "${SMOKE[@]}"
  fi
  smoke_ok "$arm" || { log "smoke $arm failed the sanity check; stopping before the full runs"; touch "$STOP_FILE"; }
  log "smoke $arm done, spend \$$(spent)"
done
bash "$CB" report smoke_none smoke_messages smoke_full > queue_out/cb_smoke_report.txt 2>&1

for arm in "${ARMS[@]}"; do
  stop_requested && break
  log "$SUBSET $arm: running"
  bash "$CB" run "$arm" "${SUBSET}_$arm" -s "$SUBSET"
  log "$SUBSET $arm done, spend \$$(spent)"
done

bash "$CB" report "${SUBSET}_none" "${SUBSET}_messages" "${SUBSET}_full" > queue_out/cb_report.txt 2>&1
kill "$WATCHDOG" 2>/dev/null
bash "$CB" stop
log "queue end, spend \$$(spent); report in queue_out/cb_report.txt"
