#!/bin/bash
# Solar-Open2 W4AFP8 serving image entrypoint.
# Runs the boot gate against the ACTUAL launch args, then execs them.
# The gate is not optional: a cell that trips it must not serve, because every
# precondition it checks fails silently -- the engine would boot and answer.
set -euo pipefail

echo "[SOLAR-IMAGE] $(date -u +%FT%TZ) ${SOLAR_IMAGE_TAG:-<untagged>}"

# SOLAR_FSM=1 is baked in, but the FSM needs a tokenizer dir and that is
# model-path dependent. Default it to --model-path so callers cannot boot with
# the FSM half-configured; an explicit env still wins.
if [ "${SOLAR_FSM:-0}" = "1" ] && [ -z "${SOLAR_FSM_TOKENIZER_DIR:-}" ]; then
  _mp=""
  _prev=""
  for a in "$@"; do
    case "$a" in
      --model-path=*) _mp="${a#--model-path=}" ;;
    esac
    if [ "$_prev" = "--model-path" ]; then _mp="$a"; fi
    _prev="$a"
  done
  if [ -n "$_mp" ]; then
    export SOLAR_FSM_TOKENIZER_DIR="$_mp"
    echo "[SOLAR-IMAGE] SOLAR_FSM_TOKENIZER_DIR defaulted to $_mp"
  else
    echo "[SOLAR-IMAGE] WARN: SOLAR_FSM=1 but no --model-path found and" \
         "SOLAR_FSM_TOKENIZER_DIR unset" >&2
  fi
fi

python3 /opt/solar/solar_gate.py "$@"
echo "[SOLAR-IMAGE] gate passed, exec: $*"
exec "$@"
