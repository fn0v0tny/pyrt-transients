#!/bin/bash
# Keep the cross-night transient page current: run tools/cross_night_transients.py
# every REFRESH_S seconds (default 6 hours). New frames land all night and the
# page only changes when candidates.tbl files do, so a few runs a day suffice;
# a run after the first night's frames costs a couple of seconds per new
# observation (its frames are read once for the detection cells).
#
# Started with: setsid nohup <checkout>/tools/cross_night_refresher.sh &
# It survives logout but not a reboot. One instance at a time (flock).
#
# Environment: PYRT_STATUS_DATA_DIR / PYRT_STATUS_PUBLIC_DIR as for the status
# page; CROSS_NIGHT_ARGS for extra options (e.g. "--days 60").
set -u
REFRESH_S=${REFRESH_S:-21600}
HERE=$(cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-python3}
LOCK=${LOCK:-~/tmp/cross_night_refresher.lock}
LOG=${LOG:-~/logs/cross_night_transients.log}
mkdir -p "$(dirname "$LOCK")" "$(dirname "$LOG")"
exec 9>"$LOCK"
flock -n 9 || { echo "$(date -u +%FT%TZ) another refresher is running" >> "$LOG"; exit 0; }
echo "$(date -u +%FT%TZ) refresher started (every ${REFRESH_S}s, pid $$, $HERE)" >> "$LOG"
while true; do
  echo "$(date -u +%FT%TZ) run" >> "$LOG"
  nice -n 15 ionice -c2 -n7 "$PYTHON" "$HERE/cross_night_transients.py" ${CROSS_NIGHT_ARGS:-} >> "$LOG" 2>&1 \
    || echo "$(date -u +%FT%TZ) run failed" >> "$LOG"
  sleep "$REFRESH_S"
done
