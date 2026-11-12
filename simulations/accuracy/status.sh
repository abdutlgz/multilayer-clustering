#!/usr/bin/env bash
# Progress of the Section 4.2 runs (208 settings x 100 networks).
cd "$(dirname "$0")"
OUT="${OUT:-../output/accuracy}"
echo "settings with a file: $(ls "$OUT"/setting_*.mat 2>/dev/null | wc -l) / 208"
done=$(grep -h -c " rep " logs/accuracy_*.out 2>/dev/null | awk '{s+=$1} END {print s+0}')
echo "networks finished (from logs): $done / 20800"
command -v squeue >/dev/null && squeue -u "$USER" -n accuracy -h | wc -l | xargs echo "array tasks queued or running:"
grep -l -i -E "error|killed|oom" logs/accuracy_*.out 2>/dev/null | head
