#!/usr/bin/env bash
# Progress of the Section 4.3 runs: finished (network, method) runs per setting.
cd "$(dirname "$0")"
for part in runtime active_sweep; do
  RAW="../output/$part/raw"
  echo "$part: $(ls "$RAW"/*.csv 2>/dev/null | wc -l) runs"
  for id in $(tail -n +2 settings.csv | cut -d, -f1); do
    k=$(ls "$RAW"/$(printf "s%02d" "$id")_*.csv 2>/dev/null | wc -l)
    [ "$k" -gt 0 ] && printf "  setting %s: %s\n" "$id" "$k"
  done
done
echo "expected: runtime 40 per setting (10 networks x 4 methods); active_sweep 120 (10 x 12)"
command -v squeue >/dev/null && squeue -u "$USER" -n runtime,sweep -h -o "%i %j %T %M %R"
grep -h "FAILED" logs/*.out 2>/dev/null | head
