#!/usr/bin/env bash
# run_network.sh ID "REPS" "METHODS" OUT
# For each network r in REPS of setting ID: generate it (not timed), run each method of
# METHODS in a fresh single-thread MATLAB process under /usr/bin/time (GNU time, Linux),
# append peak memory, wall time, host and CPU model, then delete the network.
# One CSV line per (network, method) goes to OUT/raw/s<ID>_r<r>_<method>.csv;
# runs that already have a line are skipped, so a stopped job can be resubmitted.
set -euo pipefail
ID=$1; REPS=$2; METHODS=$3; OUT=$4
NET="$OUT/networks"; RAW="$OUT/raw"; mkdir -p "$NET" "$RAW"
WORK="${SLURM_TMPDIR:-/tmp}/bench_${SLURM_JOB_ID:-local}_${ID}_$$"; mkdir -p "$WORK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
CPU=$(LC_ALL=C lscpu | sed -n 's/^Model name:[[:space:]]*//p' | head -1)
echo "node $(hostname) cpu: $CPU"
for r in $REPS; do
  todo=""
  for m in $METHODS; do
    [ -s "$(printf "%s/s%02d_r%02d_%s.csv" "$RAW" "$ID" "$r" "$m")" ] || todo="$todo $m"
  done
  [ -z "$todo" ] && continue
  f=$(printf "%s/s%02d_r%02d.mat" "$NET" "$ID" "$r")
  export MATLAB_PREFDIR="$WORK/prefs_gen_$r"; mkdir -p "$MATLAB_PREFDIR"
  matlab -singleCompThread -batch "make_network_one($ID,$r,'$NET')"
  for m in $todo; do
    out=$(printf "%s/s%02d_r%02d_%s.csv" "$RAW" "$ID" "$r" "$m")
    export MATLAB_PREFDIR="$WORK/prefs_${r}_${m}"; mkdir -p "$MATLAB_PREFDIR"
    /usr/bin/time -f '%M|%e' -o "$WORK/time.txt" \
      matlab -singleCompThread -batch "bench_one('$f','$m','$WORK/res.csv')" > "$WORK/matlab.log" 2>&1 \
      || { echo "FAILED setting $ID rep $r $m"; tail -40 "$WORK/matlab.log"; exit 1; }
    IFS='|' read -r rss wall < <(tail -1 "$WORK/time.txt")
    echo "$(cat "$WORK/res.csv"),$rss,$wall,$(hostname),\"$CPU\"" > "$out"
    echo "setting $ID rep $r $m: $(cat "$out")"
  done
  rm -f "$f"
done
rm -rf "$WORK"
echo "DONE setting $ID"
