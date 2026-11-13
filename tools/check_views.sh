#!/bin/sh
# Force the GGUF mapping into several small Metal views so that tensors straddle
# view boundaries, and check that the output matches the default mapping.
set -eu
model=${1:-models/Qwen3-1.7B-Q8_0.gguf}
cap=${2:-700000000}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
prompt='The capital of France is'
./dsm -m "$model" -p "$prompt" -n 16 --temp 0 --backend metal \
    --dump-logits "$tmp/default.bin" > "$tmp/default.txt" 2> "$tmp/default.err"
DSM_MAX_BUFFER_BYTES=$cap ./dsm -m "$model" -p "$prompt" -n 16 --temp 0 --backend metal \
    --dump-logits "$tmp/capped.bin" > "$tmp/capped.txt" 2> "$tmp/capped.err"
views=$(sed -n 's/^dsm: \([0-9]*\) Metal weight views.*/\1/p' "$tmp/capped.err")
if [ -z "$views" ] || [ "$views" -lt 3 ]; then
    echo "check-views: expected at least 3 views, got '${views:-none}'" >&2
    cat "$tmp/capped.err" >&2
    exit 1
fi
cmp -s "$tmp/default.bin" "$tmp/capped.bin" || { echo "check-views: logits differ" >&2; exit 1; }
cmp -s "$tmp/default.txt" "$tmp/capped.txt" || { echo "check-views: text differs" >&2; exit 1; }
echo "check-views: $views views, logits and text identical"
