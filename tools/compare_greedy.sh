#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model=${MODEL:-"$root/models/Qwen3-1.7B-Q8_0.gguf"}
oracle="$root/../llama.cpp/build/bin/llama-simple"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
i=0
for prompt in 'The capital of Italy is' 'Write one line of Python that prints hello:' '2 + 2 ='; do
	"$root/dsm" -m "$model" -p "$prompt" -n 16 >"$tmp/dsm-$i" 2>"$tmp/dsm-$i.err"
	"$oracle" -m "$model" -n 16 -ngl 0 "$prompt" >"$tmp/llama-$i" 2>/dev/null || true
	echo "prompt $((i + 1)): dsm=$(wc -c <"$tmp/dsm-$i") bytes; oracle=$(wc -c <"$tmp/llama-$i") bytes"
	i=$((i + 1))
done
echo 'greedy outputs captured; inspect manually because llama-simple output banners vary by build'
