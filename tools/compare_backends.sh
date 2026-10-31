#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model=${MODEL:-"$root/models/Qwen3-1.7B-Q8_0.gguf"}
oracle="$root/../llama.cpp/build/bin/llama-simple"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

compare() {
	label=$1 prompt=$2 count=$3
	"$root/dsm" -m "$model" -p "$prompt" -n "$count" --backend cpu >"$tmp/cpu-$label" 2>"$tmp/cpu-$label.err"
	"$root/dsm" -m "$model" -p "$prompt" -n "$count" --backend metal >"$tmp/metal-$label" 2>"$tmp/metal-$label.err"
	cmp -s "$tmp/cpu-$label" "$tmp/metal-$label" || {
		echo "$label: CPU/Metal continuation mismatch"
		return 1
	}
	"$oracle" -m "$model" -n "$count" "$prompt" 2>/dev/null >"$tmp/llama-$label" || true
	python3 - "$prompt" "$tmp/metal-$label" "$tmp/llama-$label" <<'PY'
import sys

prompt, dsm_path, llama_path = sys.argv[1:]
dsm = open(dsm_path).read()
llama = open(llama_path).read()
if llama.startswith(prompt):
    llama = llama[len(prompt):]
if dsm.rstrip() != llama.rstrip():
    raise SystemExit("llama-simple continuation mismatch")
PY
	echo "$label: CPU/Metal/llama-simple exact ($count tokens)"
	grep 'dsm:' "$tmp/metal-$label.err"
}

compare france 'The capital of France is' 16
compare fibonacci 'def fibonacci(n):' 32
compare long 'Explain in concise terms how causal self-attention uses queries, keys, values, masking, residual connections, normalization, and feed-forward layers while generating one next token at a time from this deliberately long prompt with enough words to exercise a fuller prefix cache and a larger attention sweep across many preceding tokens for the decoder.' 64

"$root/dsm" -m "$model" -p Hi -n 0 --backend cpu --dump-logits "$tmp/cpu.bin" >/dev/null
"$root/dsm" -m "$model" -p Hi -n 0 --backend metal --dump-logits "$tmp/metal.bin" >/dev/null
python3 - "$tmp/cpu.bin" "$tmp/metal.bin" <<'PY'
import array
import sys

cpu = array.array("f")
metal = array.array("f")
cpu.frombytes(open(sys.argv[1], "rb").read())
metal.frombytes(open(sys.argv[2], "rb").read())
print(f"first-step logits: max_abs_diff={max(abs(a-b) for a, b in zip(cpu, metal)):.8g}")
PY
