#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model=${MODEL:-"$root/models/Qwen3-1.7B-Q8_0.gguf"}
oracle="$root/../llama.cpp/build/bin/llama-tokenize"
for prompt in 'Hello, world!' 'Ciao, perché è così?' 'def add(a, b): return a + b' '123456 3.14159' 'emoji: 😀 café' '<|im_start|>user
Hi<|im_end|>
'; do
	ours=$("$root/dsm" -m "$model" -p "$prompt" --tokens-only)
	ref=$("$oracle" -m "$model" -p "$prompt" --ids 2>/dev/null | tr -d '[],' | tr -s ' ' | sed 's/^ //;s/ $//')
	[ "$ours" = "$ref" ] || {
		echo "mismatch: $prompt"
		echo "  dsm:   $ours"
		echo "  llama: $ref"
		exit 1
	}
done
echo 'token comparison: 6/6 exact'
