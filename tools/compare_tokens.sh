#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model=${MODEL:-"$root/models/Qwen3-1.7B-Q8_0.gguf"}
oracle=${ORACLE:-/opt/homebrew/bin/llama-tokenize}
if [ ! -x "$oracle" ]; then
	echo "oracle is not executable: $oracle" >&2
	exit 2
fi
count=0
while IFS= read -r prompt || [ -n "$prompt" ]; do
	count=$((count + 1))
	ours=$("$root/dsm" -m "$model" -p "$prompt" --tokens-only)
	ref=$("$oracle" -m "$model" -p "$prompt" --ids --log-disable 2>/dev/null | tr -d '[],' | tr -s ' ' | sed 's/^ //;s/ $//')
	[ "$ours" = "$ref" ] || {
		echo "mismatch: $prompt"
		echo "  dsm:   $ours"
		echo "  llama: $ref"
		exit 1
	}
done <<'PROMPTS'
Hello, world!
Ciao, perché è così?
café e naïve
你好，世界。
こんにちは、世界。
emoji: 😀👩🏽‍💻✨
12345678901234567890 3.1415926535
def add(a, b): return a + b
spaces   and tabs
— “quotes”… [brackets] {json}
we're I'LL don't
ＡＢＣ１２３！
Łódź, Ελληνικά
Привет, мир!
á̧̈
foo​bar baz
<think>reason</think>
<tool_call>reason</tool_call>
<|im_start|>user Hi<|im_end|>
line one
line two
PROMPTS
echo "token comparison: $count/$count exact"
