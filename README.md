# dwarfstar-mini

Minimal C + Metal LLM inference engine for Apple Silicon, inspired by
[antirez/ds4](https://github.com/antirez/ds4). Target: M4 Pro 48 GB.

Status: dense Qwen3-1.7B and Qwen3-30B-A3B MoE decode run on Metal; the 30B
MoE uses an asymmetric 14.36 GiB GGUF (Q3_K gate/up experts, Q4_K down
experts, Q8_0/F32 elsewhere). Dense and routed Q3_K/Q4_K GEMVs use optimized
Metal kernels, and `check-metal-gemv MODEL [TENSOR EXPERT_ID]` validates an
individual routed expert slice. The same asymmetric layout supports
Qwen3-Coder-30B-A3B-Instruct. The OpenAI-compatible server caps an unspecified
model context at 8,192 tokens to keep KV allocation practical; use `--ctx` to
override it. The OpenAI-compatible server supports both.
See `docs/F4b.md` and `docs/ds4-notes.md`.

Reference oracle: `../llama.cpp/build/bin` (Metal build).
Model: `models/Qwen3-1.7B-Q8_0.gguf` (not committed).

Metal GEMV kernel ports include code from llama.cpp, licensed under MIT.
