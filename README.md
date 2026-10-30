# dwarfstar-mini

Minimal C + Metal LLM inference engine for Apple Silicon, inspired by
[antirez/ds4](https://github.com/antirez/ds4). Target: M4 Pro 48 GB.

Roadmap: dense Qwen3-1.7B (CPU reference -> Metal) -> MoE Qwen3-30B-A3B with
asymmetric quantization -> OpenAI-compatible server. See `docs/ds4-notes.md`.

Reference oracle: `../llama.cpp/build/bin` (Metal build).
Model: `models/Qwen3-1.7B-Q8_0.gguf` (not committed).
