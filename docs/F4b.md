# F4b - Qwen3-MoE CPU forward

F4b adds a CPU decode path for `qwen3moe`, including Qwen3-30B-A3B. The
loader reads the MoE expert metadata and checks the router plus 3D expert
weight tensors for every layer.

The production Qwen3-30B-A3B Q8_0 GGUF has 48 layers, `embedding_length`
2048, 32 query heads with 128-dimensional keys/values (4096-wide Q, 512-wide
K/V), 4 KV heads, 128 experts, top-8 routing, and 768-wide expert FFNs. The
loader therefore derives the Q and attention-output widths from head counts
rather than requiring `embedding_length == head_count * key_length`.

For each token and layer, DSM computes the router softmax, selects the top 8
of 128 experts, renormalizes their weights, and evaluates only those experts.
Each selected expert uses SwiGLU followed by its own down projection. Qwen3
attention remains QK-normalized GQA as in the dense path.

CPU GEMVs split independent output rows between pthread workers. The default
is eight workers and `--threads N` selects a different count. Because a worker
owns complete rows, results are invariant to the thread count.

Metal does not implement MoE yet. Selecting Metal for a Qwen3-MoE model exits
with `MoE not yet supported on Metal, use --backend cpu`.

```sh
./dsm --backend cpu --threads 8 -m models/Qwen3-30B-A3B-Q8_0.gguf \
  -p 'The capital of France is' -n 8
```

On the local host, the exact greedy France continuation at 8 generated tokens
measured 1.81 tok/s with eight workers and 3.06 tok/s with ten workers. The
16-token Fibonacci continuation also matched llama.cpp exactly at eight
workers (2.40 tok/s).
