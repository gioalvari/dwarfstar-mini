# F4b - Qwen3-MoE CPU forward

F4b adds a CPU decode path for `qwen3moe`, including Qwen3-30B-A3B. The
loader reads the MoE expert metadata and checks the router plus 3D expert
weight tensors for every layer.

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
