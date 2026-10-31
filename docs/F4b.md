# F4b - Qwen3-MoE Metal decode and asymmetric quantization

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

`models/Qwen3-30B-A3B-asym.gguf` is generated read-only from the Q8_0 source
with `llama-quantize --allow-requantize`: routed `ffn_gate_exps` and
`ffn_up_exps` are Q3_K; routed `ffn_down_exps` is Q4_K; all other tensors
retain their Q8_0 or F32 source types. It is 15,421,502,976 bytes (14.36 GiB,
4.04 bpw). The tensor histogram is 288 Q3_K expert tensors, 48 Q4_K expert
tensors, 147 Q8_0 tensors, and 96 F32 tensors. `llama-simple` Metal produced
the coherent France continuation `Paris. The capital of the United Kingdom is
London...` and measured 64.76 decode tok/s (15 generated-token evaluations).

Metal runs the complete MoE decode on GPU: it computes the F32 router GEMV,
softmaxes all 128 routes, selects and renormalizes top-8 in one threadgroup,
then performs selected-expert GEMVs through an expert-index indirection. The
three expert tensors remain `newBufferWithBytesNoCopy` views of the GGUF mmap;
large tensors are addressed through contiguous max-buffer-length views. Q and
attention buffers use their derived widths (`heads * key_length` and
`heads * value_length`) rather than assuming either equals embedding width.

```sh
./dsm --backend metal -m models/Qwen3-30B-A3B-asym.gguf \
  -p 'The capital of France is' -n 8
```
