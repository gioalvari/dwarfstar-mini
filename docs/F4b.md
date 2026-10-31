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
4.04 bpw). `tools/gguf_inspect` reports 96 Q3_K expert tensors, 48 Q4_K
expert tensors, 194 Q8_0 tensors, and 241 F32 tensors. `llama-simple` Metal produced
the coherent France continuation `Paris. The capital of the United Kingdom is
London...` and measured 64.76 decode tok/s (15 generated-token evaluations).

Metal runs the complete MoE decode on GPU: it computes the F32 router GEMV,
softmaxes all 128 routes, selects and renormalizes top-8 in one threadgroup,
then performs selected-expert GEMVs through an expert-index indirection. The
three expert tensors remain `newBufferWithBytesNoCopy` views of the GGUF mmap;
large tensors are addressed through contiguous max-buffer-length views. Q and
attention buffers use their derived widths (`heads * key_length` and
`heads * value_length`) rather than assuming either equals embedding width.

## Gates on M4 Pro (48 GB)

| Gate | Result |
| --- | --- |
| Asymmetric CPU vs Metal first prompt logits | max abs `4.20e-05`; argmax `12095` on both |
| Asymmetric greedy, France `-n 32` | DSM Metal exactly matches `llama-simple` continuation |
| Asymmetric greedy, Fibonacci `-n 32` | DSM Metal exactly matches `llama-simple` continuation |
| 1.7B Q4_K_M long prompt, `-n 128` | CPU and Metal output exactly matches |
| `make test` | passed |
| `check-metal-gemv`, Q3_K/Q4_K and expert slices | all relative errors below `1e-3` |
| `dsm serve` + one Metal curl | passed; chat completion returned |

| Model / backend | Decode tok/s |
| --- | ---: |
| 30B asymmetric, DSM Metal, France `-n 32` | 45.92 |
| 30B asymmetric, llama.cpp Metal, France | 64.41 |
| 30B asymmetric, llama.cpp Metal, Fibonacci | 60.07 |

## MoE profile and dispatch batching

With `DSM_PROFILE=1`, the asymmetric model profile now separates the router,
GPU top-k, expert gate/up, expert down, and weighted accumulation stages. A
short France run reports expert gate/up at 32.10 ms/token (16.17 GB/s) and
expert down at 25.00 ms/token (13.59 GB/s); together they remain the principal
decode cost. The profile runs each category in separate command buffers, so it
is diagnostic-only and slower than normal decode.

The normal path batches all eight selected experts in one dispatch per gate,
up, and down projection (threadgroup `z` selects the expert slot), fuses the
eight SwiGLU activations, and combines all weighted down outputs in one kernel.
It has no CPU round trip. This reduces normal asymmetric France decode from
16.92 tok/s before batching to 20.64 tok/s; Fibonacci `-n 32` measures 20.96
tok/s. The Q3_K expert GEMV remains the simple compatible kernel; the profile
identified it as the next kernel-level target.

## K-quantized expert GEMVs

The Q4_K down-expert path now reuses the dense Q4_K GEMV geometry: each
two-simdgroup threadgroup computes four output rows, with the selected
expert's byte stride applied before the row offset. Q3_K uses the faithful
llama.cpp two-row/two-simdgroup kernel for both dense matrices and routed
gate/up experts. The GEMV check tool accepts an optional tensor name and
expert id, so an individual 3D expert slice can be compared against the CPU:

```sh
./check-metal-gemv models/Qwen3-30B-A3B-asym.gguf \
  blk.0.ffn_down_exps.weight 0
```

On the M4 Pro, a profile run for the asymmetric France prompt (`-n 16`)
reports `5.387 ms/token` for Q3_K expert gate/up and `2.753 ms/token` for
Q4_K expert down. Profile categories synchronize separately and therefore
are diagnostic only. The corresponding normal France decode (`-n 32`) is
`45.92 tok/s`, faster than the preceding `20.64 tok/s` baseline; normal
decode is the acceptance metric.

The 32.48 GB Q8_0 model was not run through the final full-GPU comparison:
this M4 Pro has 48 GB unified memory, but the model is larger than the device's
recommended working set after runtime/KV allocations. It is intentionally
skipped rather than relying on memory pressure.

```sh
./dsm --backend metal -m models/Qwen3-30B-A3B-asym.gguf \
  -p 'The capital of France is' -n 8
```
