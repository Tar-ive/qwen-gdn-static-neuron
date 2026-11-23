# Static-graph Gated DeltaNet for AWS Inferentia2 / Trainium

A fixed-shape rewrite of the Qwen3.5 / Qwen3.8 hybrid decoder (Gated DeltaNet + gated full attention) that compiles on
AWS Inferentia2 / Trainium1 (Neuron SDK 2.28) as one graph per inference phase, and reproduces the fp32 reference token
for token on the chip.

Paper: [`paper/static_gdn_inferentia2.pdf`](paper/static_gdn_inferentia2.pdf)

## Findings and contributions
1. **`torch.linalg.solve_triangular` fails silently on Inferentia2.**
   * With `torch_neuronx.trace` (SDK 2.28), torch-xla has no device lowering for it. It runs on the host, and its
     *output* is frozen into the NEFF as a constant (the HLO is an identity on one parameter).
   * It compiles without error and is exact on the tracing input. On any other input it is wrong: median relative
     error 0.56, max 1.44.
   * Inside the whole model it compiles and then segfaults. Tests that evaluate on the tracing input cannot see this.
2. **Exact matmul-only inverse** (`static_qwen.unit_lower_inverse`).
   * log2(64) = 6 block-doubling steps, `X <- X - X O_b X`, with constant masks.
   * On device, on 256 real systems it was not traced with (‖L‖F up to 42), the max error vs fp64 is 1.6e-7.
   * The transformers row-loop fallback, the other formulation that generalises, is 2.9x slower per call
     (4.05 vs 1.41 ms), 5.1x slower to compile and 22x larger in HLO.
   * The Neumann product reaches 1.3e10 error on the same systems.
3. **One static graph per phase for the whole hybrid model.**
   * Padding with `beta = 0` and `g = 0` provably leaves the state unchanged.
   * State is explicit and fixed-shape; max tensor rank is 4; prefix sums are matmuls; there is no data-dependent
     control flow.
4. **Results.**
   * CPU fp32 vs transformers: max |Δlogit| 3.2e-5, 24/24 greedy tokens.
   * **Inferentia2**, Qwen3.5-0.8B, bf16 weights:
     * 32/32 greedy tokens identical to fp32 on both prompts, KL ≤ 7.3e-4;
     * prefill on a batch it was not traced with: KL ≤ 5.8e-4, same argmax;
     * prefill 2x256 in 41-52 ms; batch-2 decode step 21.7 ms, including a 51 MB state round-trip.
5. **Spot recipe** (`infra/inf2_spot.sh`): a one-time Spot instance uploads its results and terminates itself. The three
   device runs cost about $0.20 in total.

## Layout
| path | what |
|---|---|
| `static_qwen.py` | the static model: `prefill(ids[B,T], lengths[B])`, `decode(tok[B], pos[B], state)` |
| `test_static.py` | CPU checks vs transformers: inverse, prefill, greedy decode, export op scan |
| `make_tiny_gqa.py` | random-weight model with the Qwen3.8-27B head layout |
| `neuron/run_inf2.py` | on-device run: inverse, whole-model compile, greedy decode vs fp32, latency |
| `neuron/solve_vs_doubling.py` | on-device comparison: solve_triangular vs row loop vs doubling, HLO dump, unseen inputs |
| `paper/` | technical report (LaTeX + PDF) |
| `infra/inf2_spot.sh` | the Spot launcher |
| `results/` | `results_0p8b.json`, `results_tiny_gqa.json` (CPU); `2026112*-inf2.*/` device runs |

## Reproduce
```bash
# CPU (any machine)
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch transformers safetensors huggingface_hub
.venv/bin/hf download Qwen/Qwen3.5-0.8B --local-dir ckpt/Qwen3.5-0.8B
.venv/bin/python test_static.py ckpt/Qwen3.5-0.8B --json results/results_0p8b.json

# Inferentia2 (Neuron 2.28 DLAMI, the last SDK that supports Inf2/Trn1 with NxDI); on the instance:
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/python neuron/run_inf2.py ckpt/Qwen3.5-0.8B out/
# or from your laptop, on Spot:
cp infra/env.example infra/.env && $EDITOR infra/.env && ./infra/inf2_spot.sh && ./infra/inf2_spot.sh fetch
```
For a pinned Neuron 2.28 environment off the DLAMI, see
[neuron-2.28-trn1-inf2](https://github.com/Tar-ive/neuron-2.28-trn1-inf2).

## Scaling to Qwen3.8-27B
54 GB in bf16 needs tensor parallelism, e.g. TP=8 on inf2.24xlarge (4 chips, 128 GB). TP=8 divides the 16 GDN key heads
(2 key and 6 value heads per core) and the 24 query heads, and replicates the 4 KV heads 2x. GDN heads are independent,
so the recurrence adds no collectives. Compile time grows with the unrolled `bucket/64` chunk loop in each of 48 GDN
layers: keep buckets ≤ 4096 or move the chunk scan into an NKI kernel.

## License
Apache-2.0
