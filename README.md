# Static-graph Gated DeltaNet for AWS Inferentia2 / Trainium

A fixed-shape rewrite of the Qwen3.5 / Qwen3.8 hybrid decoder (Gated DeltaNet + gated full attention) that neuronx-cc
can compile as one graph per inference phase. Each 64-token chunk of the gated delta rule normally needs a
triangular solve, `(I + L)^-1`. neuronx-cc rejects `torch.linalg.solve_triangular` as an unsupported operator
([NCC_EVRF001]). This repo replaces it with an exact inverse that uses only matmuls.

[NCC_EVRF001]: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/compiler/error-codes/EVRF001.html

## Contributions
1. **Matmul-only exact unit-lower inverse** (`static_qwen.unit_lower_inverse`). It takes log2(C) block-doubling steps,
   `X <- X - X O_b X`, with constant masks `O_b`: 6 steps and 12 dense 64x64 matmuls for C = 64. There is no solve,
   no row loop and no large matrix power. On the real Qwen3.5-0.8B systems it has a max relative error of 2.6e-7 vs
   fp64. The Neumann product, the usual matmul-only alternative, reaches 1.3e10 error on the same systems
   (‖L‖F up to 42).
2. **Whole-model static prefill/decode graphs.**
   * Prompts are right-padded to a 64-multiple bucket. Padded tokens get `beta = 0` and `g = 0`, so they neither write
     to nor decay the recurrent state.
   * The KV cache, conv window and recurrent state are explicit fixed-shape tensors, updated with one-hot blends.
   * Prefix sums are matmuls with a constant triangular matrix.
   * Max tensor rank is 4 (NCC_INLA001 is a 5-D codegen crash).
   * There is no data-dependent control flow.
   * `torch.export` finds 43 (prefill) and 33 (decode) distinct aten ops, and no solve, linalg, nonzero or `.item()`.
3. **Exactness on CPU vs transformers** (Qwen3.5-0.8B, fp32, mixed-length batch of 5 and 167 tokens, bucket 256):
   * prefill max |Δlogit| 3.2e-5;
   * 24/24 identical greedy tokens per prompt;
   * per-step KL ≤ 3.7e-7.
   * A random-weight model with the Qwen3.8-27B head layout (48 value heads per 16 key heads, 24/4 attention heads)
     agrees the same way.
4. **On-device run** on Inferentia2 with Neuron SDK 2.28 (`neuron/run_inf2.py`):
   * a negative control showing that `solve_triangular` does not compile;
   * the inverse compiled alone on the hardest real systems;
   * both whole-model graphs compiled, with greedy decoding vs the fp32 reference, plus latency.
   * Results: `results/<run>/results.json`.
5. **Reproducible Spot recipe** (`infra/inf2_spot.sh`): a one-time Spot inf2 instance that uploads results to S3 and
   terminates itself, with a hard time budget.

## Layout
| path | what |
|---|---|
| `static_qwen.py` | the static model: `prefill(ids[B,T], lengths[B])`, `decode(tok[B], pos[B], state)` |
| `test_static.py` | CPU checks vs transformers: inverse, prefill, greedy decode, export op scan |
| `make_tiny_gqa.py` | random-weight model with the Qwen3.8-27B head layout |
| `neuron/run_inf2.py` | the on-device run (Inferentia2 / Trainium, torch-neuronx) |
| `infra/inf2_spot.sh` | the Spot launcher |
| `results/` | `results_0p8b.json`, `results_tiny_gqa.json` (CPU), plus device runs |

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
