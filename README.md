# Static-graph Gated DeltaNet for AWS Inferentia2 / Trainium

A fixed-shape rewrite of the Qwen3.5 / Qwen3.8 hybrid decoder (Gated DeltaNet + gated full attention) that compiles on
AWS Inferentia2 / Trainium1 (Neuron SDK 2.28) as one graph per inference phase, and reproduces the fp32 reference token
for token on the chip, up to **Qwen3.8-27B on Trainium1** (8 NeuronCores, 500 tok/s per copy at batch 64,
$0.30 per million tokens at Spot prices).

Explainer video: [`video/`](video/) (Remotion project; render with `cd video && npm i && npm run render`).

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
5. **Qwen3.8-27B on Trainium1** (`tp_qwen.py`, `neuron/run_tp.py`; `trn1.32xlarge`, bf16, TP=8, run
   `results/20261126T191354Z-trn1.32xlarge`):

   | | |
   |---|---|
   | greedy tokens identical to fp32 (prompts of 5 and 167 tokens) | 32/32, 32/32 |
   | KL to fp32, mean / max | 1.9e-4 / 3.4e-3 |
   | prefill on an unseen batch | KL ≤ 9.7e-5, same argmax |
   | prefill 2×256 | 117 ms |
   | decode, batch 2 / 8 / 32 / 64 | 38 / 52 / 78 / 128 ms per step = 52 / 153 / 410 / 500 tok/s |

   * Weights are sliced by head across 8 cores (9 GB each). The recurrent, conv and KV state stay on the device between
     calls (aliased NxD buffers). There are 2 all-reduces per layer and one logits all-gather.
   * On Qwen3.5-0.8B, device-resident state with TP=2 cut the batch-2 decode step from 21.7 ms to 6.3 ms.
6. **Cost per million output tokens** (`results/20261126T191354Z-trn1.32xlarge/cost.json`). A trn1.32xlarge holds 4
   copies; that figure is projected from one measured copy.

   | | batch 8 | batch 32 | batch 64 |
   |---|---|---|---|
   | trn1.32xlarge Spot ($2.15/h) | $0.97 | $0.36 | **$0.30** |
   | trn1.32xlarge on-demand ($21.50/h) | $9.73 | $3.64 | $2.99 |
   | cheapest hosted API (OpenRouter) | $1.05 | $1.05 | $1.05 |

   * Spot is 3.5x cheaper than the cheapest API at batch 64. On-demand is about 3x more expensive.
   * Caveats: contexts are ≤ 320 tokens, only output tokens are counted, and there is no continuous batching.
7. **Launcher** (`infra/inf2_spot.sh`): a one-time Spot or `MARKET=ondemand` instance uploads its results and terminates
   itself. All 0.8B runs cost about $0.20; the 27B run took 12 minutes and cost about $5 on-demand.

## Layout
| path | what |
|---|---|
| `static_qwen.py` | the static model: `prefill(ids[B,T], lengths[B])`, `decode(tok[B], pos[B], state)` |
| `test_static.py` | CPU checks vs transformers: inverse, prefill, greedy decode, export op scan |
| `make_tiny_gqa.py` | random-weight model with the Qwen3.8-27B head layout |
| `neuron/run_inf2.py` | on-device run: inverse, whole-model compile, greedy decode vs fp32, latency |
| `neuron/solve_vs_doubling.py` | on-device comparison: solve_triangular vs row loop vs doubling, HLO dump, unseen inputs |
| `paper/` | technical report (LaTeX + PDF) |
| `tp_qwen.py`, `test_tp.py` | tensor-parallel model (head-sliced) and its CPU check with emulated collectives |
| `neuron/run_tp.py` | NxD trace/compile/run at TP=N: accuracy vs fp32, latency, batch sweep |
| `infra/inf2_spot.sh` | the launcher (Spot or on-demand, any region) |
| `video/` | explainer video (Remotion + Kokoro narration; numbers from `video/src/facts.json`) |
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

## Qwen3.8-27B on Trainium1
```bash
REGION=us-east-2 SUBNETS=subnet-... SG=sg-... MARKET=ondemand ITYPE=trn1.32xlarge VOLUME_GB=300 MINUTES=150 \
SCRIPT=neuron/run_tp.py MODEL=Qwen/Qwen3.8-27B RUN_ARGS="--tp 8 --bucket 256 --steps 32 --decode-batches 8,32,64" \
./infra/inf2_spot.sh
```
Compile time grows with the unrolled `bucket/64` chunk loop in each of 48 GDN layers. Keep buckets ≤ 4096, or move the
chunk scan into an NKI kernel.

## License
Apache-2.0
