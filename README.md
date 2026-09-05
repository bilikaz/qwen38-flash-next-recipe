# Qwen3.8-Flash-Next on one DGX Spark

**One box. 262k context. Peaks: 55 tok/s single-stream, 305 tok/s aggregate at 32 streams. Three commands.**

Serves [myllmbox/Qwen3.8-Flash-Next-hibrid46](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid46)
— a **4.35-bit-effective** mixed-precision build — 180.0B counted parameters in 91.11 GiB — of Qwen's
flagship MoE (6B active), engineered for a **single
NVIDIA DGX Spark (GB10, 119G unified memory)**. No cluster, no tensor parallelism: at 6B active
parameters one Spark is the *correct* topology, not the compromise — a second box adds
interconnect tax, not speed.

## Quick start

```bash
git clone https://github.com/bilikaz/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./run.sh        # downloads ~91G from HF on first run, serves OpenAI API on :8000
```

`./stop.sh` stops it. `./view.sh` shows live stats (throughput, KV usage, speculative-decoding
acceptance). Requirements: a DGX Spark with docker + NVIDIA container runtime. First boot reaches
healthy in ~15 minutes (weights load + one-time compile warmup); later boots are faster.

**All configuration lives in [`recipe.yaml`](recipe.yaml)** — one file: image, weights repo,
port, context length, KV budget, every vLLM flag. Nothing else to edit.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Measured performance (this exact kit, single Spark)

**Peak ladder** — the best clean 10-second engine window at each concurrency (all streams decoding, zero prefill in
the window), code-emission band, thinking disabled, the myllmbox "pasture" prompt. This is the ladder behind the
305 tok/s headline; every rung is a measured window on this checkpoint and image:

| concurrent requests | peak aggregate tok/s | per-stream at peak |
|---|---|---|
| 1 | 55.0 | 55.0 |
| 2 | 78.7 | 39.4 |
| 3 | 90.2 | 30.1 |
| 4 | 113.1 | 28.3 |
| 5 | 116.3 | 23.3 |
| 6 | 128.6 | 21.4 |
| 8 | 167.5 | 20.9 |
| 12 | 185.1 | 15.4 |
| 16 | 228.7 | 14.3 |
| 20 | 256.6 | 12.8 |
| 24 | 280.0 | 11.7 |
| 28 | 293.2 | 10.5 |
| 32 | **305.1** | 9.5 |

Single-stream peaks touch MTP acceptance 4.00/4.00 — the drafter at its theoretical ceiling for K=3.

**Sustained** — multi-window steady-state averages, same workload; what you should expect minute after minute,
peaks are what the box can touch:

| concurrent requests | aggregate tok/s | per-stream |
|---|---|---|
| 1 | 44 | 44 |
| 2 | 72 | 36 |
| 4 | 103 | 26 |
| 8 | 148–158 (peak window 167) | 19 |

To run the upper rungs yourself, `max-num-seqs` must admit that many streams (the shipped value is 12; the ladder
above was measured with 32 seats admitted). Full 262,144-token context; ~966k pooled KV tokens at the 19G default
(≈3.7 concurrent max-length requests, or 30+ typical ones — the kit runs the model alone, so it affords a bigger
pool than the full-platform config). Thinking-on workloads run lower (band-dependent; acceptance 2.3–3.0 instead of
3.4–4.0 — the engine's steps/s do not change). Numbers carry their conditions on purpose — rerun them yourself and count.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes): the KV pool. 19G default. KV must stay **bf16** on this model —
  the vendor's own QSA guard refuses fp8.
- **`max-num-seqs`**: aggregate throughput scales to 8; per-request speed is best ≤5.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits realistic
  multi-image requests; don't lower it if you send images.
- **`async-scheduling` on**: +9–12% throughput at 4–5 concurrent requests, neutral at 8.
  (An early corruption suspicion was disproved by a controlled A/B — quality variance under
  concurrent thinking-heavy prompts is the model's own, async or not.)
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v1`
(digest `sha256:92ccd7de6d80b9597844d7a5bffb9ded4067dd323129e18bc24192c00dac09ad`) — the vendor's
SM121 vLLM base plus one readable patch on `ple_layer.py`: it loads the checkpoint's pre-quantized
int3 n-gram table (declared in `config.json` as `ple_quantization`) instead of expecting the
original 95G bf16 table. The Dockerfile and full build ledger live in the myllmbox repo under
[`builds/qwen38-flash-next/`](https://github.com/bilikaz/myllmbox-runner/tree/main/builds/qwen38-flash-next)
— rebuild and diff it yourself. Upstreaming the patch to vLLM is planned; until then, stock vLLM
cannot load this checkpoint.

## The full box

This kit serves one model, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, dashboard,
keepalive and multi-model management — same image, same weights, one `./run.sh qwen38-flash-next`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT.
