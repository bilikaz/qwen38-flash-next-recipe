# Qwen3.8-Flash-Next on one DGX Spark

**One box. 262k context. 55 tok/s single-stream, 167 tok/s aggregate at peak. Three commands.**

Serves [myllmbox/Qwen3.8-Flash-Next-hibrid46](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid46)
— a 4.6-bit mixed-precision build of Qwen's ~176B (6B active) model, engineered for a **single
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

Peaks (best clean 10-second windows, code-emission band): **single stream ~55 tok/s** with MTP
acceptance touching 4.00/4.00 — the drafter running at its theoretical ceiling — and **167.5
tok/s aggregate** at 8 concurrent streams. Sustained numbers below are what you should expect,
peaks are what the box can touch.

Structured-output workload, thinking disabled, multi-window steady-state averages:

| concurrent requests | aggregate tok/s | per-stream |
|---|---|---|
| 1 | 44 | 44 |
| 2 | 72 | 36 |
| 4 | 103 | 26 |
| 8 | 148–158 (peak window 167) | 19 |

Full 262,144-token context; ~610k pooled KV tokens at the 19G default (≈2.3 concurrent
max-length requests, or 8+ typical ones — the kit runs the model alone, so it affords a bigger
pool than the full-platform config). Thinking-on workloads run lower (band-dependent). Numbers
carry their conditions on purpose — rerun them yourself and count.

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
