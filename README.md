# Qwen3.8-Flash-Next on one DGX Spark

**One box. 262k context. 50 tok/s sustained single-stream on code (peaks 55), 124 tok/s at 4 streams. Three commands.**

**v2 (2026-09-07)** serves [myllmbox/Qwen3.8-Flash-Next-hibrid47](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid47)
— the same checkpoint the [2-Spark kit](https://github.com/bilikaz/qwen38-flash-next-cluster-recipe) serves: 180.0B counted
parameters of Qwen's flagship MoE (6B active), the body at 4.35 bits effective, the 95 GB n-gram (PLE) table
re-quantized to **NVFP4** (26.9 GiB) — on a **single NVIDIA DGX Spark (GB10, 119G unified memory)**. Against v1 (the same
body with an int3 table in a CPU worker): **+15 % sustained single-stream (44 → 50–51 tok/s), +5 % engine steps, and the
table that draws 26 of 32 boss scenes where int3 drew half** (measured on the cluster, same checkpoint, same table).
v1 stays available: `git checkout v1` (image `…-vllm:v1`, checkpoint hibrid46).

How 99 GB of weights serve on a 119 GB box: the table is **never allocated**. Its 8 shard files are memory-mapped and the
GPU gathers rows straight out of the mapping (unified memory). The boot holds 73 GB of weights, so vLLM's autotune has the
room it needs; once the engine is warm the whole table is pulled into memory in one pass and stays there. KV is **fp8**
(391,943 tokens on a 7 GB pin). No CPU worker, no per-step detour, nothing to configure.

## Quick start

```bash
git clone https://github.com/bilikaz/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./run.sh        # downloads ~99G from HF on first run, serves OpenAI API on :8000
```

`./stop.sh` stops it. `./view.sh` shows live stats (throughput, KV usage, speculative-decoding acceptance).
`./ple.sh status` shows the table (rows in memory, free, swap); `./ple.sh populate` re-pulls it. Requirements: a DGX
Spark with docker + NVIDIA container runtime. First boot reaches healthy in ~12 minutes (weights load ~11 min + one-time
compile warmup); later boots are faster. Swap on the box is a plus, not a must (see Memory).

**All configuration lives in [`recipe.yaml`](recipe.yaml)** — one file: image, weights repo, port, context length,
KV budget, every vLLM flag. Nothing else to edit.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

## Measured performance (this exact kit, single Spark, K=3, `vm.compaction_proactiveness=0`)

Boot 2026-09-07, myllmbox "pasture" prompt, 10-second engine windows (all streams decoding, zero prefill in the window);
**sustained** = the run average, **peak** = the best window. v1 numbers from the same table in this README's `v1` tag.

| concurrent requests | v1 sustained | **v2 sustained** | v2 peak | engine steps/s (v1 → v2) | acceptance |
|---|---|---|---|---|---|
| 1 · code (thinking off) | 44 | **50–51** (12 runs, 49.0–51.1) | 54.6 | 13.8 → **14.4** (14.1–14.5) | ~3.5 |
| 1 · thinking on, full 30k-token request | — | **39–42** (4 runs) | 52–56 | **14.4** | 2.9 (2.5 reasoning → 3.5–3.9 code) |
| 4 · code | 103 | **124** | 137 | — → **8.9** | — |

Reading it: the engine runs at a constant 70 ms per step on one GPU — a 12-minute thinking-on request held 14.0–14.6
steps/s the whole way; the text decides how many draft tokens each step yields (reasoning prose ~2.5, the written
answer ~3.5). Full 262,144-token context; the 7 GB fp8 pool holds 391,943 tokens (1.5 max-length requests, or 8
typical ones — ~1 GB of the pool per running request is the model's fixed GDN state). NVMe is idle while decoding
(0.4 reads/s) once the table is in memory. Numbers carry their conditions on purpose — rerun them and count.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver, the page cache and the kernel share one pool. This serve fills it on purpose:
85 GB GPU process (73 weights + 8 KV + graphs) + 27 GB table + 7 GB of host processes ≈ the box. The kit never asks
for your password; it stabilises memory with what a user may do:

- **waits** after removing the old container until the box reports ≥ 100 GB available (unified memory takes 30–60 s to
  come back after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the image's loader **drops each shard from the cache as soon as it has been consumed**,
- the table is read with **direct NVMe reads during boot** (no page-cache growth while autotune needs the room) and
  memory-mapped from then on; **one populate pass** right after warm-up pulls all 26.9 GiB in (~25 s).

What to expect after that pass: `free -g` shows ~27 GB of the table under buff/cache and 4–8 GB free. The kernel keeps a
few GB free but fragmented, so the pass may park ~6 GB of cold process heaps in **swap** (once; swap-ins stay ≈ 0 and
the step rate does not move). Without swap it evicts a few GB of table rows instead — they come back on demand (a GPU
page fault ≈ 200 µs, ~1 % of lookups on a cold stretch). `./ple.sh populate` repeats the pass any time; `./ple.sh status`
shows where things stand.

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0; shows the command, asks, then sudo prompts; persisted
```

`run.sh` reads the value before every launch (no privilege needed) and warns while it is not 0; it never applies the
change itself. The kernel's background page compactor migrates pages to build large contiguous blocks; on a Spark the
GPU's memory *is* those pages, so every migration first unmaps them from the GPU — measured as a 4–5 s slowdown every
~37 s. A serving box allocates once at boot and gains nothing from the upkeep.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes): 7 GB fp8 = 391,943 tokens. The table and the pool are the same memory: a bigger pin
  means fewer table rows resident (more NVMe re-reads), not a faster serve. `kv-cache-dtype: fp8` — drop the line for
  bf16 (217,808 tokens on the same pin).
- **`max-num-seqs`** 8: ~1 GB of pool per running request regardless of length; 4 leaves ~130k tokens of context each.
- **`MBX_PLE_MMAP_PREWARM`** (env): `auto` = populate the whole table after boot; `0` = fill on demand only (the hot set
  of a workload is ~19 GB); `<seconds>` = populate that long after boot.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits one max-resolution image (~4.1k tokens)
  with room; don't lower it if you send images.
- **`async-scheduling` on**: +9–12 % throughput at 4–5 concurrent requests, neutral at 8.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v2` — the v1 image (vendor SM121 vLLM + the int3 table loader) plus **six readable
patches** on the vendor's `ple_layer.py` / QSA attention:

1. the NVFP4 table as a GPU parameter (the cluster kit's path; here its parameters are zero-sized),
2. the loader drops each shard's pages from the page cache as soon as its tensors are consumed,
3. the int3/offload fallback gate (inert unless `MBX_PLE_MULTINODE` is set),
4. **demand-paged table** (`MBX_PLE_MMAP=1`): the 8 shard files are mmapped, a DLPack capsule over the mapping gives the
   GPU a tensor to gather from (GB10: `pageableMemoryAccess`, `usesHostPageTables`),
5. **dual gather path** (`vllm::mbx_ple_gather`): direct NVMe reads during boot, the mapping afterwards
   (`MBX_PLE_MMAP_MODE=auto`); populate-after-boot (`MBX_PLE_MMAP_PREWARM`); flag files under `cache/` for `ple.sh`,
6. **fp8 KV on the QSA path** — upstream vLLM PR #54846 ported; the PR's 13 tests pass on the GB10.

Each patch refuses to apply twice and fails the build if its anchor moved. The Dockerfile, the patch scripts, the
tests and the build ledger live in the myllmbox repo under
[`builds/qwen38-flash-next/solo/`](https://github.com/bilikaz/myllmbox-runner/tree/main/builds/qwen38-flash-next/solo)
— rebuild and diff it yourself. Digest: `sha256:b2f35cd81998f4d58ef4266792282f48309e0d1bbe19f4816dc2f98684e0a3ec`.

## The full box

This kit serves one model, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, keepalive and multi-model
management — same image, same weights, one `./run.sh qwen38-flash-next`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT.
