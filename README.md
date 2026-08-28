# Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS - speed test

Model: `C:\LLM\Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf` (12.63 GB, 65 hybrid
attention/MoE blocks, 1 MTP layer) on an RTX 4080 Laptop 12 GB.
The model does not fit fully in VRAM, so every config splits MoE FFN layers
between GPU and CPU; the main speed lever is where that split sits, plus
CPU threads and speculative decoding.

## The 5 new configs (in `C:\LLM`)

| File | Label | Change vs. baseline | Hypothesis | Risk |
|---|---|---|---|---|
| `Qwen3.8-27B-V1-cpu-threads16.bat` | `v1-cpu-t16` | `-t 16 -tb 16 --prio 2 --cpu-strict` | CPU-side FFN (half the model) is the bottleneck; more/faster CPU threads help | low |
| `Qwen3.8-27B-V2-gpu37-ctx32k.bat` | `v2-gpu-ctx32k` | FFN on CPU only blk 0-27 (was 0-35), ctx 32768 | 8 more FFN blocks on GPU + smaller KV cache | medium (VRAM) |
| `Qwen3.8-27B-V3-gpu45-ctx16k.bat` | `v3-gpu-ctx16k` | FFN on CPU only blk 0-19, ctx 16384 | keep as much MoE on GPU as possible | high (VRAM) |
| `Qwen3.8-27B-V4-mtp4.bat` | `v4-mtp4` | `--spec-draft-n-max 4` (was 2), ctx 32768 | deeper MTP draft chain => more accepted tokens per step | low |
| `Qwen3.8-27B-V5-ngram-prefill.bat` | `v5-ngram` | ngram-mod speculation instead of MTP, `-b 512 -ub 256` | free n-gram drafts; faster prefill | low |

Your original `Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.bat` is the **baseline**
(label `baseline`) and was not modified.

## How to benchmark

### Manual (one config at a time)
1. Close other GPU-hungry apps (browsers with hardware accel, games).
   Optional, for consistent CPU speeds:
   `powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c` (high performance).
2. Double-click one `.bat` in `C:\LLM` and wait for
   `server is listening` in its console.
3. In a **second** terminal:
   ```
   cd C:\LLM
   python bench\benchmark.py baseline        ; or v1-cpu-t16, v2-gpu-ctx32k, ...
   ```
   (options: `--runs 5 --max-tokens 512 --warmup 2`)
4. Kill the server (close its console), start the next `.bat`, repeat.

Every call appends to `bench\report.csv` and rebuilds `bench\report.md`,
which contains: model used, per-config server configuration (ctx, batch,
threads, GPU layers, KV cache, flash-attn) and generation speed
(mean/median prompt t/s + generation t/s per run).

### Fully automatic (recommended)
Double-click `bench\auto_bench_all.bat`: it loops through baseline + V1..V5
on its own (starts each server in a minimized window, waits up to 300 s for
the model to load, runs 3 measured benchmarks, kills the server, next),
reports which configs failed to start (e.g. VRAM OOM), and prints the final
comparison table. Just wait (typically ~15-30 min total) and read
`bench\report.md`.

## Notes
- Benchmark = greedy generation (`temperature 0`, `ignore_eos`) of a fixed
  prompt via the raw `/completion` endpoint, 1 warmup + 3 measured runs of
  256 tokens by default; t/s is taken from llama.cpp's own `completion_ms`.
- If V2/V3 fail to start (VRAM OOM in the server console), widen the CPU
  range by ~4 blocks or lower ctx (comments in each `.bat` say exactly what).
- `--reasoning` is left on in every `.bat` (same as baseline), but the
  benchmark uses raw completions, so reasoning tokens do not affect the
  measured speed.
