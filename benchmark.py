#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark.py - mini-benchmark for the llama.cpp server with VRAM/RAM monitoring.

Workflow:
  1. Start ONE of the server .bat scripts (baseline or V1..V5).
  2. In a second terminal, run:
        python bench\\benchmark.py <config-label>
     e.g.  python bench\\benchmark.py baseline
           python bench\\benchmark.py v2-gpu-ctx32k --runs 5
  3. Repeat for every config. Each call appends rows to
     bench\\report.csv and rebuilds bench\\report.md, so after
     benchmarking all configs you have one comparison report with
     model used, server configuration, VRAM/RAM usage, and generation speed.

The benchmark uses the raw /completion endpoint (greedy, fixed
prompt, ignore_eos) so the measured generation t/s is comparable
across configs and not affected by chat templates/reasoning.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_CSV = os.path.join(HERE, "report.csv")
REPORT_MD = os.path.join(HERE, "report.md")

DEFAULT_PROMPT = (
    "Explain how an out-of-order superscalar CPU executes instructions, "
    "step by step: fetch, decode, rename, dispatch, execution and "
    "retirement. Include a short concrete example and list the main "
    "hazards (structural, data, control) that the pipeline must handle "
    "and how modern designs (register renaming, branch prediction, "
    "out-of-order window) mitigate each of them."
)

CSV_FIELDS = [
    "timestamp",
    "config",
    "model",
    "n_ctx",
    "n_batch",
    "n_threads",
    "n_threads_batch",
    "n_gpu_layers",
    "cache_type_k",
    "cache_type_v",
    "flash_attn",
    "vram_used_gb",
    "vram_total_gb",
    "llama_ram_gb",
    "sys_ram_used_gb",
    "sys_ram_total_gb",
    "prompt_tokens",
    "prompt_tps",
    "gen_tokens",
    "gen_tps",
    "client_tps",
    "wall_s",
]

PROPS_FIELDS = [
    ("n_ctx", "n_ctx"),
    ("n_batch", "n_batch"),
    ("n_threads", "n_threads"),
    ("n_threads_batch", "n_threads_batch"),
    ("n_gpu_layers", "n_gpu_layers"),
    ("cache_type_k", "cache_type_k"),
    ("cache_type_v", "cache_type_v"),
    ("flash_attn", "flash_attn"),
]


def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def wait_ready(base, timeout_s):
    """Poll /health until the server is up and the model is loaded."""
    deadline = time.time() + timeout_s
    last = "no attempt yet"
    while time.time() < deadline:
        try:
            j = get_json(base + "/health")
            status = j.get("status", "")
            if status == "ok":
                return True
            last = "loading (%s)" % status
        except urllib.error.HTTPError as e:
            last = "HTTP %s" % e.code
        except Exception as e:  # connection refused, etc.
            last = str(e)
        time.sleep(2)
    return False


def read_model(base):
    try:
        j = get_json(base + "/v1/models")
        data = j.get("data") or []
        return data[0].get("id", "unknown") if data else "unknown"
    except Exception:
        return "unknown"


def read_props(base):
    """Attempt to extract server properties from /props, /slots or defaults."""
    props = {}
    try:
        j = get_json(base + "/props") or {}
        props.update(j)
        if "default_generation_settings" in j:
            props.update(j["default_generation_settings"])
    except Exception:
        pass

    try:
        slots = get_json(base + "/slots") or []
        if isinstance(slots, list) and len(slots) > 0:
            props.update(slots[0])
    except Exception:
        pass

    return props


def get_vram_usage_gb():
    """Query NVIDIA GPU memory used and total in GB using nvidia-smi."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        used_gb = round(float(parts[0]) / 1024.0, 2)
        total_gb = round(float(parts[1]) / 1024.0, 2)
        return used_gb, total_gb
    except Exception:
        return None, None


def get_llama_ram_gb():
    """Query memory used specifically by llama-server.exe process (in GB)."""
    try:
        res = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        total_kb = 0
        for line in res.stdout.strip().splitlines():
            parts = [p.strip('"\r') for p in line.split('","')]
            if len(parts) >= 5 and "llama-server.exe" in parts[0].lower():
                mem_clean = parts[4].replace(" K", "").replace(",", "").replace(".", "").replace(" ", "")
                total_kb += int(mem_clean)
        if total_kb > 0:
            return round(total_kb / (1024.0 * 1024.0), 2)
    except Exception:
        pass
    return None


def get_system_ram_gb():
    """Query system RAM (used, total) in GB via Windows API."""
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        mem = MEMORYSTATUSEX()
        mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            total_gb = round(mem.ullTotalPhys / (1024.0 ** 3), 2)
            used_gb = round((mem.ullTotalPhys - mem.ullAvailPhys) / (1024.0 ** 3), 2)
            return used_gb, total_gb
    except Exception:
        pass
    return None, None


def measure(base, prompt, n_tokens):
    payload = {
        "prompt": prompt,
        "n_predict": n_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "ignore_eos": True,
        "stream": False,
        "cache_prompt": False,
    }
    t0 = time.perf_counter()
    j = post_json(base + "/completion", payload)
    wall = time.perf_counter() - t0

    # Modern llama.cpp servers structure metrics inside 'timings' dict:
    timings = j.get("timings") or {}

    # Prompt processing (prefill)
    pn = timings.get("prompt_n") or j.get("tokens_evaluated") or j.get("prompt_n") or 0
    pms = timings.get("prompt_ms") or j.get("prompt_ms") or 0.0
    prompt_tps = timings.get("prompt_per_second")
    if prompt_tps is None:
        prompt_tps = (pn / (pms / 1000.0)) if pn and pms > 0 else 0.0

    # Generation (decode)
    cn = timings.get("predicted_n") or j.get("tokens_predicted") or j.get("completion_n") or 0
    cms = timings.get("predicted_ms") or j.get("completion_ms") or 0.0
    gen_tps = timings.get("predicted_per_second")
    if gen_tps is None:
        if cn and cms > 0:
            gen_tps = cn / (cms / 1000.0)
        elif cn and wall > 0:
            gen_tps = cn / wall
        else:
            gen_tps = 0.0

    # Client-side end-to-end throughput (tokens / total wall time)
    client_tps = (cn / wall) if cn and wall > 0 else 0.0

    return {
        "prompt_tokens": int(pn),
        "prompt_tps": round(float(prompt_tps), 1),
        "gen_tokens": int(cn),
        "gen_tps": round(float(gen_tps), 2),
        "client_tps": round(float(client_tps), 2),
        "wall_s": round(float(wall), 2),
        "truncated": j.get("truncated", False),
    }


def props_values(props):
    return [props.get(k) for k, _ in PROPS_FIELDS]


def append_rows(rows):
    new = not os.path.exists(REPORT_CSV)
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def rebuild_md(all_rows):
    # group by config, keep first-seen order
    order = []
    groups = {}
    for r in all_rows:
        c = r["config"]
        if c not in groups:
            groups[c] = []
            order.append(c)
        groups[c].append(r)

    def cfg_summary(rows):
        g = [float(x["gen_tps"]) for x in rows if x.get("gen_tps") not in ("", None)]
        p = [float(x["prompt_tps"]) for x in rows if x.get("prompt_tps") not in ("", None)]
        mean = statistics.fmean(g) if g else 0.0
        med = statistics.median(g) if g else 0.0
        pmean = statistics.fmean(p) if p else 0.0
        tok = rows[-1].get("gen_tokens", 0)
        vram = rows[-1].get("vram_used_gb", "-")
        lram = rows[-1].get("llama_ram_gb", "-")
        return mean, med, pmean, tok, vram, lram

    lines = []
    lines.append("# llama.cpp benchmark & speed report")
    lines.append("")
    lines.append("- Server: `127.0.0.1:8080`")
    lines.append("- Model (latest run): `%s`" % (all_rows[-1]["model"] if all_rows else "?"))
    lines.append("- Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("- **Métricas explicadas**:")
    lines.append("  - **Gen t/s (Decode Speed)**: Velocidad de generación pura de texto token a token (lo que percibes escribiendo).")
    lines.append("  - **Prompt t/s (Prefill Speed)**: Velocidad de ingestión/procesamiento del prompt inicial en paralelo.")
    lines.append("  - **VRAM**: Memoria de GPU ocupada.")
    lines.append("  - **LLM RAM**: Memoria RAM del sistema consumida directamente por `llama-server.exe`.")
    lines.append("")
    lines.append("## Resumen de Rendimiento y Memoria (Promedios por Configuración)")
    lines.append("")
    lines.append("| Config | Runs | Gen t/s (Media) | Gen t/s (Mediana) | Prompt t/s (Prefill) | VRAM Usada | RAM LLM Server | Tokens/run |")
    lines.append("|---|---|---|---|---|---|---|---|")
    summary_rows = []
    for c in order:
        mean, med, pmean, tok, vram, lram = cfg_summary(groups[c])
        summary_rows.append((mean, c, len(groups[c]), med, pmean, tok, vram, lram))
    for mean, c, n, med, pmean, tok, vram, lram in sorted(summary_rows, reverse=True):
        vram_str = f"{vram} GB" if vram and vram != "-" else "-"
        lram_str = f"{lram} GB" if lram and lram != "-" else "-"
        lines.append("| %s | %d | **%.2f** | %.2f | %.1f | %s | %s | %s |" % (c, n, mean, med, pmean, vram_str, lram_str, tok))
    lines.append("")
    lines.append("## Parámetros del Servidor (Última ejecución por config)")
    lines.append("")
    lines.append("| Config | ctx | batch | threads | threads(batch) | GPU layers | KV cache K | KV cache V | flash-attn |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in order:
        r = groups[c][-1]
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                c,
                r.get("n_ctx") or "-",
                r.get("n_batch") or "-",
                r.get("n_threads") or "-",
                r.get("n_threads_batch") or "-",
                r.get("n_gpu_layers") or "-",
                r.get("cache_type_k") or "-",
                r.get("cache_type_v") or "-",
                r.get("flash_attn") or "-",
            )
        )
    lines.append("")
    lines.append("## Registro Detallado de Todas las Ejecuciones")
    lines.append("")
    lines.append("| Timestamp | Config | Gen t/s | Prompt t/s | VRAM (GB) | RAM LLM (GB) | RAM Total (GB) | Gen Tokens | Wall Time (s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in all_rows:
        vram_val = r.get("vram_used_gb", "-") or "-"
        lram_val = r.get("llama_ram_gb", "-") or "-"
        sram_val = f"{r.get('sys_ram_used_gb', '-')}/{r.get('sys_ram_total_gb', '-')}" if r.get('sys_ram_used_gb') else "-"
        lines.append(
            "| %s | %s | **%s** | %s | %s | %s | %s | %s | %s |"
            % (
                r.get("timestamp", ""),
                r.get("config", ""),
                r.get("gen_tps", "-"),
                r.get("prompt_tps", "-"),
                vram_val,
                lram_val,
                sram_val,
                r.get("gen_tokens", "-"),
                r.get("wall_s", "-"),
            )
        )
    lines.append("")
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_all_rows():
    if not os.path.exists(REPORT_CSV):
        return []
    with open(REPORT_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="llama.cpp server mini-benchmark with memory monitoring")
    ap.add_argument("config", nargs="?", default="baseline",
                    help="label of the config being benchmarked (e.g. baseline, v1-cpu-t16)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--runs", type=int, default=3, help="measured runs (default 3)")
    ap.add_argument("--warmup", type=int, default=1, help="warmup runs, discarded (default 1)")
    ap.add_argument("--max-tokens", type=int, default=256, help="tokens per run (default 256)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help="override the benchmark prompt")
    ap.add_argument("--wait-start", type=int, default=0,
                    help="seconds to wait for the server to come up (used by auto_bench_all.bat)")
    args = ap.parse_args()

    base = "http://%s:%d" % (args.host, args.port)

    if args.wait_start > 0:
        print("Waiting up to %d s for server at %s ..." % (args.wait_start, base))
        if not wait_ready(base, args.wait_start):
            print("ERROR: server did not become ready in time (check the server console; "
                  "likely a VRAM OOM - see the .bat header for the fallback split).")
            sys.exit(2)

    if not wait_ready(base, 10):
        print("ERROR: no llama.cpp server answering at %s/health." % base)
        print("Start a server first by running one of the .bat scripts in C:\\LLM, "
              "then re-run this script.")
        sys.exit(1)

    model = read_model(base)
    props = read_props(base)
    pvals = props_values(props)

    # Capture memory usage
    vram_used, vram_total = get_vram_usage_gb()
    llama_ram = get_llama_ram_gb()
    sys_ram_used, sys_ram_total = get_system_ram_gb()

    print("Server ready. Model: %s" % model)
    mem_info = []
    if vram_used is not None:
        mem_info.append(f"VRAM: {vram_used}/{vram_total} GB")
    if llama_ram is not None:
        mem_info.append(f"llama-server RAM: {llama_ram} GB")
    if sys_ram_used is not None:
        mem_info.append(f"Sys RAM: {sys_ram_used}/{sys_ram_total} GB")
    if mem_info:
        print("Memory footprint: " + " | ".join(mem_info))

    if any(v is not None for v in pvals):
        print("Server props: ctx=%s batch=%s threads=%s/%s gpu-layers=%s KV=%s/%s fa=%s"
              % tuple(v if v is not None else "?" for v in pvals))

    print("Warmup: %d run(s) of %d tokens ..." % (args.warmup, args.max_tokens))
    for _ in range(max(0, args.warmup)):
        measure(base, args.prompt, min(args.max_tokens, 64))

    rows = []
    for i in range(1, args.runs + 1):
        print("Run %d/%d (%d tokens) ..." % (i, args.runs, args.max_tokens))
        m = measure(base, args.prompt, args.max_tokens)
        print("  prompt: %d tok @ %.1f t/s | gen: %d tok @ %.2f t/s (client %.2f) | %.1f s"
              % (m["prompt_tokens"], m["prompt_tps"], m["gen_tokens"],
                 m["gen_tps"], m["client_tps"], m["wall_s"]))
        row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config": args.config,
            "model": model,
            "vram_used_gb": vram_used if vram_used is not None else "",
            "vram_total_gb": vram_total if vram_total is not None else "",
            "llama_ram_gb": llama_ram if llama_ram is not None else "",
            "sys_ram_used_gb": sys_ram_used if sys_ram_used is not None else "",
            "sys_ram_total_gb": sys_ram_total if sys_ram_total is not None else "",
            "prompt_tokens": m["prompt_tokens"],
            "prompt_tps": m["prompt_tps"],
            "gen_tokens": m["gen_tokens"],
            "gen_tps": m["gen_tps"],
            "client_tps": m["client_tps"],
            "wall_s": m["wall_s"],
        }
        row.update({k: v for k, v in zip(
            ["n_ctx", "n_batch", "n_threads", "n_threads_batch",
             "n_gpu_layers", "cache_type_k", "cache_type_v", "flash_attn"],
            pvals)})
        rows.append(row)

    append_rows(rows)
    rebuild_md(load_all_rows())

    g = [r["gen_tps"] for r in rows if r.get("gen_tps") not in ("", None)]
    if g:
        print()
        print("Config '%s': mean gen %.2f t/s (median %.2f, min %.2f, max %.2f)"
              % (args.config, statistics.fmean(g), statistics.median(g), min(g), max(g)))
    print("Report CSV : %s" % REPORT_CSV)
    print("Report MD  : %s" % REPORT_MD)


if __name__ == "__main__":
    main()
