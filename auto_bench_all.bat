@echo off
REM ============================================================
REM  auto_bench_all.bat - FULLY AUTOMATIC benchmark of all configs.
REM  Double-click it: for each config (baseline + V1..V5) it
REM    1. kills any old llama-server
REM    2. starts the config in a minimized window
REM    3. waits up to 300 s for the model to load
REM    4. runs benchmark.py (3 measured runs of 256 tokens)
REM    5. kills the server and moves to the next config
REM  At the end it shows which configs failed (e.g. VRAM OOM)
REM  and prints the comparison report (bench\report.md).
REM
REM  Output: bench\report.md  (and bench\report.csv)
REM  Optional, for consistent CPU speeds (high performance plan):
REM    powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c
REM ============================================================
setlocal enabledelayedexpansion
cd /d %~dp0..

set BIN=.\llamacpp-cuda-13.3v0.3.0\llama-server.exe
set MODEL=C:\LLM\Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf
set RUNS=3
set FAILED=

set COMMON=--model "%MODEL%" --n-gpu-layers 99 --load-mode none --jinja --cache-type-k q4_0 --cache-type-v q4_0 --flash-attn on --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --parallel 1 --presence-penalty 0.0 --repeat-penalty 1.0 --cache-idle-slots --host 127.0.0.1 --port 8080 --timeout 36000 --sse-ping-interval 15 --reasoning on --reasoning-preserve --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}"

taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo === [0/5] BASELINE (your existing config) ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" -t 6 -tb 6 --ctx-size 77824 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 2
python bench\benchmark.py baseline --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] baseline: server did not start (see the llama-bench window)
  set FAILED=!FAILED! baseline
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo === [1/5] V1 CPU threads 16 + priority ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" -t 16 -tb 16 --prio 2 --cpu-strict --ctx-size 77824 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 2
python bench\benchmark.py v1-cpu-t16 --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v1-cpu-t16: server did not start
  set FAILED=!FAILED! v1-cpu-t16
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo === [2/5] V2 more GPU (blk 0-27 CPU), ctx 32768 ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|1[0-9]|2[0-7])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 32768 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 2
python bench\benchmark.py v2-gpu-ctx32k --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v2-gpu-ctx32k: server did not start (likely VRAM OOM)
  set FAILED=!FAILED! v2-gpu-ctx32k
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo === [3/5] V3 aggressive GPU (blk 0-19 CPU), ctx 16384 ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|1[0-5])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 16384 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 2
python bench\benchmark.py v3-gpu-ctx16k --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v3-gpu-ctx16k: server did not start (likely VRAM OOM)
  set FAILED=!FAILED! v3-gpu-ctx16k
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo === [4/5] V4 deeper MTP speculation (n-max 4), ctx 32768 ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 32768 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 4
python bench\benchmark.py v4-mtp4 --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v4-mtp4: server did not start
  set FAILED=!FAILED! v4-mtp4
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo === [5/5] V5 ngram speculation + batch 512, ctx 32768 ===
start "llama-bench" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 32768 -b 512 -ub 256 --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
python bench\benchmark.py v5-ngram --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v5-ngram: server did not start
  set FAILED=!FAILED! v5-ngram
)
taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo ============================================================
if defined FAILED (
  echo  Configs that FAILED to start: %FAILED%
  echo  (check VRAM in Task Manager; see the .bat headers for the
  echo   fallback GPU/CPU split to widen the CPU range)
) else (
  echo  All configs benchmarked OK.
)
echo.
echo  ==================== RESULTADO ====================
if exist bench\report.md (
  type bench\report.md
) else (
  echo  No report generated (all servers failed?).
)
echo  ============================================================
echo  Full report: bench\report.md   (raw data: bench\report.csv)
endlocal
pause
