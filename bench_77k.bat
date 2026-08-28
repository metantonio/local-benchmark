@echo off
REM ============================================================
REM  bench_77k.bat - Test variants with 77k context (77824)
REM  Tests:
REM    1. v5-ngram-ctx77k      (N-gram speculation + blk 0-35 CPU, ctx 77824)
REM    2. v3-gpu-ctx77k        (Max GPU viable: blk 0-27 CPU + MTP, ctx 77824)
REM    3. v6-combo-ngram-ctx77k (COMBINED: Max GPU blk 0-27 CPU + N-gram + ctx 77824)
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
echo ============================================================
echo === [1/3] v5-ngram-ctx77k (N-gram speculation, ctx 77824) ===
echo ============================================================
start "llama-bench-77k" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 77824 -b 512 -ub 256 --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
python bench\benchmark.py v5-ngram-ctx77k --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v5-ngram-ctx77k: server did not start
  set FAILED=!FAILED! v5-ngram-ctx77k
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo === [2/3] v3-gpu-ctx77k (Max GPU split blk 0-27 CPU, ctx 77824) ===
echo ============================================================
start "llama-bench-77k" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|1[0-9]|2[0-7])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 77824 -b 256 -ub 128 --spec-type draft-mtp --spec-draft-n-max 2
python bench\benchmark.py v3-gpu-ctx77k --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v3-gpu-ctx77k: server did not start (likely VRAM OOM)
  set FAILED=!FAILED! v3-gpu-ctx77k
)
taskkill /F /IM llama-server.exe >nul 2>&1
timeout /t 5 /nobreak >nul

echo.
echo ============================================================
echo === [3/3] v6-combo-ngram-ctx77k (GPU blk 0-27 + N-gram, ctx 77824) ===
echo ============================================================
start "llama-bench-77k" /MIN %BIN% %COMMON% --override-tensor "blk\.([0-9]|1[0-9]|2[0-7])\.ffn_.*=CPU" -t 16 -tb 16 --ctx-size 77824 -b 512 -ub 256 --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
python bench\benchmark.py v6-combo-ngram-ctx77k --wait-start 300 --runs %RUNS%
if errorlevel 1 (
  echo   [FAILED] v6-combo-ngram-ctx77k: server did not start (likely VRAM OOM)
  set FAILED=!FAILED! v6-combo-ngram-ctx77k
)
taskkill /F /IM llama-server.exe >nul 2>&1

echo.
echo ============================================================
if defined FAILED (
  echo  Configs that FAILED to start: %FAILED%
) else (
  echo  All 77k configs benchmarked OK.
)
echo.
echo  ==================== RESULTADO ====================
if exist bench\report.md (
  type bench\report.md
)
echo  ============================================================
echo  Full report: bench\report.md   (raw data: bench\report.csv)
endlocal
pause
