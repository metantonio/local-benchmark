```bat
@echo off
setlocal

echo ============================================================
echo Qwen3.8-Flash-Next - llama-server
echo RTX 4080 Laptop 12 GB / 32 GB RAM
echo ============================================================
echo.

set "LLAMA=C:\LLM\llama.cpp-thecodacus\build\bin\Release"
set "MODEL=C:\LLM\QWEN3.8-flash-next-UD-IQ3_XXS\Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"
set "PROFILE=C:\LLM\qwen38-profile\qwen38-merged.csv"
set "MTP=C:\LLM\QWEN3.8-flash-next-UD-IQ3_XXS\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"

REM ------------------------------------------------------------
REM MoE cache
REM Cambia solamente este valor para probar 40/48/56/64...
REM ------------------------------------------------------------
set "SLOTS=32"

REM ------------------------------------------------------------
REM Performance / memory
REM ------------------------------------------------------------
set "CONTEXT=135536"
set "BATCH=512"
set "UBATCH=256"
set "THREADS=6"

REM ------------------------------------------------------------
REM Start llama-server
REM ------------------------------------------------------------
echo Starting llama-server with %SLOTS% MoE cache slots...
echo.
echo Command:
echo %LLAMA%\llama-server.exe
echo.

"%LLAMA%\llama-server.exe" ^
  -m "%MODEL%" ^
  --moe-cache-profile "%PROFILE%" ^
  --moe-cache-slots %SLOTS% ^
  -md "%MTP%" ^
  -ngl 99 --no-sched-async-cpu ^
  -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1 ^
  -ncmoe 99 ^
  --load-mode mmap ^
  -fit off ^
  -fa on ^
  -ctk q8_0 ^
  -ctv q8_0 ^
  -c %CONTEXT% ^
  -np 1 ^
  -b %BATCH% ^
  -ub %UBATCH% ^
  -t %THREADS% ^
  --cache-reuse 256 ^
  --host 127.0.0.1 ^
  --port 8080

echo.
echo ============================================================
echo llama-server stopped.
echo ============================================================
pause
```
