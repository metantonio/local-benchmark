@echo off
REM ============================================================
REM  Qwen3.8-27B-V5-ngram-ctx77k.bat
REM  Configuracion GANADORA del benchmark:
REM  - Velocidad: ~18.8 - 19.5 tokens/segundo (vs 5.3 baseline)
REM  - Contexto completo: 77,824 tokens (~77k)
REM  - Consumo de memoria: ~10.0 GB VRAM + ~4.8 GB RAM
REM  - Especulacion: N-Gram (ngram-mod) + Batch 512/256 + 16 threads
REM ============================================================
setlocal enabledelayedexpansion

REM Ajustar directorio de trabajo a C:\LLM
if exist "%~dp0..\llamacpp-cuda-13.3v0.3.0\llama-server.exe" (
    cd /d "%~dp0.."
) else (
    cd /d "%~dp0"
)

set BIN=.\llamacpp-cuda-13.3v0.3.0\llama-server.exe
set MODEL=C:\LLM\Qwen3.8-27B-ASCII-Condensed-UD-IQ4_XS.gguf

title llama-server [Qwen 27B - V5 N-Gram 77k - 19 tok/s]

echo ============================================================
echo  Iniciando servidor llama.cpp con configuracion OPTIMA:
echo  - Modelo: %MODEL%
echo  - Contexto: 77,824 tokens (77k)
echo  - Especulacion: N-Gram (ngram-mod: match=24, min=48, max=64)
echo  - Split CPU/GPU: blk 0-35 CPU, resto en GPU
echo  - Hilos CPU: 16 (inferencia) / 16 (batch)
echo  - Batch / Micro-batch: 512 / 256
echo  - KV Cache: q4_0 con Flash Attention
echo  - Consumo VRAM esperado: ~10.0 GB (dentro de tus 12 GB)
echo  - Servidor escuchando en: http://127.0.0.1:8080
echo ============================================================
echo.

%BIN% ^
  --model "%MODEL%" ^
  --n-gpu-layers 99 ^
  --load-mode none ^
  --jinja ^
  --cache-type-k q4_0 ^
  --cache-type-v q4_0 ^
  --flash-attn on ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 20 ^
  --min-p 0.0 ^
  --parallel 1 ^
  --presence-penalty 0.0 ^
  --repeat-penalty 1.0 ^
  --cache-idle-slots ^
  --host 127.0.0.1 ^
  --port 8080 ^
  --timeout 36000 ^
  --sse-ping-interval 15 ^
  --reasoning on ^
  --reasoning-preserve ^
  --chat-template-kwargs "{\"reasoning_effort\":\"medium\"}" ^
  --override-tensor "blk\.([0-9]|[1-2][0-9]|3[0-5])\.ffn_.*=CPU" ^
  -t 16 ^
  -tb 16 ^
  --ctx-size 77824 ^
  -b 512 ^
  -ub 256 ^
  --spec-type ngram-mod ^
  --spec-ngram-mod-n-match 24 ^
  --spec-ngram-mod-n-min 48 ^
  --spec-ngram-mod-n-max 64

if errorlevel 1 (
  echo.
  echo [ERROR] El servidor finalizo con codigo de error.
  pause
)
