#requires -Version 5.1

<#
.SYNOPSIS
    Build, profile and benchmark Qwen3.8-Flash-Next UD-IQ3_XXS
    using thecodacus/llama.cpp branch "perf" on Windows.

.HARDWARE TARGET
    RTX 4080 Laptop 12 GB VRAM
    32 GB system RAM

.DESCRIPTION
    Phases:
      1. Check prerequisites
      2. Clone/update llama.cpp branch perf
      3. Build CUDA Release
      4. Validate Qwen3.8 GGUF shards
      5. Generate MoE routing profiles
      6. Merge profiles
      7. Test MoE cache slot sizes
      8. Benchmark baseline
      9. Benchmark optimized configuration

    MTP is intentionally NOT enabled during profiling.
#>

$ErrorActionPreference = "Stop"

# ============================================================
# CONFIGURATION
# ============================================================

$Root = "C:\LLM"

$RepoDir   = Join-Path $Root "llama.cpp-thecodacus"
$ModelDir  = Join-Path $Root "QWEN3.8-flash-next-UD-IQ3_XXS"
$ProfileDir = Join-Path $Root "qwen38-profile"
$LogDir     = Join-Path $ProfileDir "logs"

$RepoUrl = "https://github.com/thecodacus/llama.cpp.git"
$Branch  = "perf"

$Model = Join-Path $ModelDir `
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"

# Profiling parameters
$ProfileContext = 4096
$ProfileTokens  = 512

# Initial cache sizes to test.
# We can change these after seeing the first results.
$CacheSlotsToTest = @(24, 32, 40, 48, 56, 64, 72, 80)

# Initial inference parameters
$Threads = 6
$Context = 4096
$Batch   = 512
$UBatch  = 256

# Benchmark
$BenchmarkPromptTokens = 512
$BenchmarkGeneration   = 128

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function Write-Section {
    param([string]$Text)

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Info {
    param([string]$Text)

    Write-Host "[INFO] $Text" -ForegroundColor Gray
}

function Write-OK {
    param([string]$Text)

    Write-Host "[ OK ] $Text" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Text)

    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Write-Fail {
    param([string]$Text)

    Write-Host "[FAIL] $Text" -ForegroundColor Red
}

function Require-Command {
    param([string]$Command)

    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $Command"
    }

    Write-OK "$Command found"
}

function Find-Executable {
    param(
        [string]$BaseDir,
        [string]$Name
    )

    $result = Get-ChildItem `
        -Path $BaseDir `
        -Filter $Name `
        -Recurse `
        -File `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -eq $result) {
        throw "Could not find $Name under $BaseDir"
    }

    return $result.FullName
}

function Get-MemorySnapshot {

    $gpu = nvidia-smi `
        --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu `
        --format=csv,noheader,nounits 2>$null

    Write-Host ""
    Write-Host "GPU:" -ForegroundColor Cyan
    Write-Host $gpu

    $os = Get-CimInstance Win32_OperatingSystem

    $totalGB = [math]::Round($os.TotalVisibleMemorySize / 1MB, 2)
    $freeGB  = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
    $usedGB  = [math]::Round($totalGB - $freeGB, 2)

    Write-Host ""
    Write-Host "RAM:" -ForegroundColor Cyan
    Write-Host "Used : $usedGB GB"
    Write-Host "Free : $freeGB GB"
    Write-Host "Total: $totalGB GB"
}

function Run-And-Log {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$LogFile
    )

    Write-Info "Running:"
    Write-Host "$Exe $($Arguments -join ' ')"

    & $Exe @Arguments 2>&1 |
        Tee-Object -FilePath $LogFile

    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

# ============================================================
# INITIALIZATION
# ============================================================

Write-Section "Qwen3.8 llama.cpp perf setup"

New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

Write-Info "Root       : $Root"
Write-Info "Repository : $RepoDir"
Write-Info "Model      : $Model"
Write-Info "Profiles   : $ProfileDir"

# ============================================================
# PHASE 1 — PREREQUISITES
# ============================================================

Write-Section "Phase 1 — Checking prerequisites"

Require-Command git
Require-Command cmake
Require-Command nvcc
Require-Command nvidia-smi

Write-Host ""
Write-Host "CUDA:" -ForegroundColor Cyan
nvcc --version

Write-Host ""
Write-Host "CMake:" -ForegroundColor Cyan
cmake --version | Select-Object -First 1

Write-Host ""
Write-Host "Git:" -ForegroundColor Cyan
git --version

Write-Host ""
Write-Host "GPU:" -ForegroundColor Cyan
nvidia-smi

# ============================================================
# PHASE 2 — REPOSITORY
# ============================================================

Write-Section "Phase 2 — llama.cpp branch perf"

if (-not (Test-Path $RepoDir)) {

    Write-Info "Repository does not exist. Cloning..."

    git clone `
        --branch $Branch `
        --single-branch `
        $RepoUrl `
        $RepoDir

}
else {

    Write-Info "Repository already exists."

    Push-Location $RepoDir

    git fetch origin

    git checkout $Branch

    git pull --ff-only origin $Branch

    Pop-Location
}

Push-Location $RepoDir

$currentBranch = git branch --show-current
$commit = git rev-parse --short HEAD

Write-Info "Branch: $currentBranch"
Write-Info "Commit: $commit"

if ($currentBranch -ne $Branch) {
    Pop-Location
    throw "Expected branch '$Branch' but found '$currentBranch'"
}

Write-OK "Correct branch detected: $Branch"

Pop-Location

# ============================================================
# PHASE 3 — BUILD
# ============================================================

Write-Section "Phase 3 — Building llama.cpp with CUDA"

$BuildDir = Join-Path $RepoDir "build"

if (Test-Path $BuildDir) {

    Write-Warn "Existing build directory detected."

    $answer = Read-Host `
        "Delete build directory and rebuild from scratch? [Y/N]"

    if ($answer -match "^[Yy]$") {
        Remove-Item -Recurse -Force $BuildDir
    }
}

if (-not (Test-Path $BuildDir)) {

    cmake `
        -S $RepoDir `
        -B $BuildDir `
        -DGGML_CUDA=ON `
        -DCMAKE_BUILD_TYPE=Release

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed."
    }
}

cmake `
    --build $BuildDir `
    --config Release `
    -j 16

if ($LASTEXITCODE -ne 0) {
    throw "Compilation failed."
}

Write-OK "Build completed."

# ============================================================
# FIND BINARIES
# ============================================================

Write-Section "Locating llama.cpp executables"

$ServerExe = Find-Executable `
    $BuildDir `
    "llama-server.exe"

Write-OK "llama-server: $ServerExe"

$TraceExe = Find-Executable `
    $BuildDir `
    "llama-moe-trace.exe"

Write-OK "llama-moe-trace: $TraceExe"

$BenchExe = Find-Executable `
    $BuildDir `
    "llama-bench.exe"

Write-OK "llama-bench: $BenchExe"

# ============================================================
# PHASE 4 — VALIDATE MODEL
# ============================================================

Write-Section "Phase 4 — Validating Qwen3.8 GGUF"

if (-not (Test-Path $Model)) {
    throw "Model shard not found: $Model"
}

$shards = Get-ChildItem `
    -Path $ModelDir `
    -Filter "*.gguf" `
    -File |
    Sort-Object Name

if ($shards.Count -lt 3) {

    Write-Warn `
        "Expected at least 3 GGUF files. Found $($shards.Count)."

}
else {

    Write-OK "Found $($shards.Count) GGUF files."

    foreach ($file in $shards) {

        $sizeGB = [math]::Round($file.Length / 1GB, 2)

        Write-Host `
            ("{0,-70} {1,8} GB" -f $file.Name, $sizeGB)
    }
}

# ============================================================
# MEMORY SNAPSHOT
# ============================================================

Write-Section "Hardware memory before profiling"

Get-MemorySnapshot

# ============================================================
# PHASE 5 — CHECK SERVER OPTIONS
# ============================================================

Write-Section "Phase 5 — Checking fork-specific options"

$HelpFile = Join-Path $LogDir "llama-server-help.txt"

& $ServerExe --help 2>&1 |
    Tee-Object -FilePath $HelpFile |
    Select-String `
        "moe-cache|n-cpu-moe|load-mode|flash|cache-reuse"

Write-Host ""

$TraceHelpFile = Join-Path $LogDir "llama-moe-trace-help.txt"

& $TraceExe --help 2>&1 |
    Tee-Object -FilePath $TraceHelpFile

Write-OK "Help output saved to $LogDir"

# ============================================================
# PHASE 6 — GENERATE CHAT PROFILE
# ============================================================

Write-Section "Phase 6 — Generating Chat MoE profile"

$ChatProfile = Join-Path `
    $ProfileDir `
    "qwen38-chat.csv"

$env:MOE_TRACE_OUT = $ChatProfile

$chatPrompt = @"
You are a helpful AI assistant.

Explain the advantages and disadvantages of using local
large language models for enterprise software development.

Discuss architecture, privacy, performance, cost, security,
deployment, maintenance and scalability.

Provide a detailed technical answer.
"@

$chatLog = Join-Path `
    $LogDir `
    "profile-chat.log"

Run-And-Log `
    $TraceExe `
    @(
        "-m", $Model,
        "-ngl", "99",
        "--n-cpu-moe", "99",
        "-fa", "1",
        "-c", "$ProfileContext",
        "-n", "$ProfileTokens",
        "-p", $chatPrompt
    ) `
    $chatLog

Remove-Item Env:MOE_TRACE_OUT -ErrorAction SilentlyContinue

if (-not (Test-Path $ChatProfile)) {
    throw "Chat profile was not generated."
}

Write-OK "Chat profile generated: $ChatProfile"

# ============================================================
# PHASE 7 — GENERATE CODE PROFILE
# ============================================================

Write-Section "Phase 7 — Generating Code MoE profile"

$CodeProfile = Join-Path `
    $ProfileDir `
    "qwen38-code.csv"

$env:MOE_TRACE_OUT = $CodeProfile

$codePrompt = @"
Write a production-quality Python REST API service.

Explain the architecture, error handling, concurrency,
database access layer, authentication, testing strategy,
logging, monitoring and performance considerations.

Include complete Python code and detailed explanations.
"@

$codeLog = Join-Path `
    $LogDir `
    "profile-code.log"

Run-And-Log `
    $TraceExe `
    @(
        "-m", $Model,
        "-ngl", "99",
        "--n-cpu-moe", "99",
        "-fa", "1",
        "-c", "$ProfileContext",
        "-n", "$ProfileTokens",
        "-p", $codePrompt
    ) `
    $codeLog

Remove-Item Env:MOE_TRACE_OUT -ErrorAction SilentlyContinue

if (-not (Test-Path $CodeProfile)) {
    throw "Code profile was not generated."
}

Write-OK "Code profile generated: $CodeProfile"

# ============================================================
# PHASE 8 — MERGE PROFILES
# ============================================================

Write-Section "Phase 8 — Merging MoE profiles"

$MergedProfile = Join-Path `
    $ProfileDir `
    "qwen38-merged.csv"

$chatLines = Get-Content $ChatProfile
$codeLines = Get-Content $CodeProfile

if ($chatLines.Count -eq 0) {
    throw "Chat profile is empty."
}

if ($codeLines.Count -eq 0) {
    throw "Code profile is empty."
}

$merged = New-Object System.Collections.Generic.List[string]

foreach ($line in $chatLines) {
    $merged.Add($line)
}

# Avoid duplicate CSV header
$startIndex = 0

if ($codeLines[0] -eq $chatLines[0]) {
    $startIndex = 1
}

for ($i = $startIndex; $i -lt $codeLines.Count; $i++) {
    $merged.Add($codeLines[$i])
}

$merged | Set-Content $MergedProfile

Write-OK "Merged profile created:"
Write-Host $MergedProfile

# ============================================================
# PHASE 9 — CACHE SLOT TESTS
# ============================================================

Write-Section "Phase 9 — Testing MoE cache slots"

Write-Info "Testing slots:"
Write-Host ($CacheSlotsToTest -join ", ")

$SlotResults = @()

foreach ($slots in $CacheSlotsToTest) {

    Write-Section "Testing $slots MoE cache slots"

    Get-MemorySnapshot

    $slotLog = Join-Path `
        $LogDir `
        "cache-slots-$slots.log"

    Write-Info "Starting server with $slots cache slots."

    $args = @(
        "-m", $Model,

        "--moe-cache-profile",
        $MergedProfile,

        "--moe-cache-slots",
        "$slots",

        "-ngl", "99",

        "--n-cpu-moe", "99",

        "--load-mode",
        "mmap",

        "-fa",
        "on",

        "-ctk",
        "q8_0",

        "-ctv",
        "q8_0",

        "-c",
        "$Context",

        "-np",
        "1",

        "-b",
        "$Batch",

        "-ub",
        "$UBatch",

        "-t",
        "$Threads",

        "--host",
        "127.0.0.1",

        "--port",
        "8080"
    )

    Write-Info "Command:"
    Write-Host "$ServerExe $($args -join ' ')"

    # Run server briefly and capture initialization.
    $process = Start-Process `
        -FilePath $ServerExe `
        -ArgumentList $args `
        -RedirectStandardOutput $slotLog `
        -RedirectStandardError (
            Join-Path $LogDir "cache-slots-$slots-error.log"
        ) `
        -PassThru `
        -WindowStyle Hidden

    Start-Sleep -Seconds 15

    $stillRunning = -not $process.HasExited

    if ($stillRunning) {

        Write-OK `
            "Server initialized and is still running with $slots slots."

        Get-MemorySnapshot

        $SlotResults += [PSCustomObject]@{
            Slots = $slots
            Status = "STARTED"
            Log = $slotLog
        }

        Stop-Process `
            -Id $process.Id `
            -Force `
            -ErrorAction SilentlyContinue

    }
    else {

        Write-Warn `
            "Server exited during initialization with $slots slots."

        $SlotResults += [PSCustomObject]@{
            Slots = $slots
            Status = "FAILED"
            Log = $slotLog
        }
    }

    Start-Sleep -Seconds 3
}

# ============================================================
# SLOT RESULTS
# ============================================================

Write-Section "MoE cache slot results"

$SlotResults |
    Format-Table -AutoSize

$SlotResults |
    Export-Csv `
        (Join-Path $ProfileDir "cache-slot-results.csv") `
        -NoTypeInformation

# ============================================================
# PHASE 10 — BASELINE BENCHMARK
# ============================================================

Write-Section "Phase 10 — Baseline benchmark"

$BaselineLog = Join-Path `
    $LogDir `
    "benchmark-baseline.log"

$baselineArgs = @(
    "-m", $Model,

    "-ngl", "99",

    "--n-cpu-moe", "99",

    "-fa", "1",

    "-p", "$BenchmarkPromptTokens",

    "-n", "$BenchmarkGeneration",

    "-b", "$Batch",

    "-ub", "$UBatch",

    "-r", "3"
)

Run-And-Log `
    $BenchExe `
    $baselineArgs `
    $BaselineLog

# ============================================================
# PHASE 11 — OPTIMIZED BENCHMARK
# ============================================================

Write-Section "Phase 11 — Optimized benchmark"

$successful = $SlotResults |
    Where-Object { $_.Status -eq "STARTED" }

if ($successful.Count -eq 0) {

    Write-Warn `
        "No cache slot configuration started successfully."

}
else {

    # Pick the highest successful slot count.
    $bestSlots = ($successful |
        Sort-Object Slots -Descending |
        Select-Object -First 1).Slots

    Write-Info `
        "Highest successfully initialized cache: $bestSlots slots"

    # Enable host registration and expert prefetch.
    $env:GGML_CUDA_REGISTER_HOST = "1"
    $env:GGML_SCHED_PREFETCH_EXPERTS = "1"

    $OptimizedLog = Join-Path `
        $LogDir `
        "benchmark-optimized-$bestSlots.log"

    $optimizedArgs = @(
        "-m", $Model,

        "-ngl", "99",

        "--n-cpu-moe", "99",

        "--moe-cache-profile",
        $MergedProfile,

        "--moe-cache-slots",
        "$bestSlots",

        "-fa", "1",

        "-p", "$BenchmarkPromptTokens",

        "-n", "$BenchmarkGeneration",

        "-b", "$Batch",

        "-ub", "$UBatch",

        "-r", "3"
    )

    Run-And-Log `
        $BenchExe `
        $optimizedArgs `
        $OptimizedLog

    Remove-Item Env:GGML_CUDA_REGISTER_HOST `
        -ErrorAction SilentlyContinue

    Remove-Item Env:GGML_SCHED_PREFETCH_EXPERTS `
        -ErrorAction SilentlyContinue
}

# ============================================================
# FINAL MEMORY SNAPSHOT
# ============================================================

Write-Section "Final memory snapshot"

Get-MemorySnapshot

# ============================================================
# SUMMARY
# ============================================================

Write-Section "COMPLETE"

Write-Host ""
Write-Host "Repository:" -ForegroundColor Cyan
Write-Host $RepoDir

Write-Host ""
Write-Host "Model:" -ForegroundColor Cyan
Write-Host $Model

Write-Host ""
Write-Host "Profiles:" -ForegroundColor Cyan
Write-Host $ProfileDir

Write-Host ""
Write-Host "Logs:" -ForegroundColor Cyan
Write-Host $LogDir

Write-Host ""
Write-Host "Important files:" -ForegroundColor Cyan
Write-Host "  $ChatProfile"
Write-Host "  $CodeProfile"
Write-Host "  $MergedProfile"
Write-Host "  $(Join-Path $ProfileDir 'cache-slot-results.csv')"

Write-Host ""
Write-Host "The profiling/benchmark phase has completed." `
    -ForegroundColor Green

Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "MTP was intentionally NOT enabled."
Write-Host "Do not change the configuration until the slot results"
Write-Host "and benchmark logs have been reviewed."

Write-Host ""
