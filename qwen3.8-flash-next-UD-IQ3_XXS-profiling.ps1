#requires -Version 5.1

<#
.SYNOPSIS
    Build, profile and benchmark Qwen3.8-Flash-Next UD-IQ3_XXS
    using thecodacus/llama.cpp branch "perf" on Windows.

.HARDWARE
    RTX 4080 Laptop GPU - 12 GB VRAM
    32 GB system RAM

.DESCRIPTION
    This script is intentionally re-runnable / idempotent.

    First execution:
        1. Check prerequisites
        2. Clone llama.cpp perf branch if missing
        3. Build llama.cpp if required binaries are missing
        4. Validate model shards
        5. Generate Chat MoE profile
        6. Generate Code MoE profile
        7. Merge profiles
        8. Test MoE cache slots
        9. Benchmark baseline
       10. Benchmark optimized configuration

    Subsequent executions:
        - Do NOT rebuild if required binaries already exist.
        - Do NOT regenerate profiles that already exist.
        - Do NOT repeat successful cache-slot tests.
        - Do NOT repeat baseline benchmark if its log already exists.
        - Continue from the last incomplete phase.

    MTP is intentionally NOT enabled at this stage.

.NOTES
    Compatible with Windows PowerShell 5.1.
    Native processes are launched through System.Diagnostics.Process
    to avoid PowerShell 5.1 NativeCommandError behavior when programs
    write normal diagnostic output to stderr.
#>

$ErrorActionPreference = "Stop"

trap {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host " UNEXPECTED SCRIPT ERROR" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host $_.InvocationInfo.PositionMessage -ForegroundColor Yellow
    Write-Host ""

    Read-Host "Press ENTER to close"
    exit 1
}

# ============================================================
# CONFIGURATION
# ============================================================

$Root = "C:\LLM"

$RepoDir    = Join-Path $Root "llama.cpp-thecodacus"
$ModelDir   = Join-Path $Root "QWEN3.8-flash-next-UD-IQ3_XXS"
$ProfileDir = Join-Path $Root "qwen38-profile"
$LogDir     = Join-Path $ProfileDir "logs"

$RepoUrl = "https://github.com/thecodacus/llama.cpp.git"
$Branch  = "perf"

# IMPORTANT:
# Keep this FALSE for reproducible profiling.
# Set TRUE only when you explicitly want to update llama.cpp.
$UpdateRepo = $false

# Model - first shard.
# llama.cpp automatically discovers the other shards.
$Model = Join-Path $ModelDir `
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"

# Profiling
$ProfileContext = 4096
$ProfileTokens  = 512

# MoE cache sweep.
#
# These are deliberately conservative starting points.
# We can change them after seeing actual results.
$CacheSlotsToTest = @(24, 32, 40, 48, 56, 64, 72, 80)

# Inference
$Threads = 6
$Context = 4096
$Batch   = 512
$UBatch  = 256

# Benchmark
$BenchmarkPromptTokens = 512
$BenchmarkGeneration   = 128
$BenchmarkRuns         = 3

# Server startup timeout.
#
# A 76 GB model can take considerably longer than 15 seconds
# to initialize on a laptop.
$ServerStartupTimeoutSeconds = 180

# Wait between cache-slot tests.
$ServerShutdownWaitSeconds = 3

# ============================================================
# PATHS
# ============================================================

$BuildDir = Join-Path $RepoDir "build"

$ChatProfile   = Join-Path $ProfileDir "qwen38-chat.csv"
$CodeProfile   = Join-Path $ProfileDir "qwen38-code.csv"
$MergedProfile = Join-Path $ProfileDir "qwen38-merged.csv"

$SlotResultsFile = Join-Path $ProfileDir "cache-slot-results.csv"

$BaselineLog = Join-Path $LogDir "benchmark-baseline.log"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

function Write-Section {
    param(
        [string]$Text
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Write-Info {
    param(
        [string]$Text
    )

    Write-Host "[INFO] $Text" -ForegroundColor Gray
}

function Write-OK {
    param(
        [string]$Text
    )

    Write-Host "[ OK ] $Text" -ForegroundColor Green
}

function Write-Warn {
    param(
        [string]$Text
    )

    Write-Host "[WARN] $Text" -ForegroundColor Yellow
}

function Write-Fail {
    param(
        [string]$Text
    )

    Write-Host "[FAIL] $Text" -ForegroundColor Red
}

function Require-Command {
    param(
        [string]$Command
    )

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
        return $null
    }

    return $result.FullName
}

# ------------------------------------------------------------
# Robust native process runner for Windows PowerShell 5.1
# ------------------------------------------------------------
#
# IMPORTANT:
# Do NOT replace this with:
#
#   & $Exe @Arguments 2>&1 | Tee-Object ...
#
# because PowerShell 5.1 can convert stderr from native programs
# into NativeCommandError records.
#
# This implementation captures stdout/stderr directly from the
# child process.
#
function Run-NativeLogged {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @(),

        [Parameter(Mandatory = $true)]
        [string]$LogFile
    )

    Write-Info "Running:"
    Write-Host "$Exe $($Arguments -join ' ')"

    $logDirectory = Split-Path -Parent $LogFile

    if (-not (Test-Path $logDirectory)) {
        New-Item -ItemType Directory -Force -Path $logDirectory |
            Out-Null
    }

    # ------------------------------------------------------------
    # Build command line for Windows PowerShell 5.1 / .NET Framework
    # ------------------------------------------------------------

    $quotedArguments = foreach ($arg in $Arguments) {

        if ($null -eq $arg) {
            '""'
            continue
        }

        $text = [string]$arg

        if ($text -match '[\s"]') {

            # Escape quotes for Windows command-line parsing.
            $escaped = $text -replace '(\\*)"', '$1$1\"'

            # Escape trailing backslashes before closing quote.
            $escaped = $escaped -replace '(\\+)$', '$1$1'

            '"' + $escaped + '"'
        }
        else {
            $text
        }
    }

    $argumentString = $quotedArguments -join " "

    # ------------------------------------------------------------
    # Start process
    # ------------------------------------------------------------

    $psi = New-Object System.Diagnostics.ProcessStartInfo

    $psi.FileName = $Exe
    $psi.Arguments = $argumentString

    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    try {

        if (-not $process.Start()) {
            throw "Could not start process: $Exe"
        }

        Write-Info "Process started. PID = $($process.Id)"

        # --------------------------------------------------------
        # IMPORTANT:
        # Read stdout/stderr synchronously.
        #
        # Do NOT use BeginOutputReadLine()/event handlers here.
        # --------------------------------------------------------

        $stdout = $process.StandardOutput.ReadToEnd()
        $stderr = $process.StandardError.ReadToEnd()

        # Wait until the process has completely terminated.
        $process.WaitForExit()

        $exitCode = $process.ExitCode

        Write-Info "Process finished. Exit code = $exitCode"

        # --------------------------------------------------------
        # Save complete log
        # --------------------------------------------------------

        $logContent = New-Object System.Text.StringBuilder

        [void]$logContent.AppendLine(
            "============================================================"
        )

        [void]$logContent.AppendLine("Executable:")
        [void]$logContent.AppendLine($Exe)
        [void]$logContent.AppendLine("")

        [void]$logContent.AppendLine("Arguments:")
        [void]$logContent.AppendLine(
            ($Arguments -join " ")
        )

        [void]$logContent.AppendLine("")

        [void]$logContent.AppendLine(
            "Exit code: $exitCode"
        )

        [void]$logContent.AppendLine(
            "============================================================"
        )

        [void]$logContent.AppendLine("")

        [void]$logContent.AppendLine("STDOUT:")
        [void]$logContent.AppendLine($stdout)

        [void]$logContent.AppendLine("")

        [void]$logContent.AppendLine("STDERR:")
        [void]$logContent.AppendLine($stderr)

        [System.IO.File]::WriteAllText(
            $LogFile,
            $logContent.ToString()
        )

        # --------------------------------------------------------
        # Display output after process finishes
        # --------------------------------------------------------

        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            Write-Host ""
            Write-Host "STDOUT:" -ForegroundColor Cyan
            Write-Host $stdout
        }

        if (-not [string]::IsNullOrWhiteSpace($stderr)) {
            Write-Host ""
            Write-Host "STDERR:" -ForegroundColor Yellow
            Write-Host $stderr
        }

        Write-Info "Log written to: $LogFile"

        if ($exitCode -ne 0) {
            throw "Native process failed with exit code $exitCode. See: $LogFile"
        }

        return $true
    }
    finally {

        if ($null -ne $process) {
            $process.Dispose()
        }
    }
}

# ------------------------------------------------------------
# Simple native command runner
# ------------------------------------------------------------
#
# Used for commands where we don't need streaming output.
#
function Run-Native {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    $tempLog = Join-Path $env:TEMP (
        "qwen38-native-" + [guid]::NewGuid().ToString() + ".log"
    )

    try {
        Run-NativeLogged `
            -Exe $Exe `
            -Arguments $Arguments `
            -LogFile $tempLog

        return $true
    }
    finally {
        Remove-Item $tempLog -Force -ErrorAction SilentlyContinue
    }
}

# ------------------------------------------------------------
# Memory snapshot
# ------------------------------------------------------------

function Get-MemorySnapshot {

    Write-Host ""
    Write-Host "GPU:" -ForegroundColor Cyan

    try {

        $gpu = & nvidia-smi `
            --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu `
            --format=csv,noheader,nounits 2>$null

        Write-Host $gpu
    }
    catch {
        Write-Warn "Could not query GPU memory."
    }

    try {

        $os = Get-CimInstance Win32_OperatingSystem

        $totalGB = [math]::Round(
            $os.TotalVisibleMemorySize / 1MB,
            2
        )

        $freeGB = [math]::Round(
            $os.FreePhysicalMemory / 1MB,
            2
        )

        $usedGB = [math]::Round(
            $totalGB - $freeGB,
            2
        )

        Write-Host ""
        Write-Host "RAM:" -ForegroundColor Cyan
        Write-Host "Used : $usedGB GB"
        Write-Host "Free : $freeGB GB"
        Write-Host "Total: $totalGB GB"
    }
    catch {
        Write-Warn "Could not query system RAM."
    }
}

# ------------------------------------------------------------
# Check TCP port
# ------------------------------------------------------------

function Test-PortOpen {
    param(
        [string]$HostName = "127.0.0.1",
        [int]$Port = 8080
    )

    try {

        $client = New-Object System.Net.Sockets.TcpClient

        $async = $client.BeginConnect(
            $HostName,
            $Port,
            $null,
            $null
        )

        $success = $async.AsyncWaitHandle.WaitOne(1000)

        if ($success -and $client.Connected) {
            $client.EndConnect($async)
            $client.Close()
            return $true
        }

        $client.Close()
        return $false
    }
    catch {
        return $false
    }
}

# ------------------------------------------------------------
# Wait for llama-server
# ------------------------------------------------------------

function Wait-ForServer {
    param(
        [Parameter(Mandatory = $true)]
        [System.Diagnostics.Process]$Process,

        [Parameter(Mandatory = $true)]
        [int]$TimeoutSeconds
    )

    Write-Info "Waiting for llama-server to listen on 127.0.0.1:8080..."

    $start = Get-Date

    while ($true) {

        if ($Process.HasExited) {

            Write-Warn "Server exited during startup."

            return $false
        }

        if (Test-PortOpen -HostName "127.0.0.1" -Port 8080) {

            Write-OK "Server is listening on port 8080."

            return $true
        }

        $elapsed = (
            (Get-Date) - $start
        ).TotalSeconds

        if ($elapsed -ge $TimeoutSeconds) {

            Write-Warn `
                "Server did not open port 8080 within $TimeoutSeconds seconds."

            return $false
        }

        Start-Sleep -Seconds 2
    }
}

# ------------------------------------------------------------
# Start server with stdout/stderr redirected directly to files
# ------------------------------------------------------------

function Start-LoggedServer {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [string]$StdOutFile,
        [string]$StdErrFile
    )

    $psi = New-Object System.Diagnostics.ProcessStartInfo

    $psi.FileName = $Exe

    $quotedArguments = foreach ($arg in $Arguments) {

        if ($null -eq $arg) {
            '""'
            continue
        }

        $text = [string]$arg

        if ($text -match '[\s"]') {

            $escaped = $text -replace '(\\*)"', '$1$1\"'
            $escaped = $escaped -replace '(\\+)$', '$1$1'

            '"' + $escaped + '"'
        }
        else {
            $text
        }
    }

    $psi.Arguments = ($quotedArguments -join " ")

    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process

    $process.StartInfo = $psi

    if (-not $process.Start()) {
        throw "Could not start llama-server."
    }

    # Asynchronously copy stdout/stderr to files.
    #
    # This avoids PowerShell touching the native stderr stream.
    $stdoutWriter = New-Object System.IO.StreamWriter(
        $StdOutFile,
        $false,
        [System.Text.Encoding]::UTF8
    )

    $stderrWriter = New-Object System.IO.StreamWriter(
        $StdErrFile,
        $false,
        [System.Text.Encoding]::UTF8
    )

    $process.add_OutputDataReceived({
        param($sender, $e)

        if ($null -ne $e.Data) {
            $stdoutWriter.WriteLine($e.Data)
            $stdoutWriter.Flush()
        }
    })

    $process.add_ErrorDataReceived({
        param($sender, $e)

        if ($null -ne $e.Data) {
            $stderrWriter.WriteLine($e.Data)
            $stderrWriter.Flush()
        }
    })

    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()

    return @{
        Process = $process
        StdOut  = $stdoutWriter
        StdErr  = $stderrWriter
    }
}

# ------------------------------------------------------------
# Stop server cleanly
# ------------------------------------------------------------

function Stop-LoggedServer {
    param(
        [hashtable]$Server
    )

    Write-Host ""
    Write-Host "========== STOP-LOGGED-SERVER ==========" -ForegroundColor Yellow

    if ($null -eq $Server) {
        Write-Host "Server object is NULL"
        return
    }

    Write-Host "Server object exists"

    $process = $Server.Process

    if ($null -eq $process) {
        Write-Host "Process object is NULL"
    }
    else {
        Write-Host "Process object exists"
        
        try {
            Write-Host "HasExited = $($process.HasExited)"
        }
        catch {
            Write-Host "ERROR reading HasExited: $($_.Exception.Message)" -ForegroundColor Red
        }

        try {
            if (-not $process.HasExited) {
            Write-Info "Stopping llama-server..."

            Write-Host "[DEBUG] About to Kill()"
            $process.Kill()
            Write-Host "[DEBUG] Kill() completed"

            Write-Host "[DEBUG] About to WaitForExit()"
            $waitResult = $process.WaitForExit(10000)
            Write-Host "[DEBUG] WaitForExit() returned: $waitResult"

            Write-Host "[DEBUG] Process HasExited: $($process.HasExited)"
            }
            else {
                Write-Host "Process was already exited"
            }
        }
        catch {
            Write-Host "ERROR stopping process:" -ForegroundColor Red
            Write-Host $_.Exception.ToString() -ForegroundColor Red
        }

        try {
            Write-Host "Disposing process..."
            $process.Dispose()
            Write-Host "Process disposed"
        }
        catch {
            Write-Host "ERROR disposing process: $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    try {
        Write-Host "Flushing/closing StdOut..."
        $Server.StdOut.Flush()
        $Server.StdOut.Close()
        Write-Host "StdOut closed"
    }
    catch {
        Write-Host "ERROR closing StdOut: $($_.Exception.Message)" -ForegroundColor Red
    }

    try {
        Write-Host "Flushing/closing StdErr..."
        $Server.StdErr.Flush()
        $Server.StdErr.Close()
        Write-Host "StdErr closed"
    }
    catch {
        Write-Host "ERROR closing StdErr: $($_.Exception.Message)" -ForegroundColor Red
    }

    Write-Host "========== STOP COMPLETE ==========" -ForegroundColor Yellow
}

# ============================================================
# INITIALIZATION
# ============================================================

Write-Section "Qwen3.8 llama.cpp perf setup"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $Root |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path $ProfileDir |
    Out-Null

New-Item `
    -ItemType Directory `
    -Force `
    -Path $LogDir |
    Out-Null

Write-Info "Root       : $Root"
Write-Info "Repository : $RepoDir"
Write-Info "Model      : $Model"
Write-Info "Profiles   : $ProfileDir"
Write-Info "Logs       : $LogDir"

# ============================================================
# PHASE 1 - PREREQUISITES
# ============================================================

Write-Section "Phase 1 - Checking prerequisites"

Require-Command git
Require-Command cmake
Require-Command nvcc
Require-Command nvidia-smi

Write-Host ""
Write-Host "CUDA:" -ForegroundColor Cyan

try {
    $nvccOutput = & nvcc --version 2>$null
    Write-Host $nvccOutput
}
catch {
    Write-Warn "Could not display nvcc version."
}

Write-Host ""
Write-Host "CMake:" -ForegroundColor Cyan

try {
    $cmakeOutput = & cmake --version 2>$null
    Write-Host ($cmakeOutput | Select-Object -First 1)
}
catch {
    Write-Warn "Could not display CMake version."
}

Write-Host ""
Write-Host "Git:" -ForegroundColor Cyan

try {
    $gitOutput = & git --version 2>$null
    Write-Host $gitOutput
}
catch {
    Write-Warn "Could not display Git version."
}

Write-Host ""
Write-Host "GPU:" -ForegroundColor Cyan

try {
    & nvidia-smi 2>$null
}
catch {
    Write-Warn "Could not execute nvidia-smi."
}

# ============================================================
# PHASE 2 - REPOSITORY
# ============================================================

Write-Section "Phase 2 - llama.cpp branch perf"

if (-not (Test-Path $RepoDir)) {

    Write-Info "Repository does not exist. Cloning branch '$Branch'..."

    $cloneLog = Join-Path $LogDir "git-clone.log"

    Run-NativeLogged `
        -Exe "git" `
        -Arguments @(
            "clone",
            "--branch", $Branch,
            "--single-branch",
            $RepoUrl,
            $RepoDir
        ) `
        -LogFile $cloneLog

    Write-OK "Repository cloned."
}
else {

    Write-OK "Repository already exists."

    Push-Location $RepoDir

    try {

        $currentBranch = (& git branch --show-current 2>$null).Trim()

        if ($currentBranch -ne $Branch) {

            Write-Warn `
                "Repository is on '$currentBranch', switching to '$Branch'."

            & git checkout $Branch 2>$null

            if ($LASTEXITCODE -ne 0) {
                throw "Could not checkout branch '$Branch'."
            }
        }

        if ($UpdateRepo) {

            Write-Info "Updating repository because UpdateRepo = TRUE."

            & git fetch origin 2>$null

            if ($LASTEXITCODE -ne 0) {
                throw "git fetch failed."
            }

            & git pull --ff-only origin $Branch 2>$null

            if ($LASTEXITCODE -ne 0) {
                throw "git pull failed."
            }

            Write-OK "Repository updated."
        }
        else {

            Write-Info `
                "Repository update skipped (UpdateRepo = FALSE)."
        }

        $currentBranch = (& git branch --show-current 2>$null).Trim()
        $commit = (& git rev-parse --short HEAD 2>$null).Trim()

        Write-Info "Branch: $currentBranch"
        Write-Info "Commit: $commit"

        if ($currentBranch -ne $Branch) {
            throw `
                "Expected branch '$Branch' but found '$currentBranch'."
        }

        Write-OK "Correct branch detected: $Branch"
    }
    finally {
        Pop-Location
    }
}

# ============================================================
# PHASE 3 - BUILD
# ============================================================

Write-Section "Phase 3 - Build status"

$ServerExe = Find-Executable `
    -BaseDir $BuildDir `
    -Name "llama-server.exe"

$TraceExe = Find-Executable `
    -BaseDir $BuildDir `
    -Name "llama-moe-trace.exe"

$BenchExe = Find-Executable `
    -BaseDir $BuildDir `
    -Name "llama-bench.exe"

$missingBuild = $false

if ($null -eq $ServerExe) {
    Write-Warn "llama-server.exe not found."
    $missingBuild = $true
}

if ($null -eq $TraceExe) {
    Write-Warn "llama-moe-trace.exe not found."
    $missingBuild = $true
}

if ($null -eq $BenchExe) {
    Write-Warn "llama-bench.exe not found."
    $missingBuild = $true
}

if (-not $missingBuild) {

    Write-OK "Required llama.cpp binaries already exist."
    Write-Info "Skipping CMake configuration and compilation."
}
else {

    Write-Info "Required binaries are missing."
    Write-Info "Building llama.cpp with CUDA..."

    if (-not (Test-Path $BuildDir)) {

        New-Item `
            -ItemType Directory `
            -Force `
            -Path $BuildDir |
            Out-Null

        $cmakeConfigureLog = Join-Path `
            $LogDir `
            "cmake-configure.log"

        Run-NativeLogged `
            -Exe "cmake" `
            -Arguments @(
                "-S", $RepoDir,
                "-B", $BuildDir,
                "-DGGML_CUDA=ON",
                "-DCMAKE_BUILD_TYPE=Release"
            ) `
            -LogFile $cmakeConfigureLog
    }
    else {

        Write-Info "Existing build directory detected."
        Write-Info "Reusing existing CMake configuration."
    }

    $cmakeBuildLog = Join-Path `
        $LogDir `
        "cmake-build.log"

    Run-NativeLogged `
        -Exe "cmake" `
        -Arguments @(
            "--build", $BuildDir,
            "--config", "Release",
            "-j", "16"
        ) `
        -LogFile $cmakeBuildLog

    Write-OK "Build completed."

    $ServerExe = Find-Executable `
        -BaseDir $BuildDir `
        -Name "llama-server.exe"

    $TraceExe = Find-Executable `
        -BaseDir $BuildDir `
        -Name "llama-moe-trace.exe"

    $BenchExe = Find-Executable `
        -BaseDir $BuildDir `
        -Name "llama-bench.exe"

    if ($null -eq $ServerExe) {
        throw "llama-server.exe was not produced by the build."
    }

    if ($null -eq $TraceExe) {
        throw "llama-moe-trace.exe was not produced by the build."
    }

    if ($null -eq $BenchExe) {
        throw "llama-bench.exe was not produced by the build."
    }
}

Write-OK "llama-server : $ServerExe"
Write-OK "llama-moe-trace: $TraceExe"
Write-OK "llama-bench  : $BenchExe"

# ============================================================
# PHASE 4 - VALIDATE MODEL
# ============================================================

Write-Section "Phase 4 - Validating Qwen3.8 GGUF"

if (-not (Test-Path $Model)) {
    throw "Model shard not found: $Model"
}

$shards = @(
    Get-ChildItem `
        -Path $ModelDir `
        -Filter "*.gguf" `
        -File |
        Sort-Object Name
)

if ($shards.Count -lt 3) {

    Write-Warn `
        "Expected 3 GGUF shards. Found $($shards.Count)."
}
else {

    Write-OK "Found $($shards.Count) GGUF files."

    foreach ($file in $shards) {

        $sizeGB = [math]::Round(
            $file.Length / 1GB,
            2
        )

        Write-Host (
            "{0,-75} {1,8} GB" -f
            $file.Name,
            $sizeGB
        )
    }
}

# ============================================================
# MEMORY SNAPSHOT
# ============================================================

Write-Section "Hardware memory before profiling"

Get-MemorySnapshot

# ============================================================
# Phase 5 - Checking fork-specific options
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host " Phase 5 - Checking fork-specific options"
Write-Host "============================================================"

Write-Host "[INFO] Checking llama-server..."
if (-not (Test-Path -LiteralPath $ServerExe)) {
    throw "llama-server.exe not found: $ServerExe"
}

Write-Host "[OK] llama-server.exe exists"
Write-Host "     $ServerExe"

Write-Host ""
Write-Host "[INFO] Checking llama-moe-trace..."
if (-not (Test-Path -LiteralPath $TraceExe)) {
    throw "llama-moe-trace.exe not found: $TraceExe"
}

Write-Host "[OK] llama-moe-trace.exe exists"
Write-Host "     $TraceExe"

Write-Host ""
Write-Host "[INFO] Checking model..."
if (-not (Test-Path -LiteralPath $Model)) {
    throw "Model not found: $Model"
}

Write-Host "[OK] Model exists"
Write-Host "     $Model"

Write-Host ""
Write-Host "[OK] Phase 5 completed."

# ============================================================
# PHASE 6 - CHAT PROFILE
# ============================================================

Write-Section "Phase 6 - Chat MoE profile"

if (Test-Path $ChatProfile) {

    $chatSize = (Get-Item $ChatProfile).Length

    if ($chatSize -gt 0) {

        Write-OK `
            "Chat profile already exists. Skipping profiling."

        Write-Info $ChatProfile
    }
    else {

        Remove-Item $ChatProfile -Force
    }
}

if (-not (Test-Path $ChatProfile)) {

    Write-Info `
        "Generating Chat profile with $ProfileTokens tokens..."

    $env:MOE_TRACE_OUT = $ChatProfile

    try {

        $chatPrompt = `
            "You are a helpful AI assistant. Explain the advantages and disadvantages of using local large language models for enterprise software development. Discuss architecture, privacy, performance, cost, security, deployment, maintenance and scalability. Provide a detailed technical answer."

        $chatLog = Join-Path `
            $LogDir `
            "profile-chat.log"

        Run-NativeLogged `
            -Exe $TraceExe `
            -Arguments @(
                "-m", $Model,
                "-ngl", "99",
                "-ncmoe", "99",
                "-fa", "1",
                "-c", "$ProfileContext",
                "-n", "$ProfileTokens",
                "-p", $chatPrompt
            ) `
            -LogFile $chatLog
    }
    finally {

        Remove-Item `
            Env:MOE_TRACE_OUT `
            -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path $ChatProfile)) {
        throw "Chat profile was not generated."
    }

    if ((Get-Item $ChatProfile).Length -eq 0) {
        throw "Chat profile is empty."
    }

    Write-OK "Chat profile generated."
}

# ============================================================
# PHASE 7 - CODE PROFILE
# ============================================================

Write-Section "Phase 7 - Code MoE profile"

if (Test-Path $CodeProfile) {

    $codeSize = (Get-Item $CodeProfile).Length

    if ($codeSize -gt 0) {

        Write-OK `
            "Code profile already exists. Skipping profiling."

        Write-Info $CodeProfile
    }
    else {

        Remove-Item $CodeProfile -Force
    }
}

if (-not (Test-Path $CodeProfile)) {

    Write-Info `
        "Generating Code profile with $ProfileTokens tokens..."

    $env:MOE_TRACE_OUT = $CodeProfile

    try {

        $codePrompt = `
            "Write a production-quality Python REST API service. Explain the architecture, error handling, concurrency, database access layer, authentication, testing strategy, logging, monitoring and performance considerations. Include complete Python code and detailed explanations."

        $codeLog = Join-Path `
            $LogDir `
            "profile-code.log"

        Run-NativeLogged `
            -Exe $TraceExe `
            -Arguments @(
                "-m", $Model,
                "-ngl", "99",
                "-ncmoe", "99",
                "-fa", "1",
                "-c", "$ProfileContext",
                "-n", "$ProfileTokens",
                "-p", $codePrompt
            ) `
            -LogFile $codeLog
    }
    finally {

        Remove-Item `
            Env:MOE_TRACE_OUT `
            -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path $CodeProfile)) {
        throw "Code profile was not generated."
    }

    if ((Get-Item $CodeProfile).Length -eq 0) {
        throw "Code profile is empty."
    }

    Write-OK "Code profile generated."
}

# ============================================================
# PHASE 8 - MERGE PROFILES
# ============================================================

Write-Section "Phase 8 - Merging MoE profiles"

if (
    (Test-Path $MergedProfile) -and
    ((Get-Item $MergedProfile).Length -gt 0)
) {

    Write-OK "Merged profile already exists."

    $mergedTime = (Get-Item $MergedProfile).LastWriteTime
    $chatTime   = (Get-Item $ChatProfile).LastWriteTime
    $codeTime   = (Get-Item $CodeProfile).LastWriteTime

    if (
        $chatTime -gt $mergedTime -or
        $codeTime -gt $mergedTime
    ) {

        Write-Info `
            "Source profile is newer. Rebuilding merged profile."

        Remove-Item $MergedProfile -Force
    }
}

if (-not (Test-Path $MergedProfile)) {

    $chatLines = Get-Content $ChatProfile
    $codeLines = Get-Content $CodeProfile

    if ($chatLines.Count -eq 0) {
        throw "Chat profile is empty."
    }

    if ($codeLines.Count -eq 0) {
        throw "Code profile is empty."
    }

    $merged = New-Object `
        System.Collections.Generic.List[string]

    foreach ($line in $chatLines) {
        $merged.Add($line)
    }

    $startIndex = 0

    if ($codeLines[0] -eq $chatLines[0]) {
        $startIndex = 1
    }

    for (
        $i = $startIndex;
        $i -lt $codeLines.Count;
        $i++
    ) {

        $merged.Add($codeLines[$i])
    }

    $merged | Set-Content `
        -Path $MergedProfile `
        -Encoding UTF8

    Write-OK "Merged profile created."
}
else {

    Write-OK "Using existing merged profile."
}

Write-Info $MergedProfile

# ============================================================
# PHASE 9 - CACHE SLOT TESTS
# ============================================================

Write-Section "Phase 9 - Testing MoE cache slots"

Write-Info "Slots to test:"
Write-Host ($CacheSlotsToTest -join ", ")

# Load previous results if available.
$SlotResults = @()

if (Test-Path $SlotResultsFile) {

    try {

        $SlotResults = @(
            Import-Csv $SlotResultsFile
        )

        Write-Info `
            "Loaded $($SlotResults.Count) previous slot results."
    }
    catch {

        Write-Warn `
            "Could not read previous slot results. Starting fresh."

        $SlotResults = @()
    }
}

foreach ($slots in $CacheSlotsToTest) {

    # --------------------------------------------------------
    # Check if this slot count already succeeded in THIS
    # script's result file.
    # --------------------------------------------------------

    $previous = $SlotResults |
        Where-Object {
            ([int]$_.Slots -eq $slots) -and
            ($_.Status -eq "READY")
        } |
        Select-Object -First 1

    if ($null -ne $previous) {

        Write-OK `
            "$slots slots already tested successfully. Skipping."

        continue
    }

    Write-Section "Testing $slots MoE cache slots"

    Get-MemorySnapshot

    $slotLog = Join-Path `
        $LogDir `
        "cache-slots-$slots.log"

    $slotErrorLog = Join-Path `
        $LogDir `
        "cache-slots-$slots-error.log"

    Write-Info `
        "Starting server with $slots cache slots."

    $serverArgs = @(
        "-m", $Model,

        "--moe-cache-profile",
        $MergedProfile,

        "--moe-cache-slots",
        "$slots",

        "-ngl", "99",

        "-ncmoe", "99",

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
    Write-Host "$ServerExe $($serverArgs -join ' ')"

    $server = $null

    try {

        $server = Start-LoggedServer `
            -Exe $ServerExe `
            -Arguments $serverArgs `
            -StdOutFile $slotLog `
            -StdErrFile $slotErrorLog

        $ready = Wait-ForServer `
            -Process $server.Process `
            -TimeoutSeconds $ServerStartupTimeoutSeconds

        if ($ready) {

            Write-OK `
                "Server reached listening state with $slots slots."

            Get-MemorySnapshot

            # Remove any previous result for this slot.
            $SlotResults = @(
                $SlotResults |
                    Where-Object {
                        [int]$_.Slots -ne $slots
                    }
            )

            $SlotResults += [PSCustomObject]@{
                Slots  = $slots
                Status = "READY"
                Log    = $slotLog
                Error  = $slotErrorLog
            }
        }
        else {

            Write-Warn `
                "Server failed to reach listening state with $slots slots."

            # Remove old result for this slot.
            $SlotResults = @(
                $SlotResults |
                    Where-Object {
                        [int]$_.Slots -ne $slots
                    }
            )

            $SlotResults += [PSCustomObject]@{
                Slots  = $slots
                Status = "FAILED"
                Log    = $slotLog
                Error  = $slotErrorLog
            }

            Write-Warn "Check:"
            Write-Host $slotLog
            Write-Host $slotErrorLog
        }
    }
    finally {

        Stop-LoggedServer -Server $server
    }

    # Save after EVERY slot.
    #
    # If the machine crashes, the successful tests before the
    # crash remain recorded.

    Write-Host ""
    Write-Host "========== POST SERVER CLEANUP ==========" -ForegroundColor Cyan
    Write-Host "[DEBUG] Stop-LoggedServer returned successfully."
    Write-Host "[DEBUG] SlotResults count: $($SlotResults.Count)"
    Write-Host "[DEBUG] SlotResultsFile: $SlotResultsFile"
    Write-Host "[DEBUG] Starting Export-Csv..."

    try {

        $SlotResults |
            Sort-Object {
                [int]$_.Slots
            } |
            Export-Csv `
                -Path $SlotResultsFile `
                -NoTypeInformation `
                -ErrorAction Stop

        Write-Host "[DEBUG] Export-Csv completed successfully." -ForegroundColor Green
    }
    catch {

        Write-Host ""
        Write-Host "========== EXPORT-CSV ERROR ==========" -ForegroundColor Red
        Write-Host $_.Exception.ToString() -ForegroundColor Red
        Write-Host "=======================================" -ForegroundColor Red

        Read-Host "Export-Csv failed. Press ENTER"

        throw
    }

    Write-Host "[DEBUG] Starting shutdown wait: $ServerShutdownWaitSeconds seconds"

    Start-Sleep `
        -Seconds $ServerShutdownWaitSeconds

    Write-Host "[DEBUG] Shutdown wait completed."
    Write-Host "[DEBUG] End of slot iteration."
}

# ============================================================
# SLOT RESULTS
# ============================================================

Write-Section "MoE cache slot results"

if (Test-Path $SlotResultsFile) {

    $SlotResults = @(
        Import-Csv $SlotResultsFile
    )

    $SlotResults |
        Sort-Object {
            [int]$_.Slots
        } |
        Format-Table -AutoSize
}
else {

    Write-Warn "No slot results file exists."
}

# ============================================================
# PHASE 10 - BASELINE BENCHMARK
# ============================================================

Write-Section "Phase 10 - Baseline benchmark"

if (
    (Test-Path $BaselineLog) -and
    ((Get-Item $BaselineLog).Length -gt 0)
) {

    Write-OK `
        "Baseline benchmark log already exists. Skipping."

    Write-Info $BaselineLog
}
else {

    Write-Info "Running baseline benchmark."

    $baselineArgs = @(
        "-m", $Model,

        "-ngl", "99",

        "-ncmoe", "99",

        "-fa", "1",

        "-p", "$BenchmarkPromptTokens",

        "-n", "$BenchmarkGeneration",

        "-b", "$Batch",

        "-ub", "$UBatch",

        "-r", "$BenchmarkRuns"
    )

    Run-NativeLogged `
        -Exe $BenchExe `
        -Arguments $baselineArgs `
        -LogFile $BaselineLog

    Write-OK "Baseline benchmark completed."
}

# ============================================================
# PHASE 11 - OPTIMIZED BENCHMARK
# ============================================================

Write-Section "Phase 11 - Optimized benchmark"

$successful = @(
    $SlotResults |
        Where-Object {
            $_.Status -eq "READY"
        }
)

if ($successful.Count -eq 0) {

    Write-Warn `
        "No cache slot configuration reached server-ready state."

    Write-Warn `
        "Optimized benchmark will NOT be executed."
}
else {

    # Pick the highest configuration that actually reached
    # the server listening state.
    $bestSlots = (
        $successful |
            Sort-Object {
                [int]$_.Slots
            } -Descending |
            Select-Object -First 1
    ).Slots

    Write-OK `
        "Highest server-ready cache configuration: $bestSlots slots"

    # --------------------------------------------------------
    # Enable host registration and expert prefetch.
    # --------------------------------------------------------

    $env:GGML_CUDA_REGISTER_HOST = "1"
    $env:GGML_SCHED_PREFETCH_EXPERTS = "1"

    try {

        $OptimizedLog = Join-Path `
            $LogDir `
            "benchmark-optimized-$bestSlots.log"

        if (
            (Test-Path $OptimizedLog) -and
            ((Get-Item $OptimizedLog).Length -gt 0)
        ) {

            Write-OK `
                "Optimized benchmark already exists. Skipping."

            Write-Info $OptimizedLog
        }
        else {

            $optimizedArgs = @(
                "-m", $Model,

                "-ngl", "99",

                "-ncmoe", "99",

                "--moe-cache-profile",
                $MergedProfile,

                "--moe-cache-slots",
                "$bestSlots",

                "-fa", "1",

                "-p", "$BenchmarkPromptTokens",

                "-n", "$BenchmarkGeneration",

                "-b", "$Batch",

                "-ub", "$UBatch",

                "-r", "$BenchmarkRuns"
            )

            Write-Info `
                "Running optimized benchmark with $bestSlots slots."

            Run-NativeLogged `
                -Exe $BenchExe `
                -Arguments $optimizedArgs `
                -LogFile $OptimizedLog

            Write-OK "Optimized benchmark completed."
        }
    }
    finally {

        Remove-Item `
            Env:GGML_CUDA_REGISTER_HOST `
            -ErrorAction SilentlyContinue

        Remove-Item `
            Env:GGML_SCHED_PREFETCH_EXPERTS `
            -ErrorAction SilentlyContinue
    }
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
Write-Host "  $SlotResultsFile"
Write-Host "  $BaselineLog"

Write-Host ""
Write-Host "============================================================" `
    -ForegroundColor Green

Write-Host `
    " PROFILING / BENCHMARK COMPLETE" `
    -ForegroundColor Green

Write-Host "============================================================" `
    -ForegroundColor Green

Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "MTP was intentionally NOT enabled."
Write-Host ""
Write-Host "The cache-slot test now considers a configuration successful"
Write-Host "only after llama-server actually listens on port 8080."
Write-Host ""
Write-Host "Results directory:" -ForegroundColor Cyan
Write-Host $ProfileDir -ForegroundColor White
Write-Host ""
Write-Host ""
Write-Host "Script finished. Press ENTER to close." -ForegroundColor Green
Read-Host