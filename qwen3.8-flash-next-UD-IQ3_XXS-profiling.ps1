#requires -Version 5.1

$ErrorActionPreference = "Stop"

# ============================================================
# CONFIG
# ============================================================

$Root = "C:\LLM"

$RepoDir = Join-Path $Root "llama.cpp-thecodacus"
$ProfileDir = Join-Path $Root "qwen38-profile"
$LogDir = Join-Path $ProfileDir "logs"

$ModelDir = Join-Path $Root "QWEN3.8-flash-next-UD-IQ3_XXS"

$Model = Join-Path $ModelDir `
    "Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"

$RepoURL = "https://github.com/thecodacus/llama.cpp.git"
$Branch = "perf"

$SlotsToTest = @(24,32,40,48,56,64)

# ============================================================
# HELPERS
# ============================================================

function Section($text) {
    Write-Host ""
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host $text -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
}

function Find-Exe($base,$name) {

    $r = Get-ChildItem `
        -Path $base `
        -Recurse `
        -Filter $name `
        -File `
        -ErrorAction SilentlyContinue |
        Select-Object -First 1

    if ($null -eq $r) {
        throw "$name not found"
    }

    return $r.FullName
}

function Run-Logged {
    param(
        [string]$Exe,
        [string[]]$Args,
        [string]$LogFile
    )

    Write-Host ""
    Write-Host "Running:" -ForegroundColor Cyan
    Write-Host $Exe -ForegroundColor DarkGray
    Write-Host ($Args -join " ") -ForegroundColor DarkGray
    Write-Host ""

    # Build command line safely for Windows native executable.
    $commandLine = ""

    foreach ($arg in $Args) {

        if ($arg -match '[\s"]') {
            $escaped = $arg.Replace('\', '\\').Replace('"', '\"')
            $commandLine += '"' + $escaped + '" '
        }
        else {
            $commandLine += $arg + " "
        }
    }

    $psi = New-Object System.Diagnostics.ProcessStartInfo

    $psi.FileName = $Exe
    $psi.Arguments = $commandLine.Trim()
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi

    $stdout = New-Object System.Text.StringBuilder
    $stderr = New-Object System.Text.StringBuilder

    $process.add_OutputDataReceived({
        param($sender, $e)

        if ($null -ne $e.Data) {
            [void]$stdout.AppendLine($e.Data)
            Write-Host $e.Data
        }
    })

    $process.add_ErrorDataReceived({
        param($sender, $e)

        if ($null -ne $e.Data) {
            [void]$stderr.AppendLine($e.Data)
            Write-Host $e.Data
        }
    })

    if (-not $process.Start()) {
        throw "Could not start process: $Exe"
    }

    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()

    $process.WaitForExit()

    Start-Sleep -Milliseconds 200

    $logContent = @"
============================================================
COMMAND
============================================================

$Exe $($Args -join " ")

============================================================
STDOUT
============================================================

$stdout

============================================================
STDERR
============================================================

$stderr
"@

    Set-Content `
        -Path $LogFile `
        -Value $logContent `
        -Encoding UTF8

    $exitCode = $process.ExitCode

    Write-Host ""
    Write-Host "Exit code: $exitCode" -ForegroundColor Cyan
    Write-Host "Log: $LogFile" -ForegroundColor DarkGray

    if ($exitCode -ne 0) {
        throw "Process failed with exit code $exitCode"
    }

    Write-Host "Process completed successfully." -ForegroundColor Green
}

# ============================================================
# DIRECTORIES
# ============================================================

New-Item -ItemType Directory -Force -Path $Root | Out-Null
New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ============================================================
# CHECK TOOLS
# ============================================================

Section "Checking tools"

git --version
cmake --version
nvcc --version
nvidia-smi

# ============================================================
# REPO
# ============================================================

Section "Repository"

if (-not (Test-Path $RepoDir)) {

    git clone `
        --branch $Branch `
        --single-branch `
        $RepoURL `
        $RepoDir
}
else {

    Push-Location $RepoDir

    git fetch
    git checkout $Branch
    git pull

    Pop-Location
}

# ============================================================
# BUILD
# ============================================================

Section "Build"

$BuildDir = Join-Path $RepoDir "build"

if (Test-Path $BuildDir) {
    Remove-Item $BuildDir -Recurse -Force
}

cmake `
    -S $RepoDir `
    -B $BuildDir `
    -DGGML_CUDA=ON `
    -DCMAKE_BUILD_TYPE=Release

cmake `
    --build $BuildDir `
    --config Release `
    -j 16

# ============================================================
# EXES
# ============================================================

Section "Executables"

$ServerExe = Find-Exe $BuildDir "llama-server.exe"
$TraceExe = Find-Exe $BuildDir "llama-moe-trace.exe"
$BenchExe = Find-Exe $BuildDir "llama-bench.exe"

Write-Host $ServerExe
Write-Host $TraceExe
Write-Host $BenchExe

# ============================================================
# MODEL CHECK
# ============================================================

Section "Model"

Get-ChildItem $ModelDir *.gguf

if (-not (Test-Path $Model)) {
    throw "Model not found"
}

# ============================================================
# HELP
# ============================================================

Section "Checking fork options"

& $ServerExe --help |
    Tee-Object "$LogDir\server-help.txt"

& $TraceExe --help |
    Tee-Object "$LogDir\trace-help.txt"

# ============================================================
# CHAT PROFILE
# ============================================================

Section "Chat profile"

$ChatCSV = Join-Path $ProfileDir "chat.csv"

$env:MOE_TRACE_OUT = $ChatCSV

$chatPrompt = "Explain local LLM architecture, privacy, enterprise deployment, performance and scalability."

$chatArgs = @(
    "-m",$Model,
    "-ngl","99",
    "-ncmoe","99",
    "-fa","1",
    "-c","4096",
    "-n","512",
    "-p",$chatPrompt
)

Run-Logged `
    $TraceExe `
    $chatArgs `
    "$LogDir\chat.log"

Remove-Item Env:MOE_TRACE_OUT -ErrorAction SilentlyContinue

# ============================================================
# CODE PROFILE
# ============================================================

Section "Code profile"

$CodeCSV = Join-Path $ProfileDir "code.csv"

$env:MOE_TRACE_OUT = $CodeCSV

$codePrompt = "Write a Python REST API with authentication, database access and monitoring."

$codeArgs = @(
    "-m",$Model,
    "-ngl","99",
    "-ncmoe","99",
    "-fa","1",
    "-c","4096",
    "-n","512",
    "-p",$codePrompt
)

Run-Logged `
    $TraceExe `
    $codeArgs `
    "$LogDir\code.log"

Remove-Item Env:MOE_TRACE_OUT -ErrorAction SilentlyContinue

# ============================================================
# MERGE
# ============================================================

Section "Merge profiles"

$Merged = Join-Path $ProfileDir "merged.csv"

$chat = Get-Content $ChatCSV
$code = Get-Content $CodeCSV

$chat | Set-Content $Merged

if ($code.Count -gt 1) {

    $code[1..($code.Count-1)] |
        Add-Content $Merged
}

Write-Host $Merged

# ============================================================
# SLOT TESTS
# ============================================================

Section "Cache slots"

foreach ($slot in $SlotsToTest) {

    Write-Host ""
    Write-Host "Testing slot: $slot"

    $args = @(
        "-m",$Model,
        "--moe-cache-profile",$Merged,
        "--moe-cache-slots",$slot,
        "-ngl","99",
        "-ncmoe","99",
        "-fa","on",
        "-ctk","q8_0",
        "-ctv","q8_0",
        "-c","4096",
        "-b","512",
        "-ub","256",
        "-t","6",
        "--host","127.0.0.1",
        "--port","8080"
    )

    $proc = Start-Process `
        -FilePath $ServerExe `
        -ArgumentList $args `
        -PassThru `
        -WindowStyle Hidden

    Start-Sleep 20

    if (-not $proc.HasExited) {

        Write-Host "SUCCESS"

        Stop-Process `
            -Id $proc.Id `
            -Force
    }
    else {

        Write-Host "FAILED"
    }
}

# ============================================================
# END
# ============================================================

Section "COMPLETE"

Write-Host ""
Write-Host "Results:"
Write-Host $ProfileDir

Write-Host ""
Write-Host "Logs:"
Write-Host $LogDir