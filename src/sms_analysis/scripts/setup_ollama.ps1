# Install Ollama. Model pulling is handled by sms-analyze after this script returns.
# Safe to run multiple times (idempotent).
#Requires -Version 5.1

# Already running? Skip everything — Python will proceed directly to model check.
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 -ErrorAction Stop | Out-Null
    Write-Host "==> Ollama is already running."
    exit 0
} catch {}

# Already installed? Skip download — Python will offer to start it.
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if ((Get-Command ollama -ErrorAction SilentlyContinue) -or (Test-Path $OllamaExe)) {
    Write-Host "==> Ollama is already installed."
    exit 0
}

# Fresh install.
Write-Host "==> Downloading Ollama installer..."
$Installer = "$env:TEMP\OllamaSetup.exe"
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Installer
Write-Host "==> Running installer (follow the prompts)..."
Start-Process -FilePath $Installer -Wait
Remove-Item $Installer
# Refresh PATH so 'ollama' is found without reopening the terminal.
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
            [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "==> Ollama installed."
