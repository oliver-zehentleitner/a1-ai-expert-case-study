# Install Ollama. Model pulling is handled by sms-analyze after this script returns.
# Safe to run multiple times (idempotent).
#Requires -Version 5.1

# Already running? Skip everything -- Python will proceed directly to model check.
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 -ErrorAction Stop | Out-Null
    Write-Host "==> Ollama is already running."
    exit 0
} catch {}

# Already installed? Skip download -- Python will offer to start it.
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if ((Get-Command ollama -ErrorAction SilentlyContinue) -or (Test-Path $OllamaExe)) {
    Write-Host "==> Ollama is already installed."
    exit 0
}

# Fresh install: start without -Wait because OllamaSetup.exe never exits --
# it transitions into the running Ollama tray app after installation.
Write-Host "==> Downloading Ollama installer..."
$Installer = "$env:TEMP\OllamaSetup.exe"
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Installer
Write-Host "==> Starting Ollama installer -- please complete the setup wizard..."
Start-Process -FilePath $Installer

# Poll until Ollama becomes reachable (installer starts it when done).
Write-Host "==> Waiting for Ollama to start (up to 5 minutes)..."
for ($i = 1; $i -le 60; $i++) {
    Start-Sleep -Seconds 5
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 2 -ErrorAction Stop | Out-Null
        Write-Host "==> Ollama is running."
        exit 0
    } catch {}
    Write-Host "   still waiting... ($i / 60)"
}
Write-Host "==> Timed out -- sms-analyze will offer to start Ollama manually."
