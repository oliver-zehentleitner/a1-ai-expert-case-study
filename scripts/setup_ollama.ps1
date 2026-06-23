# Install Ollama and pull the embedding model used for semantic clustering.
# Run from PowerShell: .\scripts\setup_ollama.ps1
# Safe to run multiple times (idempotent).
#Requires -Version 5.1

$Model = $env:SMS_OLLAMA_MODEL ?? "nomic-embed-text"

Write-Host "==> Checking Ollama installation..."
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "==> Ollama already installed: $(ollama --version 2>&1 | Select-Object -First 1)"
} else {
    Write-Host "==> Downloading Ollama installer..."
    $Installer = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Installer
    Write-Host "==> Running installer (follow the prompts)..."
    Start-Process -FilePath $Installer -Wait
    Remove-Item $Installer
}

Write-Host "==> Pulling model: $Model"
ollama pull $Model

Write-Host ""
Write-Host "Setup complete. Run the clustering analysis:"
Write-Host "  sms-analyze clustering"
