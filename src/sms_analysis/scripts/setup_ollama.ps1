# Install Ollama and pull the embedding model used for semantic clustering.
# Run from PowerShell: .\scripts\setup_ollama.ps1
# Safe to run multiple times (idempotent).
#Requires -Version 5.1

if ($env:SMS_OLLAMA_MODEL) { $Model = $env:SMS_OLLAMA_MODEL } else { $Model = "nomic-embed-text" }

Write-Host "==> Checking Ollama installation..."
$OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"
if ((Get-Command ollama -ErrorAction SilentlyContinue) -or (Test-Path $OllamaExe)) {
    Write-Host "==> Ollama already installed."
} else {
    Write-Host "==> Downloading Ollama installer..."
    $Installer = "$env:TEMP\OllamaSetup.exe"
    Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile $Installer
    Write-Host "==> Running installer (follow the prompts)..."
    Start-Process -FilePath $Installer -Wait
    Remove-Item $Installer
    # Refresh PATH for this session so 'ollama' is found without restarting.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "==> Pulling model: $Model"
ollama pull $Model

Write-Host ""
Write-Host "Setup complete. Run the clustering analysis:"
Write-Host "  sms-analyze clustering"
