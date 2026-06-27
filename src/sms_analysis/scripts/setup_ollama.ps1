# Install Ollama. Model pulling is handled by sms-analyze after this script returns.
# Safe to run multiple times (idempotent).
#Requires -Version 5.1

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
    # Refresh PATH for this session so 'ollama' is found without restarting.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "==> Ollama is installed."
