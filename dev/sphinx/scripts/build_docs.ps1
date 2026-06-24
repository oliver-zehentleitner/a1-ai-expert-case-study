# Build the Sphinx HTML documentation.
# Source: dev/sphinx/   Output: docs/   (committed and shipped)
#
# Usage:
#   .\dev\sphinx\scripts\build_docs.ps1           # build only
#   .\dev\sphinx\scripts\build_docs.ps1 --open    # build + open in default browser
#   .\dev\sphinx\scripts\build_docs.ps1 -W        # treat warnings as errors
[CmdletBinding()]
param(
    [switch]$Open,
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$SphinxArgs
)

$ErrorActionPreference = 'Stop'

$Root = Resolve-Path (Join-Path $PSScriptRoot '../../..')
$Open = $Open -or ($SphinxArgs -contains '--open') -or ($SphinxArgs -contains '-o')
$SphinxArgs = $SphinxArgs | Where-Object { $_ -notin '--open', '-o' }

# Stage README and assignment/ into the Sphinx source dir so myst-parser
# resolves relative links (./assignment/…) correctly during the build.
Copy-Item "$Root\README.md" "$Root\dev\sphinx\config\README.md" -Force
if (Test-Path "$Root\assignment") {
    Copy-Item "$Root\assignment" "$Root\dev\sphinx\config\assignment" -Recurse -Force
}

try {
    sphinx-build -b html "$Root\dev\sphinx\config" "$Root\docs" @SphinxArgs
} finally {
    # Clean up staged files — they are build-time only, not source-controlled.
    Remove-Item "$Root\dev\sphinx\config\README.md" -ErrorAction SilentlyContinue
    Remove-Item "$Root\dev\sphinx\config\assignment" -Recurse -ErrorAction SilentlyContinue
}

# Required for GitHub Pages to serve Sphinx's _static/ directory correctly.
New-Item "$Root\docs\.nojekyll" -ItemType File -Force | Out-Null

# Copy assignment files into docs/ so links work when browsing docs/index.html.
if (Test-Path "$Root\assignment") {
    Remove-Item "$Root\docs\assignment" -Recurse -ErrorAction SilentlyContinue
    Copy-Item "$Root\assignment" "$Root\docs\assignment" -Recurse -Force
}

Write-Host ""
Write-Host "Docs ready: docs/index.html"

if ($Open) {
    Start-Process "$Root\docs\index.html"
}
