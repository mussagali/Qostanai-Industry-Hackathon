$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create Python environment' }
}
& '.venv/Scripts/python.exe' -c "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(name) is None for name in ['streamlit', 'pandas', 'numpy', 'plotly', 'sklearn', 'yaml']))"
if ($LASTEXITCODE -ne 0) {
    & '.venv/Scripts/python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
& '.venv/Scripts/python.exe' -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
