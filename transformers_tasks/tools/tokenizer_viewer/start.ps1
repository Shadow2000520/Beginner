param([int]$Port = 8001)
$viewerDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& python -m streamlit run (Join-Path $viewerDir 'web_ui.py') --server.port $Port
