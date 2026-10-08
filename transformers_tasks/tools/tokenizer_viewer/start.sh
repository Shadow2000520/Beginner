#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
exec streamlit run web_ui.py --server.port "${1:-8001}"
