#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
req_file=requirements.txt
if [ -f requirements.lock ]; then
  req_file=requirements.lock
fi
.venv/bin/python -c "import numpy,pandas,PIL,scipy,sklearn,joblib; assert sklearn.__version__ == '1.8.0'" 2>/dev/null || .venv/bin/python -m pip install -r "$req_file"
.venv/bin/python app.py
