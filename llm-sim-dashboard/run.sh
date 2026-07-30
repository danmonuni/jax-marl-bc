#!/usr/bin/env bash
# Launch the dashboard (needs streamlit + an LLM client). Override the
# interpreter with DASHBOARD_PYTHON; the simulation runs in JMBC_PYTHON
# — see settings.py / .env.example.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${DASHBOARD_PYTHON:-}" ]; then
  PY="$DASHBOARD_PYTHON"
elif [ -x "../.venv/bin/python" ]; then
  PY="../.venv/bin/python"
else
  PY="python3"
fi

exec "$PY" -m streamlit run app.py "$@"
