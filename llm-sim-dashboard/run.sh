#!/usr/bin/env bash
# Launch the dashboard using the py313 conda env (streamlit + openai).
# The simulation itself runs in JMBC_PYTHON (jax313) — see settings.py / .env.
set -euo pipefail
cd "$(dirname "$0")"

PY="${DASHBOARD_PYTHON:-$HOME/miniconda3/envs/py313/bin/python}"

exec "$PY" -m streamlit run app.py "$@"
