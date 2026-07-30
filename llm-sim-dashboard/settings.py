"""Central configuration for the JaxMARL LLM dashboard.

Everything is read from environment variables (optionally a local ``.env``),
so the dashboard can be pointed at a different repo, python interpreter or LLM
backend without touching code. See ``.env.example`` for the full list.

Two interpreters are involved on purpose:

* the *dashboard* interpreter (this process) needs ``streamlit`` + ``openai``;
* the *simulation* interpreter (``JMBC_PYTHON``) needs ``jax`` + ``jaxmarl`` +
  ``jmbc``.

Those two may be the same interpreter or two different ones — the runner shells
out rather than importing JAX in-process precisely so they can differ.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:  # dotenv is optional
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

# The jax-marl-bc repository (contains jmbc/, configs/, runs/). This dashboard
# lives inside the repo (repo/llm-sim-dashboard/), so the repo root is the
# parent directory by default; override with JMBC_REPO if it moves.
JMBC_REPO = Path(
    os.environ.get("JMBC_REPO", str(HERE.parent))
).resolve()

# Interpreter that has jax + jaxmarl + jmbc installed. Prefers the repo's own
# .venv, falls back to the interpreter running the dashboard; override with
# JMBC_PYTHON when the two live in separate environments.
_venv_python = JMBC_REPO / ".venv" / "bin" / "python"
JMBC_PYTHON = os.environ.get(
    "JMBC_PYTHON",
    str(_venv_python) if _venv_python.exists() else sys.executable,
)

CONFIGS_DIR = JMBC_REPO / "configs"
EXP_DIR = CONFIGS_DIR / "exp"
RUNS_DIR = JMBC_REPO / "runs"

# ── LLM provider ─────────────────────────────────────────────────────────────
# "openai" or "ollama". The dashboard exposes a toggle that overrides this.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "") or None

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")


def summary() -> dict:
    """Human-readable snapshot of the resolved configuration (for the UI)."""
    return {
        "repo": str(JMBC_REPO),
        "repo_exists": JMBC_REPO.exists(),
        "sim_python": JMBC_PYTHON,
        "sim_python_exists": Path(JMBC_PYTHON).exists(),
        "runs_dir": str(RUNS_DIR),
        "provider": LLM_PROVIDER,
        "openai_model": OPENAI_MODEL,
        "openai_key_set": bool(OPENAI_API_KEY),
        "ollama_host": OLLAMA_HOST,
        "ollama_model": OLLAMA_MODEL,
    }
