"""The contract between natural language and ``python -m jmbc.run``.

This module is the single source of truth for *what the LLM is allowed to set*.
It defines a flat list of parameters (each mapping to a dotted OmegaConf path
such as ``env.n_agents``), derives:

* a JSON Schema used as the OpenAI / Ollama tool ("function") definition, so the
  model translates a request into a validated ``run_simulation`` call;
* an argv builder that turns the validated parameters into the exact dotlist
  ``python -m jmbc.run exp=... key=value ...`` the framework expects.

Nothing here imports JAX — it only knows the shape of a run, so it is safe to
load in the dashboard interpreter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import settings


@dataclass(frozen=True)
class Param:
    name: str            # LLM-facing key
    path: str            # dotted OmegaConf override path (or "exp" selector)
    type: str            # json schema type: string|integer|number|boolean|array
    description: str
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None


def experiment_choices() -> List[str]:
    """Base experiment templates the LLM may derive from (configs/exp/*.yaml).

    Dashboard-generated configs (prefixed ``dash-``) are excluded — they are
    outputs of this tool, not templates to derive from.
    """
    if not settings.EXP_DIR.exists():
        return ["rbc", "ks", "ks_local", "general"]
    names = sorted(p.stem for p in settings.EXP_DIR.glob("*.yaml")
                   if not p.stem.startswith("dash-"))
    return names or ["rbc", "ks", "general"]


# ── Parameter catalogue ──────────────────────────────────────────────────────
# Only a curated, safe subset of the full config is exposed. Anything not here
# can still be reached through `extra_overrides` (a free-form dotlist), which is
# validated to be well-formed before use.
def build_params() -> List[Param]:
    exps = experiment_choices()
    return [
        Param("experiment", "exp", "string",
              "Base template to derive the new experiment config from. "
              "'rbc' = single-agent Real Business Cycle; 'ks' = Krusell-Smith "
              "heterogeneous-agent model; 'general' = heterogeneous RBC grid. "
              "The 'ks_local' variant is a ~5-minute CPU budget; "
              "ks_n20/n200/n2000 are population-scaling cells. Pick the closest "
              "template; the other parameters override it.",
              enum=exps),
        Param("n_agents", "env.n_agents", "integer",
              "Number of economic agents in the population (KS/general). RBC is "
              "single-agent. Larger = slower and more memory.",
              minimum=1, maximum=5000),
        Param("max_steps", "env.max_steps", "integer",
              "Episode length in environment steps.", minimum=10, maximum=20000),
        Param("total_timesteps", "train.total_timesteps", "integer",
              "Training budget in *sequential* env steps (total transitions = "
              "this x num_envs). This is the main knob for run duration: use "
              "10000-50000 for a quick demo, 100000+ for a serious run.",
              minimum=1000, maximum=2_000_000),
        Param("num_envs", "train.num_envs", "integer",
              "Parallel environments per update (batch width).",
              minimum=1, maximum=256),
        Param("num_minibatches", "train.num_minibatches", "integer",
              "PPO minibatches per update.", minimum=1, maximum=256),
        Param("update_epochs", "train.update_epochs", "integer",
              "PPO epochs per update.", minimum=1, maximum=64),
        Param("lr", "train.lr", "number",
              "Optimizer learning rate.", minimum=1e-6, maximum=1e-1),
        Param("gamma", "train.gamma", "number",
              "Discount factor.", minimum=0.0, maximum=0.9999),
        Param("hidden_dims", "net.hidden_dims", "array",
              "Actor-critic hidden layer sizes, e.g. [64, 64] or [128, 128]."),
        Param("n_snapshots", "diag.n_snapshots", "integer",
              "Number of training snapshots probed for diagnostics/figures.",
              minimum=1, maximum=500),
        Param("sim_steps", "diag.sim_steps", "integer",
              "Rollout length used for each diagnostic evaluation.",
              minimum=100, maximum=20000),
        Param("device", "run.device", "string",
              "Compute backend. Use 'cpu' for small local runs, 'auto' to let "
              "JAX choose, 'gpu' if a GPU is available.",
              enum=["auto", "cpu", "gpu"]),
        Param("seed", "run.seed", "integer",
              "Random seed.", minimum=0, maximum=2**31 - 1),
    ]


PARAMS: List[Param] = build_params()
PARAM_BY_NAME: Dict[str, Param] = {p.name: p for p in PARAMS}

# jmbc.run prints:  ... -> runs/<exp>/<run_id>
_DISPATCH_RE = re.compile(r"->\s*(\S+?/[^/\s]+/[^/\s]+)\s*$")


def tool_schema() -> dict:
    """JSON schema for the `run_simulation` tool given to the LLM."""
    props: Dict[str, dict] = {}
    for p in PARAMS:
        entry: Dict[str, Any] = {"description": p.description}
        if p.type == "array":
            entry["type"] = "array"
            entry["items"] = {"type": "integer"}
        else:
            entry["type"] = p.type
        if p.enum is not None:
            entry["enum"] = p.enum
        if p.minimum is not None:
            entry["minimum"] = p.minimum
        if p.maximum is not None:
            entry["maximum"] = p.maximum
        props[p.name] = entry

    props["extra_overrides"] = {
        "type": "object",
        "description": "Escape hatch for any other jmbc config path not listed "
                       "above, as a flat map of dotted path -> value, e.g. "
                       "{\"env.alpha\": 0.36}. Use sparingly.",
        "additionalProperties": True,
    }
    return {
        "type": "function",
        "function": {
            "name": "run_simulation",
            "description": "Run a JaxMARL-BC macroeconomic RL simulation with "
                           "the given configuration. Only include parameters the "
                           "user asked about or that are needed; everything else "
                           "falls back to the framework defaults. Always set an "
                           "'experiment'.",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": ["experiment"],
            },
        },
    }


_DOTPATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def normalize(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and clean an LLM-produced parameter dict.

    Returns a dict safe to hand to :func:`to_argv`. Raises ``ValueError`` on
    anything malformed so the UI can surface it rather than shelling out with
    garbage.
    """
    if not isinstance(params, dict):
        raise ValueError("parameters must be an object")

    out: Dict[str, Any] = {}
    extra = params.get("extra_overrides") or {}

    exp = params.get("experiment")
    if not exp:
        raise ValueError("'experiment' is required")
    if exp not in experiment_choices():
        raise ValueError(
            f"unknown experiment {exp!r}; choose one of {experiment_choices()}"
        )
    out["experiment"] = exp

    for name, value in params.items():
        if name in ("experiment", "extra_overrides") or value is None:
            continue
        p = PARAM_BY_NAME.get(name)
        if p is None:
            raise ValueError(f"unknown parameter {name!r}")
        out[name] = _coerce(p, value)

    clean_extra: Dict[str, Any] = {}
    if extra:
        if not isinstance(extra, dict):
            raise ValueError("extra_overrides must be an object")
        for path, value in extra.items():
            if not _DOTPATH_RE.match(str(path)):
                raise ValueError(f"invalid override path {path!r}")
            clean_extra[str(path)] = value
    out["extra_overrides"] = clean_extra
    return out


def _coerce(p: Param, value: Any) -> Any:
    try:
        if p.type == "integer":
            value = int(value)
        elif p.type == "number":
            value = float(value)
        elif p.type == "boolean":
            value = bool(value)
        elif p.type == "array":
            if isinstance(value, str):
                value = [int(x) for x in re.findall(r"-?\d+", value)]
            value = [int(x) for x in value]
            if not value:
                raise ValueError("empty array")
    except (TypeError, ValueError) as e:
        raise ValueError(f"{p.name}: cannot interpret {value!r} as {p.type} ({e})")

    if p.enum is not None and value not in p.enum:
        raise ValueError(f"{p.name}: {value!r} not in {p.enum}")
    if p.minimum is not None and value < p.minimum:
        raise ValueError(f"{p.name}: {value} < min {p.minimum}")
    if p.maximum is not None and value > p.maximum:
        raise ValueError(f"{p.name}: {value} > max {p.maximum}")
    return value


def to_dotlist(params: Dict[str, Any]) -> List[str]:
    """Ordered dotlist tokens (without the run_name), e.g. ['env.n_agents=20']."""
    params = normalize(params)
    tokens: List[str] = [f"exp={params['experiment']}"]
    for name, value in params.items():
        if name in ("experiment", "extra_overrides"):
            continue
        p = PARAM_BY_NAME[name]
        tokens.append(f"{p.path}={_fmt(value)}")
    for path, value in params.get("extra_overrides", {}).items():
        tokens.append(f"{path}={_fmt(value)}")
    return tokens


def _fmt(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(str(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def to_argv(params: Dict[str, Any], run_name: str) -> List[str]:
    """Full argv for ``python -m jmbc.run`` including our controlled run_name."""
    argv = to_dotlist(params)
    argv.append(f"log.run_name={run_name}")
    return argv


def parse_run_base(stdout_line: str) -> Optional[str]:
    """Extract the ``runs/<exp>/<run_id>`` base path jmbc.run announces."""
    m = _DISPATCH_RE.search(stdout_line.strip())
    return m.group(1) if m else None


def discover_run_dirs(run_name: str, exp_hint: Optional[str] = None,
                      base_hint: Optional[str] = None) -> List[Path]:
    """Find the output directory(ies) for a run.

    The RBC driver appends ``_textbook`` / ``_typical`` suffixes, so a single
    logical run can produce several directories that share the ``run_name``
    prefix. ``base_hint`` (parsed from stdout) pins the exact ``runs/<exp>``.
    """
    candidates: List[Path] = []
    search_roots: List[Path] = []
    if base_hint:
        # base_hint is like "runs/ks/<run_name>"; its parent is runs/<exp>.
        bp = (settings.JMBC_REPO / base_hint).resolve()
        search_roots.append(bp.parent)
    if exp_hint:
        search_roots.append(settings.RUNS_DIR / exp_hint)
    search_roots.append(settings.RUNS_DIR)

    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        for d in root.glob(f"**/{run_name}*"):
            if d.is_dir() and d.resolve() not in seen:
                seen.add(d.resolve())
                candidates.append(d)
    return sorted(candidates, key=lambda p: p.name)
