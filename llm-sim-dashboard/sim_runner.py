"""The CLI/execution server that actually runs the JAX framework.

``run_stream`` launches ``python -m jmbc.run`` in the simulation interpreter as
a subprocess, streaming its stdout line by line so the dashboard can show live
logs, then reports the output directory(ies) the run produced.

It is deliberately a subprocess boundary, not an in-process import: JAX lives in
a different interpreter/env than Streamlit on this machine, and shelling out
also isolates long, memory-heavy runs from the UI process.

Usable standalone as a tiny CLI server too::

    python sim_runner.py '{"experiment": "rbc", "total_timesteps": 20000, "device": "cpu"}'
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional

import config_gen
import settings
import sim_spec


@dataclass
class RunResult:
    run_name: str
    argv: List[str]
    cmd: List[str]
    config_path: Optional[Path] = None
    config_yaml: str = ""
    base_driver: Optional[str] = None
    returncode: Optional[int] = None
    run_dirs: List[Path] = field(default_factory=list)
    base_hint: Optional[str] = None
    log: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and bool(self.run_dirs)


def make_run_name(params: Optional[Dict[str, Any]] = None) -> str:
    """A human-readable, filesystem-safe run id used as ``log.run_name`` (and so
    as the output directory name), e.g.::

        ks_50agents_20260723-143052
        rbc_20260723-143052
        general_9agents_20260723-143052

    Built from the experiment + agent count + a timestamp (the timestamp keeps
    same-day repeats from colliding). RBC is single-agent so the count is
    omitted. This name is intentionally distinct from the generated config file
    name (which carries the ``dash-`` prefix, see :func:`run_stream`).
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    if not params:
        return f"run_{ts}"
    exp = str(params.get("experiment", "run"))
    parts = [exp]
    n = params.get("n_agents")
    if n and not exp.startswith("rbc"):
        parts.append(f"{int(n)}agents")
    parts.append(ts)
    return "_".join(parts)


def build_command(argv: List[str]) -> List[str]:
    return [settings.JMBC_PYTHON, "-m", "jmbc.run", *argv]


def preflight() -> List[str]:
    """Return a list of human-readable problems, empty if good to go."""
    problems: List[str] = []
    if not settings.JMBC_REPO.exists():
        problems.append(f"jmbc repo not found: {settings.JMBC_REPO}")
    if not Path(settings.JMBC_PYTHON).exists():
        problems.append(f"simulation python not found: {settings.JMBC_PYTHON}")
    if not (settings.JMBC_REPO / "jmbc" / "run.py").exists():
        problems.append(f"jmbc/run.py missing under {settings.JMBC_REPO}")
    if not settings.EXP_DIR.exists():
        problems.append(f"configs/exp not found: {settings.EXP_DIR}")
    return problems


def run_stream(
    params: Dict[str, Any],
    run_name: Optional[str] = None,
    on_line: Optional[Callable[[str], None]] = None,
) -> RunResult:
    """Run a simulation, streaming stdout to ``on_line`` as it arrives.

    Blocks until the subprocess exits. Returns a populated :class:`RunResult`.
    """
    run_name = run_name or make_run_name(params)

    # Two identities, on purpose:
    #  * run_name        -> log.run_name -> the OUTPUT dir (meaningful, shown to
    #                       the user), e.g. ks_50agents_20260723-143052.
    #  * gen_name        -> the generated configs/exp/<gen_name>.yaml + `exp=`
    #                       selector. Carries the `dash-` prefix so it stays out
    #                       of the base-template list and never clashes with a
    #                       real experiment name.
    gen_name = config_gen.GEN_PREFIX + run_name
    gen = config_gen.write_config(params, gen_name)
    argv = [f"exp={gen_name}", f"log.run_name={run_name}"]
    cmd = build_command(argv)
    result = RunResult(run_name=run_name, argv=argv, cmd=cmd,
                       config_path=gen.path, config_yaml=gen.yaml_text,
                       base_driver=gen.base_driver)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(settings.JMBC_REPO),   # so `runs/` lands in the repo
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        result.log.append(line)
        base = sim_spec.parse_run_base(line)
        if base and result.base_hint is None:
            result.base_hint = base
        if on_line is not None:
            on_line(line)
    proc.wait()
    result.returncode = proc.returncode
    result.duration_s = time.perf_counter() - t0
    result.run_dirs = sim_spec.discover_run_dirs(
        run_name, exp_hint=gen.base_driver, base_hint=result.base_hint
    )
    return result


# ── standalone CLI ───────────────────────────────────────────────────────────
def _main(argv: List[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        print('usage: python sim_runner.py \'{"experiment": "rbc", ...}\'')
        return 2
    try:
        params = json.loads(argv[0])
    except json.JSONDecodeError as e:
        print(f"invalid JSON: {e}", file=sys.stderr)
        return 2

    problems = preflight()
    if problems:
        print("preflight failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 3

    res = run_stream(params, on_line=lambda l: print(l, flush=True))
    print("\n=== run finished ===")
    print(f"config file : {res.config_path}")
    print(f"return code : {res.returncode}")
    print(f"duration    : {res.duration_s:.1f}s")
    print(f"output dirs : {[str(d) for d in res.run_dirs] or 'NONE FOUND'}")
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
