"""Load the artifacts a jmbc run writes and shape them for the dashboard.

A run directory (``runs/<exp>/<run_id>/``) contains:
    config.yaml        resolved configuration
    metrics.csv        per-update training metrics
    diagnostics.json   economic + distributional probes across snapshots
    timing.json        wall time / throughput / device
    rollouts.npz       raw snapshot rollouts (large)
    figures/*.png      rendered figures

This module reads them defensively (any file may be absent) and exposes a
single :class:`RunArtifacts` plus a zip helper for the download button.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class RunArtifacts:
    run_dir: Path
    config_text: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[pd.DataFrame] = None
    diagnostics: Optional[Dict[str, Any]] = None
    timing: Optional[Dict[str, Any]] = None
    figures: List[Path] = field(default_factory=list)
    rollouts_bytes: int = 0

    @property
    def name(self) -> str:
        return self.run_dir.name


def load(run_dir: Path) -> RunArtifacts:
    run_dir = Path(run_dir)
    art = RunArtifacts(run_dir=run_dir)

    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        art.config_text = cfg_path.read_text()
        try:
            import yaml
            art.config = yaml.safe_load(art.config_text) or {}
        except Exception:
            art.config = {}

    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        try:
            art.metrics = pd.read_csv(metrics_path)
        except Exception:
            art.metrics = None

    diag_path = run_dir / "diagnostics.json"
    if diag_path.exists():
        try:
            art.diagnostics = json.loads(diag_path.read_text())
        except Exception:
            art.diagnostics = None

    timing_path = run_dir / "timing.json"
    if timing_path.exists():
        try:
            art.timing = json.loads(timing_path.read_text())
        except Exception:
            art.timing = None

    fig_dir = run_dir / "figures"
    if fig_dir.exists():
        art.figures = sorted(fig_dir.glob("*.png"))

    roll = run_dir / "rollouts.npz"
    if roll.exists():
        art.rollouts_bytes = roll.stat().st_size

    return art


def diagnostics_rows(diag: Optional[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """Flatten the per-snapshot diagnostics into a tidy table (best-effort).

    The schema is nested (snapshots -> economic/distributional -> metrics); we
    pull the scalar leaves so the dashboard can show a compact dataframe.
    """
    if not diag or "snapshots" not in diag:
        return None
    rows: List[Dict[str, Any]] = []
    for snap in diag.get("snapshots", []):
        row: Dict[str, Any] = {"update_idx": snap.get("update_idx")}
        for section in ("economic", "distributional"):
            sec = snap.get(section) or {}
            for k, v in _flatten(sec).items():
                if isinstance(v, (int, float)):
                    row[f"{section}.{k}"] = v
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    # Drop all-NaN columns to keep the table readable.
    return df.dropna(axis=1, how="all")


def _flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def final_diagnostics_summary(diag: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Scalar leaves of the final snapshot, for headline metric tiles."""
    if not diag:
        return {}
    final = diag.get("final") or {}
    flat: Dict[str, Any] = {}
    for section in ("economic", "distributional"):
        for k, v in _flatten(final.get(section) or {}).items():
            if isinstance(v, (int, float)):
                flat[f"{section}.{k}"] = v
    return flat


def zip_run_dir(run_dir: Path) -> bytes:
    """Zip an entire run directory in memory for a download button."""
    run_dir = Path(run_dir)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(run_dir.parent)))
    buf.seek(0)
    return buf.read()


def human_size(n: int) -> str:
    x = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if x < 1024 or unit == "GB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} GB"
