"""JaxMARL-BC LLM control dashboard.

A Streamlit front end where you describe a macroeconomic RL experiment in plain
English; an LLM (OpenAI or Ollama) translates it into JaxMARL-BC parameters and
calls the framework via a CLI runner; logs stream live; and the run's figures,
metrics, diagnostics and on-disk artifacts are shown with a download button.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config_gen
import llm_agent
import results as R
import settings
import sim_runner
import sim_spec

st.set_page_config(page_title="JaxMARL-BC LLM Dashboard", page_icon="🧮",
                   layout="wide")

# ── session state ────────────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("reply", None)         # last AgentReply
ss.setdefault("params_json", "")     # editable params (JSON string)
ss.setdefault("result", None)        # last RunResult
ss.setdefault("history", [])         # chat turns for the LLM


# ── sidebar: configuration & status ──────────────────────────────────────────
def sidebar() -> str:
    st.sidebar.title("⚙️ Configuration")

    provider = st.sidebar.radio(
        "LLM provider", ["openai", "ollama"],
        index=0 if settings.LLM_PROVIDER == "openai" else 1,
        help="Toggle between the OpenAI API (uses OPENAI_API_KEY) and a local "
             "Ollama server.",
    )

    s = settings.summary()
    if provider == "openai":
        st.sidebar.caption(f"Model: `{s['openai_model']}`")
        if s["openai_key_set"]:
            st.sidebar.success("OPENAI_API_KEY detected", icon="✅")
        else:
            st.sidebar.error("OPENAI_API_KEY missing — set it in .env", icon="🚫")
    else:
        st.sidebar.caption(f"Model: `{s['ollama_model']}`  ·  {s['ollama_host']}")

    st.sidebar.divider()
    st.sidebar.subheader("Simulation backend")
    _status_line("jmbc repo", s["repo"], s["repo_exists"])
    _status_line("sim python", s["sim_python"], s["sim_python_exists"])
    st.sidebar.caption(f"Runs stored in: `{s['runs_dir']}`")

    problems = sim_runner.preflight()
    if problems:
        st.sidebar.error("Preflight problems:\n\n- " + "\n- ".join(problems))

    st.sidebar.divider()
    st.sidebar.caption("Experiments available: "
                       + ", ".join(f"`{e}`" for e in sim_spec.experiment_choices()))
    return provider


def _status_line(label: str, value: str, ok: bool) -> None:
    icon = "✅" if ok else "🚫"
    st.sidebar.markdown(f"{icon} **{label}**")
    st.sidebar.code(value, language=None)


# ── step 1: natural language -> parameters ───────────────────────────────────
def prompt_section(provider: str) -> None:
    st.subheader("1 · Describe the experiment")
    with st.form("prompt_form", clear_on_submit=False):
        request = st.text_area(
            "What should we simulate?",
            placeholder="e.g. Run a quick Krusell-Smith model with 50 agents on "
                        "CPU for a short training budget.",
            height=90,
        )
        submitted = st.form_submit_button("🧠 Interpret with LLM", type="primary")

    ex = st.expander("Example prompts", expanded=False)
    ex.markdown(
        "- *Run a quick RBC experiment on CPU as a smoke test.*\n"
        "- *Train Krusell-Smith with 200 agents, 100k timesteps, on GPU.*\n"
        "- *Do a fast heterogeneous (general) run with a bigger network [128,128].*\n"
        "- *Krusell-Smith, 20 agents, 30000 steps, and 32 envs.*"
    )

    if submitted and request.strip():
        with st.spinner(f"Asking {provider} to configure the run..."):
            reply = llm_agent.translate(
                request, provider=provider, history=ss.history[-6:]
            )
        ss.reply = reply
        ss.history.append({"role": "user", "content": request})
        if reply.kind == "run" and reply.params is not None:
            ss.params_json = json.dumps(reply.params, indent=2)
            ss.history.append(
                {"role": "assistant",
                 "content": "Configured run: " + " ".join(reply.dotlist)})
        elif reply.kind == "message":
            ss.history.append({"role": "assistant", "content": reply.text})

    _render_reply()


def _render_reply() -> None:
    reply = ss.reply
    if reply is None:
        return
    if reply.kind == "error":
        st.error(reply.text)
        return
    if reply.kind == "message":
        st.info(reply.text)
        st.caption(f"{reply.provider} · {reply.model} — no run configured. "
                   "Refine your request to launch a simulation.")
        return

    # kind == "run"
    st.success(f"{reply.provider} · {reply.model} translated your request into "
               "a new experiment config.")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("**Generated `configs/exp/…yaml` (preview)**")
        try:
            params = sim_spec.normalize(json.loads(ss.params_json))
            yaml_preview = config_gen.preview_yaml(params, "dash-<run_id>")
            st.code(yaml_preview, language="yaml")
            st.caption("Runs as: "
                       "`python -m jmbc.run exp=dash-<run_name> "
                       "log.run_name=<run_name>` — the output dir is the "
                       "meaningful `<run_name>` (e.g. `ks_50agents_DATE`).")
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Invalid parameters: {e}")
    with col_b:
        st.markdown("**Parameters (editable)**")
        ss.params_json = st.text_area(
            "params", value=ss.params_json, height=260,
            label_visibility="collapsed",
            help="Edit before running; the YAML preview updates on the next "
                 "interaction.",
        )


# ── step 2: run (with live logs) ──────────────────────────────────────────────
def _log_html(log_lines: list[str], height: int = 320) -> str:
    """Fixed-height terminal-style log pane that auto-scrolls to the bottom."""
    body = escape("\n".join(log_lines))
    return f"""
    <div id="logpane" style="height:{height}px;overflow-y:auto;background:#0e1117;
         color:#d7dae0;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
         font-size:12px;line-height:1.4;white-space:pre-wrap;word-break:break-word;
         padding:10px 12px;border-radius:8px;border:1px solid #2a2f3a;">{body}</div>
    <script>
      var el = document.getElementById('logpane');
      if (el) {{ el.scrollTop = el.scrollHeight; }}
    </script>
    """


def run_section() -> None:
    reply = ss.reply
    can_run = reply is not None and reply.kind == "run" and ss.params_json.strip()
    st.subheader("2 · Run the simulation")

    problems = sim_runner.preflight()
    if problems:
        st.warning("Cannot run until the backend is reachable (see sidebar).")

    run_clicked = st.button("▶️ Run simulation", type="primary",
                            disabled=not can_run or bool(problems))
    if not run_clicked:
        return

    try:
        params = sim_spec.normalize(json.loads(ss.params_json))
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Invalid parameters: {e}")
        return

    run_name = sim_runner.make_run_name(params)
    st.caption(f"Run id: `{run_name}`")
    # Fixed-height log pane that auto-scrolls to the bottom as output arrives.
    # Rendered via an HTML component (iframe) so a tiny bit of JS can pin the
    # scroll position to the newest line — a plain container keeps scroll at the
    # top when its content is replaced.
    log_box = st.empty()
    lines: list[str] = []

    def on_line(line: str) -> None:
        lines.append(line)
        log_box.empty()
        with log_box:
            components.html(_log_html(lines[-400:]), height=340)

    with st.status("Running JaxMARL-BC…", expanded=True) as status:
        result = sim_runner.run_stream(params, run_name=run_name, on_line=on_line)
        if result.config_path:
            st.caption(f"Generated config → `{result.config_path}`")
        if result.ok:
            status.update(label=f"Done in {result.duration_s:.1f}s "
                                f"→ {len(result.run_dirs)} output dir(s)",
                          state="complete")
        else:
            status.update(
                label=f"Finished with return code {result.returncode} "
                      f"(no output found)" if not result.run_dirs
                      else f"Return code {result.returncode}",
                state="error")
    ss.result = result


# ── step 3: results ──────────────────────────────────────────────────────────
def results_section() -> None:
    result = ss.result
    if result is None:
        st.info("Run a simulation to see results here, "
                "or open the **📂 Browse runs** tab to load a past run.")
        return

    st.subheader("3 · Results")
    if not result.run_dirs:
        st.error("The run produced no discoverable output directory. "
                 "Check the logs above.")
        with st.expander("Raw command"):
            st.code(" ".join(result.cmd))
        return

    if len(result.run_dirs) == 1:
        _render_run(result.run_dirs[0])
    else:
        tabs = st.tabs([d.name for d in result.run_dirs])
        for tab, d in zip(tabs, result.run_dirs):
            with tab:
                _render_run(d)


def _render_run(run_dir: Path, key_prefix: str = "") -> None:
    art = R.load(run_dir)

    # ── store location + download ──
    st.markdown("#### 📁 Stored data")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.code(str(run_dir), language=None)
        files = sorted(p.name for p in run_dir.iterdir())
        st.caption("Files: " + ", ".join(files))
    with c2:
        try:
            st.download_button(
                "⬇️ Download run (.zip)",
                data=R.zip_run_dir(run_dir),
                file_name=f"{run_dir.name}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{key_prefix}dl_{run_dir.name}",
            )
        except Exception as e:
            st.caption(f"zip failed: {e}")

    # ── headline metrics ──
    if art.timing:
        st.markdown("#### ⏱️ Timing")
        cols = st.columns(4)
        t = art.timing
        cols[0].metric("Wall time", _fmt_s(t.get("wall_time_s")))
        cols[1].metric("Run time", _fmt_s(t.get("run_time_s")))
        thr = t.get("throughput_steps_per_s")
        cols[2].metric("Throughput", f"{thr:,.0f} steps/s" if thr else "—")
        cols[3].metric("Device", str(t.get("device", "—")))

    final = R.final_diagnostics_summary(art.diagnostics)
    if final:
        st.markdown("#### 📌 Final-snapshot diagnostics")
        keys = list(final.keys())[:8]
        cols = st.columns(min(4, len(keys)) or 1)
        for i, k in enumerate(keys):
            v = final[k]
            cols[i % len(cols)].metric(k.split(".")[-1], f"{v:.4g}")

    # ── figures ──
    if art.figures:
        st.markdown("#### 🖼️ Figures")
        fig_cols = st.columns(2)
        for i, fig in enumerate(art.figures):
            with fig_cols[i % 2]:
                st.image(str(fig), caption=fig.name, use_container_width=True)

    # ── training metrics ──
    if art.metrics is not None and not art.metrics.empty:
        st.markdown("#### 📈 Training metrics")
        df = art.metrics
        numeric = [c for c in df.columns if c != "update"
                   and pd.api.types.is_numeric_dtype(df[c])]
        default = [c for c in ("total_loss", "value_loss", "actor_loss",
                               "approx_kl", "entropy") if c in numeric][:3]
        picked = st.multiselect("Series to plot", numeric,
                                default=default or numeric[:3],
                                key=f"{key_prefix}ms_{run_dir.name}")
        xcol = "update" if "update" in df.columns else df.columns[0]
        if picked:
            st.line_chart(df.set_index(xcol)[picked])
        with st.expander("metrics.csv (table)"):
            st.dataframe(df, use_container_width=True, height=280)

    # ── diagnostics table ──
    diag_df = R.diagnostics_rows(art.diagnostics)
    if diag_df is not None and not diag_df.empty:
        st.markdown("#### 🔬 Diagnostics across snapshots")
        st.dataframe(diag_df, use_container_width=True, height=260)
        num = [c for c in diag_df.columns if c != "update_idx"
               and pd.api.types.is_numeric_dtype(diag_df[c])]

    # ── config ──
    if art.config_text:
        with st.expander("Resolved config.yaml"):
            st.code(art.config_text, language="yaml")
    if art.rollouts_bytes:
        st.caption(f"rollouts.npz (raw record): {R.human_size(art.rollouts_bytes)}")


def browse_page() -> None:
    st.subheader("📂 Browse existing runs on disk")
    runs_root = settings.RUNS_DIR
    st.caption(f"Runs directory: `{runs_root}`")
    if not runs_root.exists():
        st.info("No runs directory yet — launch a simulation from the "
                "**🚀 Run a simulation** tab first.")
        return
    dirs = R.discover_runs(runs_root)
    if not dirs:
        st.info("No runs found yet.")
        return
    labels = [str(d.relative_to(runs_root)) for d in dirs[:200]]
    choice = st.selectbox(f"Pick a run ({len(dirs)} found, newest first)",
                          labels, key="browse_pick")
    if choice:
        st.divider()
        _render_run(dirs[labels.index(choice)], key_prefix="browse_")


# ── helpers ──
def _fmt_s(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f} s"
    except (TypeError, ValueError):
        return str(v)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🧮 JaxMARL-BC — LLM Simulation Dashboard")
    st.caption("Describe a macroeconomic RL experiment in plain language. An LLM "
               "translates it into JaxMARL-BC parameters and runs the JAX "
               "framework; logs, figures and data appear below.")
    provider = sidebar()
    tab_run, tab_browse = st.tabs(["🚀 Run a simulation", "📂 Browse runs"])
    with tab_run:
        prompt_section(provider)
        st.divider()
        run_section()
        st.divider()
        results_section()
    with tab_browse:
        browse_page()


main()
