# `runs/`

Output directory — intentionally empty in the repository.

`python -m jmbc.run exp=<name>` writes one self-contained directory per run
here (`runs/<exp>/<run_id>/`: resolved config, metrics, diagnostics, timing,
raw rollouts and figures), and `python -m jmbc.analyze runs/<exp>/<run_id>`
rebuilds every figure and diagnostic from that record without retraining.

Nothing under `runs/` is a source file: the inputs to every experiment are
`configs/` (what to run) and `jmbc/` (how it runs), so this directory is
git-ignored and populated locally.
