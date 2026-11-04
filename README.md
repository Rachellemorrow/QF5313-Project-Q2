# Question 2: Chinese futures Monte Carlo pricing and deep hedging
Python 3.13; fresh real IF/CSI300/SHIBOR snapshots, assignment GBM convention.
Set cache: `export UV_CACHE_DIR=.cache/uv`
Install frozen dependencies: `uv sync --frozen`
One-command full offline reproduction: `uv run python tools/reproduce.py --seed 42`
Data: `uv run python data/fetch.py --offline` (fresh retrieval: `--live`; help: `--help`).
Pricing: `uv run python code/approach1_pricing.py --seed 42`
Hedging: `uv run python code/approach2_hedging.py --seed 42`
Compare: `uv run python code/compare.py`; outputs in `results/`, figures inline Plotly.
Report is a group-review draft; see RESEARCH_PROTOCOL.md and SESSION.md for decisions/evidence.
