# Q2: independently rebuilt Monte Carlo pricing and CVaR hedging
Python 3.13; English report is a draft requiring student-group review.
`export UV_CACHE_DIR=.cache/uv XDG_CACHE_HOME="$PWD/.cache"`
`uv sync --frozen`
Fresh market retrieval: `uv run python data/fetch.py --fetch`; offline: `uv run python data/fetch.py --offline`.
`uv run pytest -q`
`uv run python code/approach1_pricing.py --seed 42`
`uv run python code/approach2_hedging.py --seed 42`
`uv run python code/compare.py`
Full offline reproduction: `uv run python tools/reproduce.py`; see `REBUILD_PROTOCOL.md` and `report.md`.
