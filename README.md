# Advanced Programming 2026 — Q2: MC pricing & hedging on CSI 300 futures
```bash
export UV_CACHE_DIR=.cache/uv
uv sync --frozen
uv run python data/fetch.py --verify-only   # offline; --refresh to re-download
uv run python code/approach1_pricing.py --seed 42
uv run python code/approach2_hedging.py --seed 42
uv run python code/compare.py
uv run pytest tests/ -q
```
Data/provenance: `data/metadata.json`. Figures: Plotly HTML in `results/`. Details: `report.md`.