# Advanced Programming 2026 — Question 2: Monte Carlo pricing & hedging on Chinese futures

Monte Carlo pricing (NumPy + PyTorch) and deep hedging of Asian/barrier/digital options on CSI 300 futures (IF0).

**Run (offline-safe; snapshot committed in `data/clean/`)**

```bash
export UV_CACHE_DIR=.cache/uv
uv sync --frozen
uv run python data/fetch.py --verify-only       # or --refresh to re-download from AKShare
uv run python code/approach1_pricing.py --seed 42
uv run python code/approach2_hedging.py --seed 42
uv run python code/compare.py
uv run pytest tests/ -q
```

Data: IF0 main continuous (AKShare/Sina) + CSI 300 index + SHIBOR; provenance & SHA-256 in `data/metadata.json`. Figures: self-contained Plotly HTML in `results/`. Seed 42 default; report in `report.md`.
