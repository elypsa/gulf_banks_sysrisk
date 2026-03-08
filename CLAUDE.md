# Project: GCC Bank SRISK Calculator

## Goals
- Calculate SRISK for Gulf region banks using LSEG data.
- Provide a modular, auditable pipeline for financial risk analysis.

## Tech Stack & Workflow
- **Environment**: Python managed via `uv`.
- **Execution**: Always use `uv run <script>.py` and `uv add <package>`.
- **Core Libraries**: `pandas`, `numpy`, `arch` (for GARCH), `scipy`.


## Coding Standards
- **Style**: Functional and modular. Avoid monolithic scripts.
- **Documentation**: Google-style docstrings with LaTeX for mathematical formulas.
- **Precision**: Use `numpy` for vectorization; handle NaN values (common in regional market holidays) using forward-fill/interpolation.
- **Variables**: Use clear financial naming conventions (e.g., `market_cap`, `total_liabilities`, `prudential_ratio`).

## Financial Logic Defaults
- **Systemic Benchmark**: Primary: Calculate SRISK using the broad benchmark to align with standard literature. Use also Banking Sector Benchmarks to generate a "Financial Sector Sensitivity".
- **SRISK** guide is provided in @SRISK-Implementation-Guide.md


## Development Patterns
- **Refactoring**: Prefer modular `.py` files over `.ipynb` for core logic.
- **Data Persistence**: Save intermediate LSEG outputs to `data/raw/` (parquet format preferred) to avoid redundant API calls!
- **Error Handling**: Implement robust try-except blocks for LSEG connection timeouts.
