# Contributing to Quant Finance Lab

Thank you for contributing to Quant Finance Lab!

Quant Finance Lab is an educational quantitative finance and mathematical exploration laboratory. We welcome contributions that expand models, improve computational efficiency, refine statistical methods, or clarify documentation.

---

## Guiding Principles

1. **Preserve Educational Identity**: The platform exists to explore mathematical and quantitative ideas, not to serve as a commercial signal service or automated trading bot.
2. **Mathematical Rigor**: Every new model or metric must include its mathematical definition, assumptions, derivations, and test coverage.
3. **Reproducibility**: Experiments must support seeded random number generation and produce consistent results.
4. **Clean Architecture**: Follow the package directory structure. Computation must remain strictly decoupled from visualization and CLI presentation.

---

## Development Setup

1. Clone or download the repository:
   ```bash
   git clone https://github.com/manishsai-rv/quant-finance-lab.git
   cd quant-finance-lab
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows PowerShell
   py -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install package with development and dashboard dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -e ".[all]"
   ```

4. Run the test suite:
   ```bash
   python -m pytest
   ```

---

## Submitting Pull Requests

* Ensure all tests pass (`python -m pytest`).
* Include tests for any new models, metrics, or validation functions.
* Document any new mathematical formulas in `docs/mathematical_foundations.md`.
* Follow PEP 8 style conventions and provide type hints on public interfaces.
