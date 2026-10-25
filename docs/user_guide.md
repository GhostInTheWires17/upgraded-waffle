# User Guide: Quant Finance Lab

This guide covers running simulations, analyzing results, using the Python API, executing CLI commands, and launching the interactive dashboard.

---

## 1. Quick Start via Command-Line (CLI)

The CLI provides instant access to all core simulation and analysis modules:

```bash
# Run standard Monte Carlo simulation with 10,000 paths
python -m quant_finance_lab simulate --initial-capital 1000 --win-prob 0.60 --win-return 0.02 --loss-return -0.01 --simulations 10000 --trades 100 --seed 42

# Save publication-grade research plots and JSON metadata
python -m quant_finance_lab simulate --simulations 10000 --trades 100 --seed 42 --save-plots results/my_run --export-json results/my_run/summary.json

# Run 1D sensitivity parameter sweep across win probabilities
python -m quant_finance_lab sensitivity --param win_probability --start 0.40 --stop 0.70 --steps 7 --simulations 5000 --seed 42

# Run Pessimistic, Base-Case, and Optimistic scenario evaluation
python -m quant_finance_lab scenarios --capital 1000 --trades 100 --simulations 5000

# Launch interactive Streamlit research dashboard
python -m quant_finance_lab dashboard --port 8501
```

---

## 2. Python API Usage

Quant Finance Lab can be imported and scripted directly in Python or Jupyter notebooks:

### Basic Simulation
```python
from quant_finance_lab import run_simulation, print_console_report

result = run_simulation(
    initial_capital=1000.0,
    win_probability=0.60,
    win_return=0.02,
    loss_return=-0.01,
    num_simulations=10000,
    num_trades=100,
    seed=42,
)

print_console_report(result)
```

### Advanced Simulation with Custom Models & Costs
```python
from quant_finance_lab import (
    SimulationConfig,
    FixedReturnModel,
    TransactionCostModel,
    PositionSizer,
    MonteCarloSimulator,
)

# 1. Configure simulation parameters
config = SimulationConfig(
    initial_capital=5000.0,
    num_simulations=20000,
    num_trades=200,
    seed=101,
    ruin_threshold=2500.0,
)

# 2. Configure model with transaction friction and position sizing
model = FixedReturnModel(win_probability=0.58, win_return=0.025, loss_return=-0.015)
costs = TransactionCostModel(fixed_cost=1.00, proportional_cost=0.0005)  # 5 bps
sizer = PositionSizer(sizing_type="fixed_fractional", fraction=0.50)      # 50% capital per trade

# 3. Execute
sim = MonteCarloSimulator(config=config, model=model, cost_model=costs, position_sizer=sizer)
result = sim.run()

print(f"Mean Final Balance: ${result.distribution_metrics.mean:,.2f}")
print(f"Median Max Drawdown: {result.drawdown_metrics.median_max_drawdown:.1%}")
```

### Generating Visualizations
```python
from quant_finance_lab.visualization import (
    plot_equity_curves,
    plot_final_balance_distribution,
    plot_drawdown_distribution,
    plot_target_probabilities,
)

fig1 = plot_equity_curves(result, save_path="equity_curves.png")
fig2 = plot_final_balance_distribution(result, save_path="balance_hist.png")
fig3 = plot_drawdown_distribution(result, save_path="mdd_hist.png")
fig4 = plot_target_probabilities(result, save_path="targets.png")
```

---

## 3. Interactive Web Dashboard

To launch the graphical exploration dashboard:
```bash
python -m quant_finance_lab dashboard
# or
streamlit run src/quant_finance_lab/dashboard/app.py
```
Open `http://localhost:8501` in your browser.
