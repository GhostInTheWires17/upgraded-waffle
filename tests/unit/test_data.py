"""Unit tests for historical data loading and returns calculation."""

import pandas as pd
import pytest
from quant_finance_lab.core.exceptions import DataError
from quant_finance_lab.data.validation import validate_ohlcv_dataframe
from quant_finance_lab.data.returns import compute_price_returns, summarize_empirical_returns


def test_validate_ohlcv_dataframe() -> None:
    valid_data = {
        "Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "Open": [100.0, 102.0, 101.0],
        "High": [103.0, 104.0, 102.0],
        "Low": [99.0, 100.0, 98.0],
        "Close": [102.0, 101.0, 100.0],
        "Volume": [1000, 1200, 1100],
    }
    df = pd.DataFrame(valid_data)
    clean = validate_ohlcv_dataframe(df)
    assert len(clean) == 3
    assert "date" in clean.columns

    # Test missing columns
    bad_df = df.drop(columns=["High"])
    with pytest.raises(DataError):
        validate_ohlcv_dataframe(bad_df)

    # Test high < low inconsistency
    bad_high = df.copy()
    bad_high.loc[0, "High"] = 90.0
    with pytest.raises(DataError):
        validate_ohlcv_dataframe(bad_high)


def test_returns_and_summary() -> None:
    df = pd.DataFrame({"close": [100.0, 102.0, 100.98]})
    rets = compute_price_returns(df, price_col="close")
    assert len(rets) == 2
    assert rets.iloc[0] == pytest.approx(0.02)
    assert rets.iloc[1] == pytest.approx(-0.01)

    summary = summarize_empirical_returns(rets)
    assert summary["win_probability"] == 0.50
    assert summary["mean_win_return"] == pytest.approx(0.02)
    assert summary["mean_loss_return"] == pytest.approx(-0.01)
