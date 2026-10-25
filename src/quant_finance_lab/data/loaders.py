"""CSV historical data loaders with automatic schema verification."""

from __future__ import annotations

import os
from pathlib import Path
import pandas as pd

from quant_finance_lab.core.exceptions import DataError
from quant_finance_lab.data.validation import validate_ohlcv_dataframe


def load_csv_price_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load and strictly validate historical price data from a CSV file.
    
    Expected format:
      date,open,high,low,close,volume
      2025-01-02,100.0,102.5,99.5,101.8,1500000
    """
    path = Path(file_path)
    if not path.is_file():
        raise DataError(f"File not found: {path.resolve()}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataError(f"Failed to read CSV at {path}: {e}")

    return validate_ohlcv_dataframe(df)
