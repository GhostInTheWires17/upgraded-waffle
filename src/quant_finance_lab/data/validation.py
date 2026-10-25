"""Data integrity and schema validation for financial price series."""

from __future__ import annotations

import pandas as pd
from quant_finance_lab.core.exceptions import DataError


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def validate_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate historical OHLCV pricing dataframe.
    
    Checks:
      - Column names normalized to lower case.
      - Required columns present.
      - Chronological ordering and uniqueness of timestamps.
      - Positive price values (open, high, low, close > 0).
      - Price consistency (low <= open <= high, low <= close <= high).
      - Non-negative volume (volume >= 0).
    """
    if not isinstance(df, pd.DataFrame):
        raise DataError(f"Expected pandas DataFrame, got {type(df).__name__}")

    clean_df = df.copy()
    clean_df.columns = [str(c).strip().lower() for c in clean_df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in clean_df.columns]
    if missing:
        raise DataError(f"Historical data missing required OHLCV columns: {missing}")

    # Parse and validate timestamp
    try:
        clean_df["date"] = pd.to_datetime(clean_df["date"])
    except Exception as e:
        raise DataError(f"Failed to parse 'date' column into datetime: {e}")

    # Check duplicates
    if clean_df["date"].duplicated().any():
        dup_count = clean_df["date"].duplicated().sum()
        raise DataError(f"Historical data contains {dup_count} duplicate timestamps.")

    # Sort chronologically
    clean_df = clean_df.sort_values("date").reset_index(drop=True)

    # Check non-empty
    if len(clean_df) < 2:
        raise DataError(f"Dataset too small ({len(clean_df)} rows). Need at least 2 price bars.")

    # Validate price positivity
    for price_col in ["open", "high", "low", "close"]:
        if (clean_df[price_col] <= 0).any():
            raise DataError(f"Column '{price_col}' contains zero or negative prices.")

    # Validate high >= low, high >= open, high >= close, etc.
    invalid_bounds = (
        (clean_df["high"] < clean_df["low"])
        | (clean_df["high"] < clean_df["open"])
        | (clean_df["high"] < clean_df["close"])
        | (clean_df["low"] > clean_df["open"])
        | (clean_df["low"] > clean_df["close"])
    )
    if invalid_bounds.any():
        bad_rows = clean_df[invalid_bounds].head(3)
        raise DataError(f"Price consistency violation (high < low or open/close outside range) in:\n{bad_rows}")

    if (clean_df["volume"] < 0).any():
        raise DataError("Volume column contains negative values.")

    return clean_df
