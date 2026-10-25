"""Reproducibility metadata generator."""

from __future__ import annotations

import sys
import platform
import time
from typing import Dict, Any
import numpy as np
import scipy
import pandas as pd


def generate_environment_metadata() -> Dict[str, Any]:
    """Capture runtime environment versions and system parameters for reproducible reporting."""
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
    }
