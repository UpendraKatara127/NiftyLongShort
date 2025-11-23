from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class BaseStrategy(ABC):
    def __init__(
        self,
        capital: float,
        cost_bps_per_side: float = 10.0,
        stop_loss_pct: Optional[float] = None,
    ):
        self.capital = capital
        self.cost_bps_per_side = cost_bps_per_side
        self.stop_loss_pct = stop_loss_pct

    def on_backtest_start(self, data: pd.DataFrame) -> None:
        pass

    def on_backtest_end(self, results: dict) -> None:
        pass

    @abstractmethod
    def generate_target_weights(
        self, timestamp: pd.Timestamp, prices_slice: pd.DataFrame
    ) -> Dict[str, float]:
        raise NotImplementedError
