from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EngineConfig:
    capital: float = 10_000_000.0
    cost_bps_per_side: float = 10.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    rebalance_times: List[str] = field(
        default_factory=lambda: ["10:15", "11:15", "12:15", "13:15", "14:15"]
    )
    market_open: str = "09:15"
    market_close: str = "15:30"
    timezone: str = "Asia/Kolkata"

    def as_dict(self) -> dict:
        return {
            "capital": self.capital,
            "cost_bps_per_side": self.cost_bps_per_side,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "rebalance_times": list(self.rebalance_times),
            "market_open": self.market_open,
            "market_close": self.market_close,
            "timezone": self.timezone,
        }
