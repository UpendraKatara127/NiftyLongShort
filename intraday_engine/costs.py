class TransactionCostModel:
    def __init__(self, cost_bps_per_side: float = 10.0):
        self.cost_bps_per_side = cost_bps_per_side

    def cost(self, notional: float) -> float:
        """Return cost for a single side of the trade."""
        return abs(notional) * (self.cost_bps_per_side / 10_000.0)

    def round_trip_cost(self, notional: float) -> float:
        return 2.0 * self.cost(notional)
