from dataclasses import dataclass


@dataclass
class Position:
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: object


class PortfolioBook:
    def __init__(self, capital, transaction_cost):
        self.capital = capital
        self.transaction_cost = transaction_cost
        self.nav = capital
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []

    def close_positions(self, timestamp, prices):
        updates = {}
        for symbol, position in list(self.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            side = 1 if position.side == "long" else -1
            qty = abs(position.quantity)
            entry_notional = position.entry_price * qty
            exit_notional = price * qty
            pnl = side * (price - position.entry_price) * qty
            cost = (entry_notional + exit_notional) * self.transaction_cost
            net_pnl = pnl - cost
            self.nav += net_pnl
            self.trade_log.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "side": position.side,
                    "entry_price": position.entry_price,
                    "exit_price": price,
                    "quantity": qty,
                    "pnl": net_pnl,
                    "pnl_pct": net_pnl / entry_notional if entry_notional else 0,
                    "entry_time": position.entry_time,
                }
            )
            updates[symbol] = net_pnl
            del self.positions[symbol]
        return updates

    def open_positions(self, timestamp, orders):
        for order in orders:
            symbol = order["symbol"]
            price = order["price"]
            quantity = order["quantity"]
            side = order["side"]
            notional = price * abs(quantity)
            cost = notional * self.transaction_cost
            self.nav -= cost
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
            )

    def mark_nav(self, timestamp):
        self.equity_curve.append({"timestamp": timestamp, "nav": self.nav})
