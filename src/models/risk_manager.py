"""Risk management model for trailing stop-loss and take-profit (SIMULATION ONLY)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class PositionRiskState:
    symbol: str
    entry_price: float
    peak_price: float
    trailing_stop_pct: float
    take_profit_pct: float
    active: bool = True

    @property
    def trailing_stop_price(self) -> float:
        return self.peak_price * (1.0 - self.trailing_stop_pct)

    @property
    def take_profit_price(self) -> float:
        return self.entry_price * (1.0 + self.take_profit_pct)


class RiskManager:
    """Tracks position risk state and emits SELL overrides when risk triggers fire."""

    def __init__(self, trailing_stop_pct: float = 0.05, take_profit_pct: float = 0.10) -> None:
        if trailing_stop_pct <= 0 or take_profit_pct <= 0:
            raise ValueError("Risk percentages must be positive.")
        self.trailing_stop_pct = trailing_stop_pct
        self.take_profit_pct = take_profit_pct
        self._states: dict[str, PositionRiskState] = {}

    def register_purchase(self, symbol: str, purchase_price: float) -> dict[str, Any]:
        symbol = symbol.upper()
        if purchase_price <= 0:
            raise ValueError("purchase_price must be > 0.")

        state = PositionRiskState(
            symbol=symbol,
            entry_price=float(purchase_price),
            peak_price=float(purchase_price),
            trailing_stop_pct=self.trailing_stop_pct,
            take_profit_pct=self.take_profit_pct,
            active=True,
        )
        self._states[symbol] = state
        return self._state_payload(state)

    def clear_position(self, symbol: str) -> None:
        self._states.pop(symbol.upper(), None)

    def has_position(self, symbol: str) -> bool:
        state = self._states.get(symbol.upper())
        return bool(state and state.active)

    def evaluate(self, symbol: str, current_price: float) -> dict[str, Any]:
        symbol = symbol.upper()
        if current_price <= 0:
            raise ValueError("current_price must be > 0.")

        state = self._states.get(symbol)
        if state is None or not state.active:
            return {
                "symbol": symbol,
                "triggered": False,
                "trigger": "none",
                "action": "HOLD",
                "reason": "No tracked open position.",
                "state": None,
            }

        state.peak_price = max(state.peak_price, float(current_price))
        trailing_stop_price = state.trailing_stop_price
        take_profit_price = state.take_profit_price

        if current_price <= trailing_stop_price:
            state.active = False
            return {
                "symbol": symbol,
                "triggered": True,
                "trigger": "trailing_stop_loss",
                "action": "SELL",
                "reason": (
                    f"Price {current_price:.4f} <= trailing stop {trailing_stop_price:.4f} "
                    f"(peak {state.peak_price:.4f}, stop {state.trailing_stop_pct:.1%})."
                ),
                "state": self._state_payload(state),
            }

        if current_price >= take_profit_price:
            state.active = False
            return {
                "symbol": symbol,
                "triggered": True,
                "trigger": "take_profit",
                "action": "SELL",
                "reason": (
                    f"Price {current_price:.4f} >= take-profit {take_profit_price:.4f} "
                    f"(entry {state.entry_price:.4f}, target {state.take_profit_pct:.1%})."
                ),
                "state": self._state_payload(state),
            }

        return {
            "symbol": symbol,
            "triggered": False,
            "trigger": "none",
            "action": "HOLD",
            "reason": "No risk trigger.",
            "state": self._state_payload(state),
        }

    def get_state(self, symbol: str) -> dict[str, Any] | None:
        state = self._states.get(symbol.upper())
        if state is None:
            return None
        return self._state_payload(state)

    def _state_payload(self, state: PositionRiskState) -> dict[str, Any]:
        payload = asdict(state)
        payload["trailing_stop_price"] = state.trailing_stop_price
        payload["take_profit_price"] = state.take_profit_price
        return payload
