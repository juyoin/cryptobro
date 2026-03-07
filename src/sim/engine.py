"""Phase 3 execution engine for Crypto Oracle (SIMULATION ONLY)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data.market_data import MarketHarvester


@dataclass(slots=True)
class TradeResult:
    executed: bool
    action: str
    symbol: str
    price_usd: float
    units: float
    fee_usd: float
    message: str


class ExecutionEngine:
    """Handles wallet persistence and paper-trade execution."""

    def __init__(
        self,
        wallet_path: str = "config/wallet.json",
        history_log_path: str = "history.log",
        fee_rate: float = 0.001,
        buy_threshold: float = 0.7,
        sell_threshold: float = -0.7,
    ) -> None:
        self.wallet_path = Path(wallet_path)
        self.history_log_path = Path(history_log_path)
        self.fee_rate = fee_rate
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def load_wallet(self) -> dict[str, Any]:
        with self.wallet_path.open("r", encoding="utf-8-sig") as file:
            wallet = json.load(file)

        wallet.setdefault("simulation", True)
        wallet.setdefault("base_currency", "USD")
        wallet.setdefault("cash_usd", 10000.0)
        wallet.setdefault("positions", {})
        wallet.setdefault("trade_log", [])
        for symbol in ("BTC", "ETH", "SOL"):
            wallet["positions"].setdefault(symbol, {"units": 0.0, "avg_entry_price_usd": 0.0})
        return wallet

    def save_wallet(self, wallet: dict[str, Any]) -> None:
        self.wallet_path.parent.mkdir(parents=True, exist_ok=True)
        with self.wallet_path.open("w", encoding="utf-8") as file:
            json.dump(wallet, file, indent=2)

    def execute_from_jury(
        self,
        coin_symbol: str,
        investment_amount_usd: float,
        jury_verdict: dict[str, Any],
        market_dossier: dict[str, Any] | None = None,
    ) -> TradeResult:
        symbol = coin_symbol.upper()
        if symbol not in {"BTC", "ETH", "SOL"}:
            raise ValueError(f"Unsupported symbol: {symbol}")

        if market_dossier is None:
            market_dossier = MarketHarvester().get_market_dossier()

        coin_data = market_dossier.get("coins", {}).get(symbol)
        if coin_data is None:
            raise ValueError(f"Missing market data for {symbol}")

        price_usd = float(coin_data.get("price_usd", 0.0))
        if price_usd <= 0:
            raise ValueError(f"Invalid market price for {symbol}: {price_usd}")

        signed_confidence = self._get_signed_confidence(jury_verdict)
        reasoning = self._get_reasoning(jury_verdict)

        if signed_confidence > self.buy_threshold:
            return self._buy(symbol=symbol, price_usd=price_usd, amount_usd=investment_amount_usd, reasoning=reasoning)

        if signed_confidence < self.sell_threshold:
            return self._sell_all(symbol=symbol, price_usd=price_usd, reasoning=reasoning)

        return TradeResult(
            executed=False,
            action="HOLD",
            symbol=symbol,
            price_usd=price_usd,
            units=0.0,
            fee_usd=0.0,
            message=f"No trade. Signed confidence {signed_confidence:.4f} inside hold band.",
        )

    def _buy(self, symbol: str, price_usd: float, amount_usd: float, reasoning: str) -> TradeResult:
        if amount_usd <= 0:
            return TradeResult(
                executed=False,
                action="BUY",
                symbol=symbol,
                price_usd=price_usd,
                units=0.0,
                fee_usd=0.0,
                message="Buy rejected: amount must be > 0.",
            )

        wallet = self.load_wallet()
        cash = float(wallet.get("cash_usd", 0.0))
        fee_usd = amount_usd * self.fee_rate
        total_cost = amount_usd + fee_usd

        if total_cost > cash:
            return TradeResult(
                executed=False,
                action="BUY",
                symbol=symbol,
                price_usd=price_usd,
                units=0.0,
                fee_usd=fee_usd,
                message=f"Buy rejected: required ${total_cost:.2f}, available ${cash:.2f}.",
            )

        units_bought = amount_usd / price_usd
        position = wallet["positions"][symbol]
        prev_units = float(position.get("units", 0.0))
        prev_avg = float(position.get("avg_entry_price_usd", 0.0))
        new_total_units = prev_units + units_bought

        if new_total_units > 0:
            weighted_cost = (prev_units * prev_avg) + (units_bought * price_usd)
            new_avg = weighted_cost / new_total_units
        else:
            new_avg = 0.0

        wallet["cash_usd"] = cash - total_cost
        wallet["positions"][symbol]["units"] = new_total_units
        wallet["positions"][symbol]["avg_entry_price_usd"] = new_avg
        self._append_wallet_trade_log(
            wallet=wallet,
            action="BUY",
            symbol=symbol,
            price_usd=price_usd,
            units=units_bought,
            fee_usd=fee_usd,
            reasoning=reasoning,
        )
        self.save_wallet(wallet)
        self._append_history_log(
            action="BUY",
            symbol=symbol,
            price_usd=price_usd,
            units=units_bought,
            fee_usd=fee_usd,
            reasoning=reasoning,
        )

        return TradeResult(
            executed=True,
            action="BUY",
            symbol=symbol,
            price_usd=price_usd,
            units=units_bought,
            fee_usd=fee_usd,
            message=f"Bought {units_bought:.8f} {symbol} for ${amount_usd:.2f} (+${fee_usd:.2f} fee).",
        )

    def _sell_all(self, symbol: str, price_usd: float, reasoning: str) -> TradeResult:
        wallet = self.load_wallet()
        position = wallet["positions"][symbol]
        units_held = float(position.get("units", 0.0))

        if units_held <= 0:
            return TradeResult(
                executed=False,
                action="SELL",
                symbol=symbol,
                price_usd=price_usd,
                units=0.0,
                fee_usd=0.0,
                message=f"Sell rejected: no {symbol} units available.",
            )

        gross_proceeds = units_held * price_usd
        fee_usd = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - fee_usd

        wallet["cash_usd"] = float(wallet.get("cash_usd", 0.0)) + net_proceeds
        wallet["positions"][symbol]["units"] = 0.0
        wallet["positions"][symbol]["avg_entry_price_usd"] = 0.0
        self._append_wallet_trade_log(
            wallet=wallet,
            action="SELL",
            symbol=symbol,
            price_usd=price_usd,
            units=units_held,
            fee_usd=fee_usd,
            reasoning=reasoning,
        )
        self.save_wallet(wallet)
        self._append_history_log(
            action="SELL",
            symbol=symbol,
            price_usd=price_usd,
            units=units_held,
            fee_usd=fee_usd,
            reasoning=reasoning,
        )

        return TradeResult(
            executed=True,
            action="SELL",
            symbol=symbol,
            price_usd=price_usd,
            units=units_held,
            fee_usd=fee_usd,
            message=f"Sold {units_held:.8f} {symbol} for ${net_proceeds:.2f} after ${fee_usd:.2f} fee.",
        )

    def _append_wallet_trade_log(
        self,
        wallet: dict[str, Any],
        action: str,
        symbol: str,
        price_usd: float,
        units: float,
        fee_usd: float,
        reasoning: str,
    ) -> None:
        wallet["trade_log"].append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "symbol": symbol,
                "price_usd": price_usd,
                "units": units,
                "fee_usd": fee_usd,
                "reasoning": reasoning,
            }
        )

    def _append_history_log(
        self,
        action: str,
        symbol: str,
        price_usd: float,
        units: float,
        fee_usd: float,
        reasoning: str,
    ) -> None:
        self.history_log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        line = (
            f"{timestamp} | {action} | {symbol} | price_usd={price_usd:.8f} | "
            f"units={units:.8f} | fee_usd={fee_usd:.8f} | reasoning={reasoning}\n"
        )
        with self.history_log_path.open("a", encoding="utf-8") as file:
            file.write(line)

    def _get_signed_confidence(self, jury_verdict: dict[str, Any]) -> float:
        consensus = jury_verdict.get("consensus", {})
        if "signed_confidence" in consensus:
            return float(consensus.get("signed_confidence", 0.0))

        avg_conf = float(consensus.get("average_confidence", 0.0))
        action = str(consensus.get("consensus_action", "HOLD")).upper()

        if action == "BUY":
            return avg_conf
        if action == "SELL":
            return -avg_conf
        return 0.0

    def _get_reasoning(self, jury_verdict: dict[str, Any]) -> str:
        votes = jury_verdict.get("votes", [])
        snippets: list[str] = []
        for vote in votes:
            if vote.get("error"):
                continue
            provider = str(vote.get("provider", "model"))
            text = str(vote.get("reasoning", "")).strip()
            if text:
                snippets.append(f"{provider}: {text}")

        if snippets:
            return " | ".join(snippets)

        consensus = jury_verdict.get("consensus", {})
        fallback = str(consensus.get("reasoning", "")).strip()
        return fallback or "No AI reasoning provided."


