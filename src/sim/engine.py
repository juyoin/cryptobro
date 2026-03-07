"""Simulation engine (paper trading only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TradeDecision:
    action: str
    confidence: float


def should_execute_buy(avg_confidence: float, threshold: float = 0.7) -> bool:
    return avg_confidence > threshold
