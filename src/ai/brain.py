"""AI Jury module (SIMULATION ONLY).

Phase 2 placeholder:
- Accept market dossier.
- Query multiple AI models.
- Return normalized votes: action/confidence/reasoning.
"""

from __future__ import annotations

from typing import Any


def evaluate_market_with_ai(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    # TODO: Integrate LiteLLM / direct provider clients.
    # Keep strict JSON schema for parser safety.
    return [
        {
            "model": "placeholder-model",
            "action": "HOLD",
            "confidence": 0.5,
            "reasoning": "Phase 2 not implemented yet.",
        }
    ]
