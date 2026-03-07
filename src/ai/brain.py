"""Phase 2 AI Jury for Crypto Oracle (SIMULATION ONLY)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

ALLOWED_ACTIONS = {"BUY", "SELL", "HOLD"}
HF_ROUTER_URL = "https://router.huggingface.co"


class AIRateLimitError(Exception):
    """Raised when an AI provider responds with HTTP 429."""


class AIJury:
    """Queries multiple AI models and normalizes their trading votes."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        gemini_model: str | None = None,
        huggingface_model: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_base_seconds: float = 2.0,
        provider_cooldown_seconds: int = 300,
    ) -> None:
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.huggingface_model = huggingface_model or os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.provider_cooldown_seconds = provider_cooldown_seconds
        self.provider_cooldowns: dict[str, float] = {"gemini": 0.0, "huggingface": 0.0}
        self.session = requests.Session()

    def get_market_prompt(self, market_dossier: dict[str, Any], risk_level: str = "medium") -> str:
        compact_dossier = json.dumps(market_dossier, separators=(",", ":"))
        return (
            "You are a crypto analyst in a paper-trading SIMULATION ONLY environment. "
            "Never give real financial advice. Analyze the market dossier and return exactly one JSON object "
            "with this schema: {\"action\":\"BUY|SELL|HOLD\",\"confidence\":0.0-1.0,\"reasoning\":\"short text\"}. "
            f"Risk level: {risk_level}. Market dossier: {compact_dossier}"
        )

    def get_market_dossier_votes(self, market_dossier: dict[str, Any], risk_level: str = "medium") -> list[dict[str, Any]]:
        prompt = self.get_market_prompt(market_dossier=market_dossier, risk_level=risk_level)

        votes = [
            self._get_vote(provider="gemini", model=self.gemini_model, request_fn=lambda: self._query_gemini(prompt)),
            self._get_vote(
                provider="huggingface",
                model=self.huggingface_model,
                request_fn=lambda: self._query_huggingface(prompt),
            ),
        ]

        return votes

    def get_consensus(self, votes: list[dict[str, Any]]) -> dict[str, Any]:
        valid_votes = [vote for vote in votes if vote.get("error") is None]

        if not valid_votes:
            return {
                "consensus_action": "HOLD",
                "average_confidence": 0.0,
                "vote_count": 0,
            }

        average_confidence = sum(float(v["confidence"]) for v in valid_votes) / len(valid_votes)

        buy_votes = sum(1 for vote in valid_votes if vote["action"] == "BUY")
        sell_votes = sum(1 for vote in valid_votes if vote["action"] == "SELL")
        hold_votes = sum(1 for vote in valid_votes if vote["action"] == "HOLD")

        if buy_votes > sell_votes and buy_votes > hold_votes:
            consensus_action = "BUY"
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            consensus_action = "SELL"
        else:
            consensus_action = "HOLD"

        return {
            "consensus_action": consensus_action,
            "average_confidence": round(average_confidence, 4),
            "vote_count": len(valid_votes),
        }

    def get_jury_verdict(self, market_dossier: dict[str, Any], risk_level: str = "medium") -> dict[str, Any]:
        votes = self.get_market_dossier_votes(market_dossier=market_dossier, risk_level=risk_level)
        consensus = self.get_consensus(votes)
        return {
            "simulation": True,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "votes": votes,
            "consensus": consensus,
        }

    def _get_vote(self, provider: str, model: str, request_fn: Callable[[], str]) -> dict[str, Any]:
        if provider == "gemini" and not self.gemini_api_key:
            return self._error_vote(provider, model, "Missing GEMINI_API_KEY")
        if provider == "huggingface" and not self.huggingface_api_key:
            return self._error_vote(provider, model, "Missing HUGGINGFACE_API_KEY")

        cooldown_remaining = self._cooldown_remaining_seconds(provider)
        if cooldown_remaining > 0:
            return self._error_vote(provider, model, f"{provider} cooldown active for {cooldown_remaining:.0f}s")

        try:
            raw_text = request_fn()
        except AIRateLimitError as exc:
            self._start_provider_cooldown(provider)
            return self._error_vote(provider, model, str(exc))
        except Exception as exc:
            return self._error_vote(provider, model, str(exc))

        # Explicit parse guard so malformed model output does not crash the jury.
        try:
            parsed = self._parse_ai_json(raw_text)
        except Exception as exc:
            if "No JSON object found" in str(exc):
                return self._error_vote(provider, model, f"Parser warning: {exc}")
            return self._error_vote(provider, model, str(exc))

        return {
            "provider": provider,
            "model": model,
            "action": parsed["action"],
            "confidence": parsed["confidence"],
            "reasoning": parsed["reasoning"],
            "error": None,
        }

    def _query_gemini(self, prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent"
        params = {"key": self.gemini_api_key}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 220,
            },
        }

        for attempt in range(self.max_retries + 1):
            response = self.session.post(url, params=params, json=payload, timeout=self.timeout_seconds)
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    raise ValueError("Gemini response contains no candidates")

                parts = candidates[0].get("content", {}).get("parts", [])
                text = "".join(str(part.get("text", "")) for part in parts)
                if not text:
                    raise ValueError("Gemini response contains empty text")
                return text

            if attempt >= self.max_retries:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise AIRateLimitError(f"Gemini rate limit hit (HTTP 429). Retry-After={retry_after}")

            retry_after_seconds = self._retry_after_to_seconds(response.headers.get("Retry-After"))
            backoff = self.backoff_base_seconds * (2**attempt)
            sleep_seconds = retry_after_seconds if retry_after_seconds is not None else backoff
            time.sleep(max(1.0, sleep_seconds))

        raise AIRateLimitError("Gemini rate limit retries exhausted")

    def _query_huggingface(self, prompt: str) -> str:
        url = f"{HF_ROUTER_URL}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.huggingface_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 220,
        }

        response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise AIRateLimitError(f"Hugging Face rate limit hit (HTTP 429). Retry-After={retry_after}")
        if response.status_code == 404:
            raise ValueError(
                f"Hugging Face router model not found: {self.huggingface_model}. "
                "Confirm model availability for Inference Providers."
            )
        response.raise_for_status()

        data = response.json()
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            raise ValueError("Unexpected Hugging Face response format: missing choices")

        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            content_text_parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
            content = "".join(content_text_parts)

        text = str(content).strip()
        if not text:
            raise ValueError("Hugging Face response contains empty content")

        return text

    def _parse_ai_json(self, response_text: str) -> dict[str, Any]:
        candidate = response_text.strip()
        candidate = re.sub(r"^```json\\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^```\\s*", "", candidate)
        candidate = re.sub(r"```$", "", candidate).strip()

        if not candidate.startswith("{"):
            json_match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in model response")
            candidate = json_match.group(0)

        parsed = json.loads(candidate)

        action = str(parsed.get("action", "")).upper()
        confidence_raw = parsed.get("confidence")
        reasoning = str(parsed.get("reasoning", "")).strip()

        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Confidence must be a number") from exc

        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("Confidence must be between 0 and 1")
        if not reasoning:
            raise ValueError("Reasoning cannot be empty")

        return {
            "action": action,
            "confidence": confidence,
            "reasoning": reasoning,
        }

    def _retry_after_to_seconds(self, retry_after: str | None) -> float | None:
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            pass

        try:
            retry_dt = parsedate_to_datetime(retry_after)
            now_dt = datetime.now(timezone.utc)
            return max(0.0, (retry_dt - now_dt).total_seconds())
        except Exception:
            return None

    def _start_provider_cooldown(self, provider: str) -> None:
        self.provider_cooldowns[provider] = time.time() + float(self.provider_cooldown_seconds)

    def _cooldown_remaining_seconds(self, provider: str) -> float:
        return max(0.0, self.provider_cooldowns.get(provider, 0.0) - time.time())

    def _error_vote(self, provider: str, model: str, error_message: str) -> dict[str, Any]:
        return {
            "provider": provider,
            "model": model,
            "action": "HOLD",
            "confidence": 0.0,
            "reasoning": "Provider unavailable; defaulting to HOLD for simulation safety.",
            "error": error_message,
        }


def evaluate_market_with_ai(market_dossier: dict[str, Any], risk_level: str = "medium") -> dict[str, Any]:
    jury = AIJury()
    return jury.get_jury_verdict(market_dossier=market_dossier, risk_level=risk_level)
