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
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class AIRateLimitError(Exception):
    """Raised when an AI provider responds with HTTP 429."""


class AIJury:
    """Queries multiple AI models and normalizes their trading votes."""

    def __init__(
        self,
        huggingface_api_key: str | None = None,
        groq_api_key: str | None = None,
        huggingface_model: str | None = None,
        groq_model: str | None = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        backoff_base_seconds: float = 15.0,
        provider_cooldown_seconds: int = 300,
    ) -> None:
        self.huggingface_api_key = huggingface_api_key or os.getenv("HUGGINGFACE_API_KEY", "")
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.huggingface_model = huggingface_model or os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.provider_cooldown_seconds = provider_cooldown_seconds
        self.provider_cooldowns: dict[str, float] = {"huggingface": 0.0, "groq": 0.0}
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
        return [
            self._get_vote(provider="groq", model=self.groq_model, request_fn=lambda: self._query_groq(prompt)),
            self._get_vote(
                provider="huggingface",
                model=self.huggingface_model,
                request_fn=lambda: self._query_huggingface(prompt),
            ),
        ]

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
        if provider == "huggingface" and not self.huggingface_api_key:
            return self._error_vote(provider, model, "Missing HUGGINGFACE_API_KEY")
        if provider == "groq" and not self.groq_api_key:
            return self._error_vote(provider, model, "Missing GROQ_API_KEY")

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

        try:
            parsed = self._parse_ai_json(raw_text)
        except Exception as exc:
            if "No JSON object found" in str(exc):
                fallback = self._fallback_vote_from_text(raw_text)
                if fallback is not None:
                    return {
                        "provider": provider,
                        "model": model,
                        "action": fallback["action"],
                        "confidence": fallback["confidence"],
                        "reasoning": fallback["reasoning"],
                        "error": None,
                    }
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

    def _query_groq(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.groq_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 220,
        }

        for attempt in range(self.max_retries + 1):
            response = self.session.post(GROQ_API_URL, headers=headers, json=payload, timeout=self.timeout_seconds)
            if response.status_code != 429:
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices", []) if isinstance(data, dict) else []
                if not choices:
                    raise ValueError("Unexpected Groq response format: missing choices")

                content = choices[0].get("message", {}).get("content", "")
                text = str(content).strip()
                if not text:
                    raise ValueError("Groq response contains empty content")
                return text

            if attempt >= self.max_retries:
                retry_after = response.headers.get("Retry-After", "unknown")
                raise AIRateLimitError(f"Groq rate limit hit (HTTP 429). Retry-After={retry_after}")

            retry_after_seconds = self._retry_after_to_seconds(response.headers.get("Retry-After"))
            backoff = self.backoff_base_seconds * (2**attempt)
            sleep_seconds = retry_after_seconds if retry_after_seconds is not None else backoff
            time.sleep(max(1.0, sleep_seconds))

        raise AIRateLimitError("Groq rate limit retries exhausted")

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

    def _fallback_vote_from_text(self, response_text: str) -> dict[str, Any] | None:
        text = response_text.strip()
        lower_text = f" {text.lower()} "

        action: str | None = None
        if any(token in lower_text for token in (" hold ", " neutral ", " wait ", " sideways ")):
            action = "HOLD"
        if any(token in lower_text for token in (" sell ", " bearish ", " take profit", " de-risk")):
            action = "SELL"
        if any(token in lower_text for token in (" buy ", " bullish ", " accumulate ", " oversold ")):
            action = "BUY"

        if action is None:
            return None

        confidence = 0.62
        pct_match = re.search(r"(\d{1,3}(?:\.\d+)?)\s*%", text)
        if pct_match:
            confidence = min(1.0, max(0.0, float(pct_match.group(1)) / 100.0))
        else:
            float_match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
            if float_match:
                confidence = min(1.0, max(0.0, float(float_match.group(1))))

        reasoning = text.split("\n")[0].strip()[:220] or "Fallback parsed from non-JSON model output."
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
