"""MVC Controller: multi-agent trading debate orchestrator (SIMULATION ONLY)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

from src.data.market_data import MarketHarvester
from src.models.risk_manager import RiskManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

HF_ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
XAI_CHAT_URL = "https://api.x.ai/v1/chat/completions"
ALLOWED_ACTIONS = {"BUY", "SELL", "HOLD"}


@dataclass(slots=True)
class JudgeDecision:
    action: str
    confidence: float
    reasoning: str


class TradingController:
    """Runs Bull vs Bear debate, then asks xAI Grok to produce the final decision."""

    def __init__(
        self,
        market_harvester: MarketHarvester | None = None,
        whale_watcher: Any | None = None,
        risk_manager: RiskManager | None = None,
    ) -> None:
        self.market_harvester = market_harvester or MarketHarvester()
        self.whale_watcher = whale_watcher
        self.risk_manager = risk_manager or RiskManager()
        self.session = requests.Session()
        self.timeout_seconds = 30

        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.xai_api_key = os.getenv("XAI_API_KEY", "")

        self.groq_bull_model = os.getenv("GROQ_BULL_MODEL", os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        self.hf_bear_model = os.getenv("HF_BEAR_MODEL", os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.2-1B-Instruct"))
        self.xai_judge_model = os.getenv("XAI_JUDGE_MODEL", os.getenv("XAI_MODEL", "grok-2-latest"))
        self.bull_system_prompt = os.getenv(
            "BULL_SYSTEM_PROMPT",
            "Your soul purpose is to find the silver lining. Even in a crash, find the ''dip-buying'' opportunity.",
        )
        self.bear_system_prompt = os.getenv(
            "BEAR_SYSTEM_PROMPT",
            "You are a skeptic. Find every reason why this coin is overvalued or why the macro-environment is dangerous.",
        )

    def register_position(self, symbol: str, entry_price: float) -> dict[str, Any]:
        """Call this after simulated BUY execution so risk rules can monitor the position."""
        return self.risk_manager.register_purchase(symbol=symbol, purchase_price=entry_price)

    def clear_position(self, symbol: str) -> None:
        """Call this after simulated SELL execution to reset risk tracking."""
        self.risk_manager.clear_position(symbol=symbol)

    def run_debate(self, symbol: str, risk_level: str = "medium") -> dict[str, Any]:
        symbol = symbol.upper()
        if symbol not in {"BTC", "ETH", "SOL"}:
            raise ValueError(f"Unsupported symbol: {symbol}")

        market_dossier = self.market_harvester.get_market_dossier()
        coin_data = market_dossier.get("coins", {}).get(symbol)
        if not coin_data:
            raise ValueError(f"Missing market data for {symbol}")

        whale_status = self._get_whale_status(symbol=symbol)

        bull_prompt = self._build_bull_prompt(symbol=symbol, coin_data=coin_data, whale_status=whale_status, risk_level=risk_level)
        bear_prompt = self._build_bear_prompt(symbol=symbol, coin_data=coin_data, whale_status=whale_status, risk_level=risk_level)

        bull_argument = self._safe_argument(
            provider="groq",
            fetch_fn=lambda: self._query_groq(prompt=bull_prompt),
            fallback=f"Groq bull agent unavailable for {symbol}.",
        )
        bear_argument = self._safe_argument(
            provider="huggingface",
            fetch_fn=lambda: self._query_huggingface(prompt=bear_prompt),
            fallback=f"Hugging Face bear agent unavailable for {symbol}.",
        )

        judge_prompt = self._build_judge_prompt(
            symbol=symbol,
            coin_data=coin_data,
            whale_status=whale_status,
            bull_argument=bull_argument,
            bear_argument=bear_argument,
            risk_level=risk_level,
        )

        judge_raw = self._safe_argument(
            provider="xai",
            fetch_fn=lambda: self._query_xai(prompt=judge_prompt),
            fallback='{"action":"HOLD","confidence":0.50,"reasoning":"xAI judge unavailable; fallback HOLD."}',
        )
        judge_decision = self._parse_judge_decision(judge_raw)

        risk_eval = self.risk_manager.evaluate(symbol=symbol, current_price=float(coin_data.get("price_usd", 0.0)))
        risk_override = False

        if risk_eval.get("triggered"):
            risk_override = True
            judge_decision = JudgeDecision(
                action="SELL",
                confidence=max(0.99, judge_decision.confidence),
                reasoning=f"Risk override: {risk_eval.get('reason', 'stop-loss/take-profit triggered.')}",
            )

        return {
            "simulation": True,
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "market": coin_data,
            "whale_status": whale_status,
            "debate": {
                "bull_agent": {
                    "provider": "groq",
                    "model": self.groq_bull_model,
                    "argument": bull_argument,
                },
                "bear_agent": {
                    "provider": "huggingface",
                    "model": self.hf_bear_model,
                    "argument": bear_argument,
                },
                "judge_agent": {
                    "provider": "xai",
                    "model": self.xai_judge_model,
                    "raw": judge_raw,
                    "decision": asdict(judge_decision),
                },
            },
            "risk": {
                "triggered": bool(risk_eval.get("triggered")),
                "trigger": risk_eval.get("trigger", "none"),
                "override_applied": risk_override,
                "details": risk_eval,
            },
            "consensus": {
                "consensus_action": judge_decision.action,
                "average_confidence": judge_decision.confidence,
                "vote_count": 1,
                "reasoning": judge_decision.reasoning,
            },
        }

    def _build_bull_prompt(self, symbol: str, coin_data: dict[str, Any], whale_status: dict[str, Any], risk_level: str) -> str:
        return (
            "SIMULATION ONLY. You are Agent A (Bull, Groq). "
            "Provide exactly 3 concise, numbered reasons to BUY this asset. "
            "Only bullish points allowed.\n"
            f"Risk level: {risk_level}\n"
            f"Asset: {symbol}\n"
            f"Market data: {json.dumps(coin_data, separators=(',', ':'))}\n"
            f"Whale status: {json.dumps(whale_status, separators=(',', ':'))}"
        )

    def _build_bear_prompt(self, symbol: str, coin_data: dict[str, Any], whale_status: dict[str, Any], risk_level: str) -> str:
        return (
            "SIMULATION ONLY. You are Agent B (Bear, Hugging Face). "
            "Provide exactly 3 concise, numbered reasons to SELL this asset. "
            "Only bearish points allowed.\n"
            f"Risk level: {risk_level}\n"
            f"Asset: {symbol}\n"
            f"Market data: {json.dumps(coin_data, separators=(',', ':'))}\n"
            f"Whale status: {json.dumps(whale_status, separators=(',', ':'))}"
        )

    def _build_judge_prompt(
        self,
        symbol: str,
        coin_data: dict[str, Any],
        whale_status: dict[str, Any],
        bull_argument: str,
        bear_argument: str,
        risk_level: str,
    ) -> str:
        schema = '{"action":"BUY|SELL|HOLD","confidence":0.0-1.0,"reasoning":"short rationale"}'
        return (
            "You are the Chief Investment Officer. You have a Bullish analyst and a Bearish analyst. "
            "Your job is to weigh their arguments against live Whale data and decide the final trade. "
            "Be decisive and cynical of hype. "
            "SIMULATION ONLY. "
            "Return ONLY one JSON object with schema: "
            f"{schema}.\n"
            f"Risk level: {risk_level}\n"
            f"Asset: {symbol}\n"
            f"Market data: {json.dumps(coin_data, separators=(',', ':'))}\n"
            f"Whale status: {json.dumps(whale_status, separators=(',', ':'))}\n"
            f"Bull argument: {bull_argument}\n"
            f"Bear argument: {bear_argument}"
        )

    def _get_whale_status(self, symbol: str) -> dict[str, Any]:
        if self.whale_watcher is None:
            return {
                "source": "simulated",
                "summary": f"No whale watcher attached for {symbol}.",
                "net_flow_bias": "neutral",
                "alerts": [],
            }

        get_status = getattr(self.whale_watcher, "get_status", None)
        if callable(get_status):
            status = get_status(symbol=symbol)
            if isinstance(status, dict):
                return status

        get_alerts = getattr(self.whale_watcher, "get_alerts", None)
        if callable(get_alerts):
            alerts = get_alerts(symbol=symbol)
            return {
                "source": "whale_watcher",
                "summary": f"{len(alerts)} whale alerts received.",
                "net_flow_bias": self._infer_whale_bias(alerts),
                "alerts": alerts,
            }

        return {
            "source": "whale_watcher",
            "summary": "Whale watcher attached but no get_status/get_alerts API found.",
            "net_flow_bias": "neutral",
            "alerts": [],
        }

    def _infer_whale_bias(self, alerts: list[dict[str, Any]]) -> str:
        inflow = sum(float(item.get("usd_value", 0.0)) for item in alerts if item.get("direction") == "exchange_in")
        outflow = sum(float(item.get("usd_value", 0.0)) for item in alerts if item.get("direction") == "exchange_out")
        if outflow > inflow:
            return "bullish"
        if inflow > outflow:
            return "bearish"
        return "neutral"

    def _safe_argument(self, provider: str, fetch_fn: Callable[[], str], fallback: str) -> str:
        try:
            return fetch_fn()
        except Exception as exc:
            return f"{fallback} Error: {provider} -> {exc}"

    def _query_groq(self, prompt: str) -> str:
        if not self.groq_api_key:
            raise ValueError("Missing GROQ_API_KEY")
        return self._post_chat_completion(
            url=GROQ_CHAT_URL,
            api_key=self.groq_api_key,
            model=self.groq_bull_model,
            prompt=prompt,
            system_prompt=self.bull_system_prompt,
        )

    def _query_huggingface(self, prompt: str) -> str:
        if not self.hf_api_key:
            raise ValueError("Missing HUGGINGFACE_API_KEY")
        return self._post_chat_completion(
            url=HF_ROUTER_URL,
            api_key=self.hf_api_key,
            model=self.hf_bear_model,
            prompt=prompt,
            system_prompt=self.bear_system_prompt,
        )

    def _query_xai(self, prompt: str) -> str:
        if not self.xai_api_key:
            raise ValueError("Missing XAI_API_KEY")
        return self._post_chat_completion(
            url=XAI_CHAT_URL,
            api_key=self.xai_api_key,
            model=self.xai_judge_model,
            prompt=prompt,
        )

    def _post_chat_completion(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 320,
        }
        response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()

        body = response.json()
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices:
            raise ValueError(f"Unexpected response from {url}: missing choices")

        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))

        text = str(content).strip()
        if not text:
            raise ValueError(f"Empty completion from {url}")
        return text

    def _parse_judge_decision(self, raw_text: str) -> JudgeDecision:
        candidate = raw_text.strip()
        candidate = re.sub(r"^```json\\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"^```\\s*", "", candidate)
        candidate = re.sub(r"```$", "", candidate).strip()

        if not candidate.startswith("{"):
            match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
            if match:
                candidate = match.group(0)

        try:
            parsed = json.loads(candidate)
            action = str(parsed.get("action", "HOLD")).upper()
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", "No reasoning provided.")).strip()
        except Exception:
            action = self._infer_action(raw_text)
            confidence = 0.5
            reasoning = raw_text[:220].strip() or "Fallback from non-JSON judge output."

        if action not in ALLOWED_ACTIONS:
            action = "HOLD"
        confidence = min(1.0, max(0.0, confidence))
        if not reasoning:
            reasoning = "No reasoning provided."

        return JudgeDecision(action=action, confidence=confidence, reasoning=reasoning)

    def _infer_action(self, text: str) -> str:
        lower = f" {text.lower()} "
        if any(token in lower for token in (" buy ", " bullish ", " long ")):
            return "BUY"
        if any(token in lower for token in (" sell ", " bearish ", " short ")):
            return "SELL"
        return "HOLD"




