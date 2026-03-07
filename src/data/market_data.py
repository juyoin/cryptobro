"""Phase 1 market data harvester for Crypto Oracle (SIMULATION ONLY)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
DEFAULT_COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
}


class RateLimitError(Exception):
    """Raised when CoinGecko returns HTTP 429."""


class MarketHarvester:
    """Fetches live market data for simulation trading decisions."""

    def __init__(self, timeout_seconds: int = 15) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def _fetch_simple_price(self) -> dict[str, Any]:
        url = f"{COINGECKO_BASE_URL}/simple/price"
        params = {
            "ids": ",".join(DEFAULT_COINS.values()),
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_24hr_vol": "true",
        }

        response = self.session.get(url, params=params, timeout=self.timeout_seconds)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise RateLimitError(f"CoinGecko rate limit hit (HTTP 429). Retry-After={retry_after}")

        response.raise_for_status()
        return response.json()

    def get_market_dossier(self) -> dict[str, Any]:
        payload = self._fetch_simple_price()

        coins: dict[str, dict[str, float | None]] = {}
        for symbol, coin_id in DEFAULT_COINS.items():
            coin_data = payload.get(coin_id, {})
            coins[symbol] = {
                "price_usd": float(coin_data.get("usd", 0.0)),
                "change_24h_pct": (
                    float(coin_data["usd_24h_change"]) if coin_data.get("usd_24h_change") is not None else None
                ),
                "volume_24h_usd": float(coin_data.get("usd_24h_vol", 0.0)),
            }

        return {
            "simulation": True,
            "source": "coingecko",
            "as_of_utc": datetime.now(timezone.utc).isoformat(),
            "coins": coins,
        }


if __name__ == "__main__":
    harvester = MarketHarvester()
    market_dossier = harvester.get_market_dossier()
    print(f"BTC price (USD): {market_dossier['coins']['BTC']['price_usd']}")
