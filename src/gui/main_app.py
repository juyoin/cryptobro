"""Phase 4 GUI for Crypto Oracle (SIMULATION ONLY)."""

from __future__ import annotations

import os
import queue
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import customtkinter as ctk
from dotenv import load_dotenv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai.brain import AIJury
from src.data.market_data import MarketHarvester
from src.sim.engine import ExecutionEngine


class CryptoOracleApp(ctk.CTk):
    """CustomTkinter desktop dashboard for simulation trading."""

    COINS = ("BTC", "ETH", "SOL")

    def __init__(self) -> None:
        super().__init__()
        load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        # Binance-like dark palette requested by user.
        self.palette = {
            "background": "#0B0E11",
            "surface": "#11161C",
            "surface_alt": "#151B22",
            "border": "#263241",
            "text": "#EAECEF",
            "text_muted": "#8B99A8",
            "primary": "#1f538d",
            "buy": "#0ECB81",
            "sell": "#F6465D",
            "hold": "#3C4755",
        }

        self.title("Crypto Oracle - Simulation")
        self.geometry("1380x860")
        self.minsize(1240, 760)
        self.configure(fg_color=self.palette["background"])

        self.harvester = MarketHarvester()
        self.jury = AIJury()
        self.engine = ExecutionEngine(wallet_path="config/wallet.json", history_log_path="history.log")
        self.update_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.config_lock = threading.Lock()

        self.risk_level = os.getenv("RISK_LEVEL", "medium")
        self.investment_amount_usd = self._safe_float(os.getenv("INVESTMENT_AMOUNT_USD", "250"), default=250.0)
        self.refresh_seconds = max(60, int(self._safe_float(os.getenv("REFRESH_SECONDS", "60"), default=60)))

        self.status_var = ctk.StringVar(value="Status: Bot Idle")
        self.total_value_var = ctk.StringVar(value="$10,000.00")
        self.active_view_var = ctk.StringVar(value="Dashboard")
        self.settings_status_var = ctk.StringVar(value="Settings loaded from environment.")

        self.coin_widgets: dict[str, dict[str, Any]] = {}
        self.portfolio_history: list[tuple[datetime, float]] = []

        self.bot_active = False
        self.heartbeat_frames = ["", ".", "..", "..."]
        self.heartbeat_index = 0

        self._build_layout()
        self._load_initial_wallet_view()
        self._process_queue()
        self._pulse_status()
        self._start_worker()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main_area()

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=220, fg_color=self.palette["surface"], corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(7, weight=1)
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="Crypto Oracle",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=self.palette["buy"],
        ).grid(row=0, column=0, padx=20, pady=(24, 8), sticky="w")

        ctk.CTkLabel(
            sidebar,
            text="SIMULATION ONLY",
            font=ctk.CTkFont(size=12),
            text_color=self.palette["text_muted"],
        ).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        self._nav_button(sidebar, "Dashboard", row=2, command=lambda: self._show_view("Dashboard"))
        self._nav_button(sidebar, "Trade History", row=3, command=lambda: self._show_view("Trade History"))
        self._nav_button(sidebar, "Settings", row=4, command=lambda: self._show_view("Settings"))

        self.bot_toggle_btn = ctk.CTkButton(
            sidebar,
            text="Pause Bot",
            command=self._toggle_bot,
            fg_color=self.palette["primary"],
            hover_color="#2a6bb4",
            text_color=self.palette["text"],
        )
        self.bot_toggle_btn.grid(row=6, column=0, padx=20, pady=(20, 8), sticky="ew")

    def _nav_button(self, parent: ctk.CTkFrame, label: str, row: int, command: Any) -> None:
        ctk.CTkButton(
            parent,
            text=label,
            command=command,
            fg_color=self.palette["surface_alt"],
            hover_color="#1D2732",
            text_color=self.palette["text"],
            anchor="w",
        ).grid(row=row, column=0, padx=20, pady=6, sticky="ew")

    def _build_main_area(self) -> None:
        container = ctk.CTkFrame(self, fg_color=self.palette["background"])
        container.grid(row=0, column=1, sticky="nsew")
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(container, fg_color=self.palette["surface"], height=100, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_propagate(False)
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Total Portfolio Value",
            font=ctk.CTkFont(size=15),
            text_color=self.palette["text_muted"],
        ).grid(row=0, column=0, padx=28, pady=(18, 2), sticky="w")

        ctk.CTkLabel(
            header,
            textvariable=self.total_value_var,
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color=self.palette["text"],
        ).grid(row=1, column=0, padx=28, pady=(0, 16), sticky="w")

        self.status_label = ctk.CTkLabel(
            header,
            textvariable=self.status_var,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.palette["text_muted"],
        )
        self.status_label.grid(row=0, column=1, rowspan=2, padx=28, pady=0, sticky="e")

        self.views_host = ctk.CTkFrame(container, fg_color=self.palette["background"])
        self.views_host.grid(row=1, column=0, sticky="nsew", padx=16, pady=16)
        self.views_host.grid_rowconfigure(0, weight=1)
        self.views_host.grid_columnconfigure(0, weight=1)

        self.dashboard_view = ctk.CTkFrame(self.views_host, fg_color=self.palette["background"])
        self.trade_history_view = ctk.CTkFrame(self.views_host, fg_color=self.palette["background"])
        self.settings_view = ctk.CTkFrame(self.views_host, fg_color=self.palette["background"])
        for frame in (self.dashboard_view, self.trade_history_view, self.settings_view):
            frame.grid(row=0, column=0, sticky="nsew")

        self._build_dashboard_view()
        self._build_trade_history_view()
        self._build_settings_view()
        self._show_view("Dashboard")

    def _build_dashboard_view(self) -> None:
        self.dashboard_view.grid_columnconfigure(0, weight=1)

        cards = ctk.CTkFrame(self.dashboard_view, fg_color=self.palette["background"])
        cards.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 12))
        for i in range(3):
            cards.grid_columnconfigure(i, weight=1)

        for index, symbol in enumerate(self.COINS):
            card = ctk.CTkFrame(
                cards,
                fg_color=self.palette["surface_alt"],
                corner_radius=12,
                border_width=1,
                border_color=self.palette["border"],
            )
            card.grid(row=0, column=index, padx=8, pady=4, sticky="nsew")

            ctk.CTkLabel(
                card,
                text=symbol,
                font=ctk.CTkFont(size=22, weight="bold"),
                text_color=self.palette["text"],
            ).grid(row=0, column=0, padx=16, pady=(16, 6), sticky="w")

            price_var = ctk.StringVar(value="$0.00")
            ctk.CTkLabel(
                card,
                textvariable=price_var,
                font=ctk.CTkFont(size=24),
                text_color=self.palette["text"],
            ).grid(row=1, column=0, padx=16, pady=(0, 12), sticky="w")

            badge = ctk.CTkLabel(
                card,
                text="HOLD",
                fg_color=self.palette["hold"],
                corner_radius=10,
                width=70,
                height=28,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=self.palette["text"],
            )
            badge.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="w")

            self.coin_widgets[symbol] = {"price_var": price_var, "badge": badge}

        bottom = ctk.CTkFrame(self.dashboard_view, fg_color=self.palette["background"])
        bottom.grid(row=1, column=0, sticky="nsew")
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_columnconfigure(1, weight=1)

        decision_frame = ctk.CTkFrame(bottom, fg_color=self.palette["surface"])
        decision_frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8))
        decision_frame.grid_rowconfigure(1, weight=1)
        decision_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            decision_frame,
            text="Decision Log",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=self.palette["buy"],
        ).grid(row=0, column=0, padx=14, pady=(12, 8), sticky="w")

        self.decision_log = ctk.CTkTextbox(
            decision_frame,
            fg_color=self.palette["background"],
            text_color=self.palette["text"],
            border_width=1,
            border_color=self.palette["border"],
        )
        self.decision_log.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.decision_log.configure(state="disabled")

        chart_frame = ctk.CTkFrame(bottom, fg_color=self.palette["surface"])
        chart_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8))

        self.figure = Figure(figsize=(5.4, 3.2), dpi=100, facecolor=self.palette["surface"])
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor(self.palette["surface"])
        self.axis.tick_params(colors=self.palette["text_muted"])
        for spine in self.axis.spines.values():
            spine.set_color(self.palette["border"])
        self.axis.set_title("Portfolio Value Over Time", color=self.palette["text"])
        self.axis.set_ylabel("USD", color=self.palette["text_muted"])

        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _build_trade_history_view(self) -> None:
        self.trade_history_view.grid_rowconfigure(1, weight=1)
        self.trade_history_view.grid_columnconfigure(0, weight=1)

        top = ctk.CTkFrame(self.trade_history_view, fg_color=self.palette["background"])
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            top,
            text="Trade History",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.palette["text"],
        ).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ctk.CTkButton(
            top,
            text="Refresh",
            command=self._refresh_trade_history,
            width=110,
            fg_color=self.palette["primary"],
            hover_color="#2a6bb4",
            text_color=self.palette["text"],
        ).grid(row=0, column=1, padx=6, pady=4, sticky="e")

        self.trade_history_text = ctk.CTkTextbox(
            self.trade_history_view,
            fg_color=self.palette["background"],
            text_color=self.palette["text"],
            border_width=1,
            border_color=self.palette["border"],
        )
        self.trade_history_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.trade_history_text.configure(state="disabled")

    def _build_settings_view(self) -> None:
        self.settings_view.grid_columnconfigure(0, weight=1)
        panel = ctk.CTkFrame(self.settings_view, fg_color=self.palette["surface"])
        panel.grid(row=0, column=0, sticky="nw", padx=8, pady=8)

        ctk.CTkLabel(
            panel,
            text="Settings",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.palette["text"],
        ).grid(row=0, column=0, columnspan=2, padx=16, pady=(16, 14), sticky="w")

        ctk.CTkLabel(panel, text="Gemini API Key", text_color=self.palette["text"]).grid(
            row=1, column=0, padx=16, pady=6, sticky="w"
        )
        self.gemini_entry = ctk.CTkEntry(panel, width=360, show="*", text_color=self.palette["text"])
        self.gemini_entry.grid(row=1, column=1, padx=12, pady=6, sticky="w")
        self.gemini_entry.insert(0, os.getenv("GEMINI_API_KEY", ""))

        ctk.CTkLabel(panel, text="Hugging Face API Key", text_color=self.palette["text"]).grid(
            row=2, column=0, padx=16, pady=6, sticky="w"
        )
        self.hf_entry = ctk.CTkEntry(panel, width=360, show="*", text_color=self.palette["text"])
        self.hf_entry.grid(row=2, column=1, padx=12, pady=6, sticky="w")
        self.hf_entry.insert(0, os.getenv("HUGGINGFACE_API_KEY", ""))

        ctk.CTkLabel(panel, text="Risk Level", text_color=self.palette["text"]).grid(
            row=3, column=0, padx=16, pady=6, sticky="w"
        )
        self.risk_var = ctk.StringVar(value=self.risk_level)
        self.risk_menu = ctk.CTkOptionMenu(
            panel,
            values=["low", "medium", "high"],
            variable=self.risk_var,
            width=160,
            fg_color=self.palette["primary"],
            button_color=self.palette["primary"],
            button_hover_color="#2a6bb4",
            text_color=self.palette["text"],
        )
        self.risk_menu.grid(row=3, column=1, padx=12, pady=6, sticky="w")

        ctk.CTkLabel(panel, text="Investment per Trade (USD)", text_color=self.palette["text"]).grid(
            row=4, column=0, padx=16, pady=6, sticky="w"
        )
        self.investment_entry = ctk.CTkEntry(panel, width=160, text_color=self.palette["text"])
        self.investment_entry.grid(row=4, column=1, padx=12, pady=6, sticky="w")
        self.investment_entry.insert(0, f"{self.investment_amount_usd:.2f}")

        ctk.CTkLabel(panel, text="Refresh Seconds", text_color=self.palette["text"]).grid(
            row=5, column=0, padx=16, pady=6, sticky="w"
        )
        self.refresh_entry = ctk.CTkEntry(panel, width=160, text_color=self.palette["text"])
        self.refresh_entry.grid(row=5, column=1, padx=12, pady=6, sticky="w")
        self.refresh_entry.insert(0, str(self.refresh_seconds))

        ctk.CTkButton(
            panel,
            text="Save Settings",
            command=self._save_settings,
            fg_color=self.palette["primary"],
            hover_color="#2a6bb4",
            text_color=self.palette["text"],
        ).grid(row=6, column=0, columnspan=2, padx=16, pady=(14, 10), sticky="ew")

        ctk.CTkLabel(
            panel,
            textvariable=self.settings_status_var,
            text_color=self.palette["buy"],
        ).grid(row=7, column=0, columnspan=2, padx=16, pady=(0, 16), sticky="w")

    def _show_view(self, view_name: str) -> None:
        self.active_view_var.set(view_name)
        if view_name == "Dashboard":
            self.dashboard_view.tkraise()
        elif view_name == "Trade History":
            self.trade_history_view.tkraise()
            self._refresh_trade_history()
        else:
            self.settings_view.tkraise()

    def _start_worker(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            return
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._bot_loop, daemon=True)
        self.worker_thread.start()
        self.bot_toggle_btn.configure(text="Pause Bot")

    def _stop_worker(self) -> None:
        self.stop_event.set()
        self.bot_toggle_btn.configure(text="Start Bot")

    def _toggle_bot(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self._stop_worker()
            self._set_status(active=False)
            return
        self._start_worker()

    def _bot_loop(self) -> None:
        while not self.stop_event.is_set():
            self.update_queue.put({"type": "status", "active": True})
            try:
                with self.config_lock:
                    risk_level = self.risk_level
                    invest_usd = self.investment_amount_usd
                    refresh_seconds = self.refresh_seconds

                market_dossier = self.harvester.get_market_dossier()
                cycle_lines: list[str] = []
                coin_actions: dict[str, str] = {}

                for symbol in self.COINS:
                    self.update_queue.put({"type": "status", "active": True})
                    per_coin_dossier = {
                        "simulation": market_dossier.get("simulation", True),
                        "source": market_dossier.get("source", "coingecko"),
                        "as_of_utc": market_dossier.get("as_of_utc"),
                        "coins": {symbol: market_dossier["coins"][symbol]},
                    }
                    verdict = self.jury.get_jury_verdict(per_coin_dossier, risk_level=risk_level)
                    result = self.engine.execute_from_jury(
                        coin_symbol=symbol,
                        investment_amount_usd=invest_usd,
                        jury_verdict=verdict,
                        market_dossier=market_dossier,
                    )

                    consensus = verdict.get("consensus", {})
                    action = str(consensus.get("consensus_action", "HOLD")).upper()
                    confidence = float(consensus.get("average_confidence", 0.0))
                    coin_actions[symbol] = action

                    cycle_lines.append(f"{symbol} -> {action} (confidence={confidence:.2f}) | {result.message}")
                    for vote in verdict.get("votes", []):
                        provider = str(vote.get("provider", "model")).title()
                        reason = str(vote.get("reasoning", "")).strip()
                        error = str(vote.get("error", "")).strip()
                        if error:
                            cycle_lines.append(f"{provider}: {reason} [error: {error}]")
                        elif reason:
                            cycle_lines.append(f"{provider}: {reason}")

                wallet = self.engine.load_wallet()
                portfolio_value = self._compute_portfolio_value(wallet=wallet, market_dossier=market_dossier)
                self.update_queue.put(
                    {
                        "type": "cycle",
                        "market_dossier": market_dossier,
                        "coin_actions": coin_actions,
                        "portfolio_value": portfolio_value,
                        "lines": cycle_lines,
                    }
                )
            except Exception as exc:
                self.update_queue.put({"type": "error", "message": f"Worker cycle failed: {exc}"})
                refresh_seconds = 60
            finally:
                self.update_queue.put({"type": "status", "active": False})

            if self.stop_event.wait(refresh_seconds):
                break

    def _process_queue(self) -> None:
        try:
            while True:
                payload = self.update_queue.get_nowait()
                payload_type = payload.get("type")
                if payload_type == "status":
                    self._set_status(active=bool(payload.get("active")))
                elif payload_type == "cycle":
                    self._apply_cycle_update(payload)
                elif payload_type == "error":
                    self._append_decision_lines([payload.get("message", "Unknown error.")])
        except queue.Empty:
            pass
        self.after(300, self._process_queue)

    def _pulse_status(self) -> None:
        if self.bot_active:
            frame = self.heartbeat_frames[self.heartbeat_index % len(self.heartbeat_frames)]
            self.heartbeat_index += 1
            self.status_var.set(f"Status: Bot Active{frame}")
        else:
            self.status_var.set("Status: Bot Idle")
        self.after(500, self._pulse_status)

    def _apply_cycle_update(self, payload: dict[str, Any]) -> None:
        market_dossier = payload["market_dossier"]
        actions = payload["coin_actions"]
        portfolio_value = float(payload["portfolio_value"])
        lines = payload["lines"]

        for symbol in self.COINS:
            coin_data = market_dossier.get("coins", {}).get(symbol, {})
            price_usd = float(coin_data.get("price_usd", 0.0))
            self.coin_widgets[symbol]["price_var"].set(f"${price_usd:,.2f}")
            self._set_badge(symbol=symbol, action=actions.get(symbol, "HOLD"))

        self.total_value_var.set(f"${portfolio_value:,.2f}")
        self.portfolio_history.append((datetime.now(), portfolio_value))
        if len(self.portfolio_history) > 220:
            self.portfolio_history = self.portfolio_history[-220:]
        self._redraw_chart()
        self._append_decision_lines(lines)
        self._refresh_trade_history()

    def _set_badge(self, symbol: str, action: str) -> None:
        badge = self.coin_widgets[symbol]["badge"]
        action_upper = action.upper()
        if action_upper == "BUY":
            badge.configure(text="BUY", fg_color=self.palette["buy"], text_color="#0B0E11")
        elif action_upper == "SELL":
            badge.configure(text="SELL", fg_color=self.palette["sell"], text_color=self.palette["text"])
        else:
            badge.configure(text="HOLD", fg_color=self.palette["hold"], text_color=self.palette["text"])

    def _set_status(self, active: bool) -> None:
        self.bot_active = active
        if active:
            self.status_label.configure(text_color=self.palette["buy"])
        else:
            self.status_label.configure(text_color=self.palette["text_muted"])

    def _append_decision_lines(self, lines: list[str]) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.decision_log.configure(state="normal")
        for line in lines:
            self.decision_log.insert("end", f"[{timestamp}] {line}\n")
        self.decision_log.see("end")
        self.decision_log.configure(state="disabled")

    def _redraw_chart(self) -> None:
        self.axis.clear()
        self.axis.set_facecolor(self.palette["surface"])
        self.axis.tick_params(colors=self.palette["text_muted"])
        for spine in self.axis.spines.values():
            spine.set_color(self.palette["border"])
        self.axis.set_title("Portfolio Value Over Time", color=self.palette["text"])
        self.axis.set_ylabel("USD", color=self.palette["text_muted"])

        if not self.portfolio_history:
            self.chart_canvas.draw_idle()
            return

        values = [entry[1] for entry in self.portfolio_history]
        self.axis.plot(values, color=self.palette["buy"], linewidth=2.2)
        self.axis.grid(color=self.palette["border"], alpha=0.5, linewidth=0.8)

        if len(values) > 1:
            tick_positions = [0, len(values) // 2, len(values) - 1]
            tick_labels = [self.portfolio_history[idx][0].strftime("%H:%M:%S") for idx in tick_positions]
            self.axis.set_xticks(tick_positions, tick_labels)
        else:
            self.axis.set_xticks([0], [self.portfolio_history[0][0].strftime("%H:%M:%S")])

        self.chart_canvas.draw_idle()

    def _refresh_trade_history(self) -> None:
        log_path = Path("history.log")
        if not log_path.exists():
            text = "No trades yet."
        else:
            text = log_path.read_text(encoding="utf-8").strip() or "No trades yet."

        self.trade_history_text.configure(state="normal")
        self.trade_history_text.delete("1.0", "end")
        self.trade_history_text.insert("end", text)
        self.trade_history_text.configure(state="disabled")

    def _save_settings(self) -> None:
        gemini_key = self.gemini_entry.get().strip()
        hf_key = self.hf_entry.get().strip()
        risk_level = self.risk_var.get().strip().lower() or "medium"
        invest_usd = self._safe_float(self.investment_entry.get().strip(), default=250.0)
        refresh_seconds = max(60, int(self._safe_float(self.refresh_entry.get().strip(), default=60)))

        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["HUGGINGFACE_API_KEY"] = hf_key
        self.jury = AIJury()

        with self.config_lock:
            self.risk_level = risk_level
            self.investment_amount_usd = invest_usd
            self.refresh_seconds = refresh_seconds

        self._upsert_env_values(
            {
                "GEMINI_API_KEY": gemini_key,
                "HUGGINGFACE_API_KEY": hf_key,
                "RISK_LEVEL": risk_level,
                "INVESTMENT_AMOUNT_USD": f"{invest_usd}",
                "REFRESH_SECONDS": f"{refresh_seconds}",
            }
        )
        self.settings_status_var.set("Settings saved to .env and applied.")

    def _upsert_env_values(self, updates: dict[str, str]) -> None:
        env_path = Path(".env")
        existing: dict[str, str] = {}

        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                existing[key.strip()] = value.strip()

        existing.update(updates)
        lines = [f"{key}={value}" for key, value in sorted(existing.items())]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_initial_wallet_view(self) -> None:
        try:
            wallet = self.engine.load_wallet()
            starter_dossier = {"coins": {symbol: {"price_usd": 0.0} for symbol in self.COINS}}
            portfolio_value = self._compute_portfolio_value(wallet=wallet, market_dossier=starter_dossier)
            self.total_value_var.set(f"${portfolio_value:,.2f}")
            self._refresh_trade_history()
        except Exception as exc:
            self._append_decision_lines([f"Initial load error: {exc}"])

    @staticmethod
    def _safe_float(raw: str, default: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _compute_portfolio_value(self, wallet: dict[str, Any], market_dossier: dict[str, Any]) -> float:
        total = float(wallet.get("cash_usd", 0.0))
        prices = market_dossier.get("coins", {})
        positions = wallet.get("positions", {})
        for symbol in self.COINS:
            units = float(positions.get(symbol, {}).get("units", 0.0))
            price = float(prices.get(symbol, {}).get("price_usd", 0.0))
            total += units * price
        return total

    def on_close(self) -> None:
        self.stop_event.set()
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.5)
        self.destroy()


def main() -> None:
    app = CryptoOracleApp()
    app.mainloop()


if __name__ == "__main__":
    main()




