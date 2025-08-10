# -*- coding: utf-8 -*-
from __future__ import annotations
"""
NuevoBot (Prematch Tenis) — Plantilla Diego
- Solo prematch (Odds API /odds endpoint)
- Detección: VALUE BET (outlier vs consenso) y ARBITRAJE (2-way H2H)
- Alertas Telegram con la PLANTILLA OFICIAL de Diego (ver build_value_msg / build_arb_msg)
- Anti-duplicados y TTL
- Libros: solo legales en Indiana (configurable)
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean, median
from datetime import datetime, timezone

import requests
from requests import HTTPError
from dateutil import parser as dtparser
import pytz

# ============================= CONFIG ============================= #

@dataclass
class LocalConfig:
    TEST_MODE: bool = False   # True: usa datos simulados
    RUN_ONCE: bool = False

    ODDS_API_KEY_INLINE: str = ""      # Solo para pruebas locales
    TELEGRAM_TOKEN_INLINE: str = ""
    TELEGRAM_CHAT_ID_INLINE: str = ""

LCFG = LocalConfig()

@dataclass
class Config:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    scan_interval_sec: int = int(os.getenv("SCAN_INTERVAL_SEC", "60"))
    timezone_name: str = os.getenv("LOCAL_TZ", "America/Indiana/Indianapolis")

    # Casas (Indiana)
    bookmakers_filter_csv: str = os.getenv(
        "BOOKMAKERS_FILTER",
        "fanduel,draftkings,betmgm,caesars,betrivers,espnbet,hardrockbet,fanatics"
    )

    # Mercados prematch
    markets: List[str] = None

    # Deportes a escanear (descubiertos automáticamente; fallback a estas claves)
    sports: List[str] = None

    # Umbrales (VALUE y ARBITRAJE)
    min_value_edge_pct: float = float(os.getenv("MIN_VALUE_EDGE_PCT", "15"))   # ≥15% por defecto
    min_arbitrage_edge: float = float(os.getenv("MIN_ARBITRAGE_EDGE", "0.03")) # ≥3% por defecto

    # Detección de errores de línea (adicional)
    umbral_cambio_brusco_pct: float = float(os.getenv("UMBRAL_CAMBIO_BRUSCO_PCT", "40"))
    max_desvio_spread: float = float(os.getenv("MAX_DESVIO_SPREAD", "1.5"))
    max_desvio_total: float = float(os.getenv("MAX_DESVIO_TOTAL", "2.0"))

def _cfg_post_init(self):
    if self.markets is None:
        self.markets = ["h2h", "spreads", "totals"]
    if self.sports is None:
        self.sports = ["tennis_atp", "tennis_wta", "tennis_itf_men", "tennis_itf_women"]
Config.__post_init__ = _cfg_post_init

CFG = Config()

# Para pruebas locales si faltan env vars
if LCFG.ODDS_API_KEY_INLINE and not CFG.odds_api_key:
    CFG.odds_api_key = LCFG.ODDS_API_KEY_INLINE
if LCFG.TELEGRAM_TOKEN_INLINE and not CFG.telegram_token:
    CFG.telegram_token = LCFG.TELEGRAM_TOKEN_INLINE
if LCFG.TELEGRAM_CHAT_ID_INLINE and not CFG.telegram_chat_id:
    CFG.telegram_chat_id = LCFG.TELEGRAM_CHAT_ID_INLINE

BOOKMAKERS_ALLOW = {bk.strip().lower() for bk in CFG.bookmakers_filter_csv.split(",") if bk.strip()}
LOCAL_TZ = pytz.timezone(CFG.timezone_name)

# ============================= UTILS ============================= #

def to_local(dt_str: str) -> str:
    try:
        dt = dtparser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str

def prob_from_decimal(odds_dec: float) -> float:
    return 0.0 if odds_dec <= 1e-9 else 1.0 / odds_dec

def decimal_from_prob(p: float) -> float:
    p = max(min(p, 0.999999), 1e-6)
    return 1.0 / p

def american_from_decimal(dec: float) -> str:
    if dec <= 1.0:
        return "N/A"
    if dec >= 2.0:
        return f"+{int(round((dec - 1) * 100))}"
    else:
        return f"-{int(round(100 / (dec - 1)))}"

def normalize_sport_key(k: str) -> str:
    return (k or "").strip().lower().replace(" ", "_")

# ============================= API CLIENTS ============================= #

class OddsClient:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions: str):
        self.api_key = api_key
        self.regions = regions
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nuevo-bot-prematch/2.0"})

    def list_sports(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE}/sports"
        params = {"apiKey": self.api_key}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        sport = normalize_sport_key(sport)
        url = f"{self.BASE}/sports/{sport}/odds"
        params = {
            "regions": "us",
            "markets": ",".join(markets),
            "apiKey": self.api_key,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()

    def send(self, text: str, disable_preview: bool = True):
        if not self.token or not self.chat_id:
            logging.warning("Telegram no configurado; impresión local:\n%s", text)
            print("\n" + text + "\n")
            return
        url = f"{self.base}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": disable_preview}
        try:
            self.session.post(url, data=data, timeout=15)
        except Exception as e:
            logging.exception("Error enviando mensaje a Telegram: %s", e)

ODDS = OddsClient(CFG.odds_api_key, "us")
TG = Telegram(CFG.telegram_token, CFG.telegram_chat_id)

# ============================= STATE ============================= #

class State:
    def __init__(self):
        self.alert_memory: Dict[str, float] = {}

    def should_suppress(self, dedup_key: str, ttl_sec: int = 600) -> bool:
        now = time.time()
        last = self.alert_memory.get(dedup_key)
        if last and (now - last) < ttl_sec:
            return True
        self.alert_memory[dedup_key] = now
        return False

STATE = State()

# ============================= CORE LOGIC ============================= #

@dataclass
class Quote:
    bookmaker: str
    price_dec: float
    point: Optional[float]

def consensus_prob(prices: List[float], exclude: Optional[float] = None) -> float:
    arr = [p for p in prices if p and p > 1.0 and p != exclude]
    if len(arr) < 2:
        arr = [p for p in prices if p and p > 1.0]
    if not arr:
        return 0.0
    probs = [prob_from_decimal(p) for p in arr]
    return mean(probs)

def find_extremes(quotes: List[Quote]) -> Tuple[Optional[Quote], Optional[Quote]]:
    if not quotes:
        return None, None
    best = max(quotes, key=lambda q: q.price_dec)  # mejor pago
    worst = min(quotes, key=lambda q: q.price_dec)
    return best, worst

def build_value_msg(
    sport_title: str, event_name: str, commence_local: str,
    market: str, selection: str,
    best_q: Quote, worst_q: Optional[Quote],
    implied_best: float, real_prob: float
) -> str:
    # Plantilla oficial Diego (VALUE BET)
    diff_pct = max(0.0, (best_q.price_dec - decimal_from_prob(real_prob)) / decimal_from_prob(real_prob) * 100.0)
    rel_spread = None
    if worst_q:
        rel_spread = (best_q.price_dec - worst_q.price_dec) / max(worst_q.price_dec, 1e-9) * 100.0

    lines = [
        "📢 <b>ALERTA: VALUE BET (Prematch)</b>",
        f"🎾 {sport_title} — {event_name}",
        f"🕒 {commence_local}",
        f"📊 Mercado: {market} | Selección: <b>{selection}</b>",
        f"🏆 Casa con cuota más alta: <b>{best_q.bookmaker}</b> — {best_q.price_dec:.2f} ({american_from_decimal(best_q.price_dec)})",
    ]
    if worst_q:
        lines.append(f"🏷️ Casa con cuota más baja: <b>{worst_q.bookmaker}</b> — {worst_q.price_dec:.2f} ({american_from_decimal(worst_q.price_dec)})")
        lines.append(f"🔁 Diferencia entre cuotas: <b>{rel_spread:.1f}%</b>")

    lines += [
        f"🧮 Probabilidad implícita: <b>{implied_best*100:.1f}%</b>",
        f"📈 Probabilidad real estimada: <b>{real_prob*100:.1f}%</b> (consenso)",
        f"💡 Valor estimado: <b>{diff_pct:.1f}%</b>",
    ]
    return "\n".join(lines)

def build_arb_msg(
    sport_title: str, event_name: str, commence_local: str,
    outcome_a: Tuple[str, Quote], outcome_b: Tuple[str, Quote],
    edge: float, stake_base: float = 100.0
) -> str:
    # Plantilla oficial Diego (ARBITRAJE Prematch)
    name_a, qa = outcome_a
    name_b, qb = outcome_b

    p_sum = prob_from_decimal(qa.price_dec) + prob_from_decimal(qb.price_dec)
    if p_sum <= 0: p_sum = 1.0
    stake_a = stake_base * (prob_from_decimal(qa.price_dec) / p_sum)
    stake_b = stake_base * (prob_from_decimal(qb.price_dec) / p_sum)

    # Pago neto aproximado (ambos lados)
    payoff_a = stake_a * (qa.price_dec - 1.0) - stake_b
    payoff_b = stake_b * (qb.price_dec - 1.0) - stake_a
    guaranteed = min(payoff_a, payoff_b)

    lines = [
        "🧩 <b>ARBITRAJE (Prematch)</b>",
        f"🎾 {sport_title} — {event_name}",
        f"🕒 {commence_local}",
        f"A: <b>{name_a}</b> — {qa.bookmaker} {qa.price_dec:.2f} ({american_from_decimal(qa.price_dec)})",
        f"B: <b>{name_b}</b> — {qb.bookmaker} {qb.price_dec:.2f} ({american_from_decimal(qb.price_dec)})",
        f"📈 Beneficio teórico: <b>{edge*100:.1f}%</b>",
        f"💵 Inversión sugerida (base ${stake_base:.0f}): A = ${stake_a:.2f} | B = ${stake_b:.2f}",
        f"✅ Ganancia neta estimada: <b>${guaranteed:.2f}</b>",
    ]
    return "\n".join(lines)

# ============================= PROCESS ============================= #

def process_event(event: Dict[str, Any]):
    sport_title = event.get("sport_title", "Tennis")
    commence_local = to_local(event.get("commence_time", ""))
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    event_name = f"{away} vs {home}" if home or away else (event.get("id") or "Partido")

    # outcome map: (market_key, outcome_name) -> List[Quote]
    omap: Dict[Tuple[str, str], List[Quote]] = {}
    lines_spreads, lines_totals = [], []

    for bk in event.get("bookmakers", []):
        bk_key = str(bk.get("key", "")).lower()
        if bk_key not in BOOKMAKERS_ALLOW:
            continue
        bk_title = bk.get("title") or bk_key
        for mkt in bk.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in ("h2h","spreads","totals"):
                continue
            for out in mkt.get("outcomes", []):
                name = out.get("name","")
                price = out.get("price")
                point = out.get("point")
                if not price or price <= 1.0:  # decimal inválida
                    continue
                omap.setdefault((mkey, name), []).append(Quote(bk_title, float(price), point))
                if mkey == "spreads" and point is not None:
                    lines_spreads.append(float(point))
                if mkey == "totals" and point is not None:
                    lines_totals.append(float(point))

    # VALUE BET: comparamos mejor cuota vs consenso (prob promedio de todos)
    for (mkey, name), quotes in omap.items():
        prices = [q.price_dec for q in quotes]
        real_p = consensus_prob(prices)  # prob real estimada
        fair_dec = decimal_from_prob(real_p)
        best, worst = find_extremes(quotes)
        if not best: 
            continue
        implied_best = prob_from_decimal(best.price_dec)
        value_edge = (best.price_dec - fair_dec) / max(fair_dec, 1e-9)  # > 0 es valor
        if value_edge >= (CFG.min_value_edge_pct / 100.0):
            dedup = f"VALUE|{event.get('id')}|{mkey}|{name}|{best.bookmaker}|{round(best.price_dec,2)}"
            if not STATE.should_suppress(dedup, ttl_sec=600):
                msg = build_value_msg(
                    sport_title, event_name, commence_local,
                    mkey.upper(), name,
                    best, worst, implied_best, real_p
                )
                TG.send(msg)

    # ARBITRAJE: solo H2H (dos vías)
    # Tomamos mejor cuota de cada outcome; edge = 1 - (1/oddsA + 1/oddsB)
    h2h_outcomes = [(k, v) for (k, v) in omap.items() if k[0] == "h2h"]
    # Reagrupar por nombre de jugador (dos lados)
    names = {}
    for (_, name), quotes in h2h_outcomes:
        best_q, _ = find_extremes(quotes)
        if best_q:
            names[name] = best_q
    if len(names) == 2:
        (name_a, qa), (name_b, qb) = list(names.items())
        s = prob_from_decimal(qa.price_dec) + prob_from_decimal(qb.price_dec)
        edge = 1.0 - s
        if edge >= CFG.min_arbitrage_edge:
            dedup = f"ARB|{event.get('id')}|{round(qa.price_dec,2)}|{qa.bookmaker}|{round(qb.price_dec,2)}|{qb.bookmaker}"
            if not STATE.should_suppress(dedup, ttl_sec=600):
                msg = build_arb_msg(sport_title, event_name, commence_local, (name_a, qa), (name_b, qb), edge, stake_base=100.0)
                TG.send(msg)

# ============================= RUN ============================= #

class OddsClientWrapper:
    def __init__(self, odds_client: OddsClient):
        self.odds_client = odds_client

    def discover_tennis(self) -> List[str]:
        try:
            available_list = self.odds_client.list_sports()
            keys = [s.get("key") for s in available_list if isinstance(s, dict) and s.get("key")]
            tennis = sorted([k for k in keys if str(k).startswith("tennis")])
            return tennis or CFG.sports
        except Exception as e:
            logging.exception("No se pudo listar sports; uso fallback. Error: %s", e)
            return CFG.sports

    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        return self.odds_client.fetch_odds(sport, markets)

def run_once(odds_wrapper: OddsClientWrapper, sports: List[str]):
    for sport in sports:
        sport = normalize_sport_key(sport)
        try:
            data = odds_wrapper.fetch_odds(sport, CFG.markets)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logging.info("Sin datos para %s (404).", sport)
                continue
            logging.exception("HTTP error en %s: %s", sport, e)
            continue
        except Exception as e:
            logging.exception("Error general en %s: %s", sport, e)
            continue

        for event in data:
            try:
                process_event(event)
            except Exception:
                logging.exception("Error procesando evento")

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info("Iniciando NuevoBot (Prematch Tenis) — Plantilla Diego")

    if not CFG.odds_api_key.strip():
        logging.critical("ODDS_API_KEY no configurada.")
        raise SystemExit(1)

    wrapper = OddsClientWrapper(ODDS)
    sports = wrapper.discover_tennis()
    logging.info("Sports a escanear: %s", ", ".join(sports))
    logging.info("Markets: %s | Casas: %s", ",".join(CFG.markets), ",".join(sorted(BOOKMAKERS_ALLOW)))

    if LCFG.RUN_ONCE:
        run_once(wrapper, sports)
        return

    while True:
        start = time.time()
        run_once(wrapper, sports)
        elapsed = time.time() - start
        sleep_for = max(1, CFG.scan_interval_sec - int(elapsed))
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
