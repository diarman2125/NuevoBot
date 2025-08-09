"""
Bot Errores de Cuotas (PALP) – Railway Ready ✅

Un solo archivo para desplegar en Railway. Incluye:
- Escaneo de tenis (ATP/WTA/ITF) + extensible a otros deportes.
- Detección por: (a) precio desfasado vs. mercado, (b) cambio brusco, (c) línea atípica.
- Anti-duplicados (TTL 10 min).
- Alertas a Telegram (HTML).
- TEST_MODE y RUN_ONCE para pruebas locales.

Requisitos (añade en Railway → Variables):
  ODDS_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  (Opcionales) SCAN_INTERVAL_SEC, UMBRAL_ERROR_PCT, UMBRAL_CAMBIO_BRUSCO_PCT, MAX_DESVIO_SPREAD, MAX_DESVIO_TOTAL,
               BOOKMAKERS_FILTER, LOCAL_TZ, ODDS_REGIONS

Instalación (Railway → Deployments → Nixpacks auto):
  - Python se detecta solo.
  - Define Start Command:  python main.py

Notas de seguridad:
  1) Te dejo inline las claves que compartiste para tu prueba inicial. MUY RECOMENDADO moverlas a variables de entorno y borrar/rotar el token luego.
  2) TEST_MODE=False y RUN_ONCE=False para 24/7.
"""
from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean, median
from datetime import datetime, timezone

import logging
import requests
from dateutil import parser as dtparser
import pytz

# ============================= CONFIGURACIÓN ============================= #

@dataclass
class LocalConfig:
    # Cambia a False/False en producción 24/7
    TEST_MODE: bool = False  # True = usa datos simulados y NO llama APIs
    RUN_ONCE: bool = False   # True = corre una pasada y termina

    # (Solo para pruebas rápidas) – RECOMENDADO usar ENV VARS en Railway
    ODDS_API_KEY_INLINE: str = "c3ef41bcf9b41bcc951a8ad2849d5826"   # ← Mover a ODDS_API_KEY
    TELEGRAM_TOKEN_INLINE: str = "7832901058:AAFgN50OSur_N24dGt-v1nwcQ4f3Rf8qsyE"  # ← Mover a TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID_INLINE: str = "5350016908"                     # ← Mover a TELEGRAM_CHAT_ID

LCFG = LocalConfig()


@dataclass
class Config:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    scan_interval_sec: int = int(os.getenv("SCAN_INTERVAL_SEC", "60"))

    # Umbrales de detección (ajustables)
    umbral_error_pct: float = float(os.getenv("UMBRAL_ERROR_PCT", "30"))
    umbral_cambio_brusco_pct: float = float(os.getenv("UMBRAL_CAMBIO_BRUSCO_PCT", "40"))

    # Desvíos líneos (spreads/totales)
    max_desvio_spread: float = float(os.getenv("MAX_DESVIO_SPREAD", "1.5"))
    max_desvio_total: float = float(os.getenv("MAX_DESVIO_TOTAL", "2.0"))

    # Casas a incluir (legales en IN preferentemente)
    bookmakers_filter_csv: str = os.getenv(
        "BOOKMAKERS_FILTER",
        "fanduel,draftkings,betmgm,caesars,betrivers,espnbet,pointsbet"
    )

    # Deportes/ligas a escanear
    sports: List[str] = None
    regions: str = os.getenv("ODDS_REGIONS", "us")
    markets: List[str] = None

    timezone_name: str = os.getenv("LOCAL_TZ", "America/Indiana/Indianapolis")

    def __post_init__(self):
        if self.sports is None:
            self.sports = [
                "tennis_atp",
                "tennis_wta",
                "tennis_itf_men",
                "tennis_itf_women",
                # Puedes agregar más deportes aquí
                # "soccer_usa_mls", "basketball_nba", "baseball_mlb"
            ]
        if self.markets is None:
            self.markets = ["h2h", "spreads", "totals"]

CFG = Config()

# Prioridad a inline si TEST_MODE o si no hay env var (para facilitar pruebas rápidas)
if LCFG.ODDS_API_KEY_INLINE and not CFG.odds_api_key:
    CFG.odds_api_key = LCFG.ODDS_API_KEY_INLINE
if LCFG.TELEGRAM_TOKEN_INLINE and not CFG.telegram_token:
    CFG.telegram_token = LCFG.TELEGRAM_TOKEN_INLINE
if LCFG.TELEGRAM_CHAT_ID_INLINE and not CFG.telegram_chat_id:
    CFG.telegram_chat_id = LCFG.TELEGRAM_CHAT_ID_INLINE

BOOKMAKERS_ALLOW = {bk.strip().lower() for bk in CFG.bookmakers_filter_csv.split(',') if bk.strip()}
LOCAL_TZ = pytz.timezone(CFG.timezone_name)

# ============================= UTILIDADES ============================= #

def to_local(dt_str: str) -> str:
    try:
        dt = dtparser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str

# decimal -> prob implícita

def prob_from_decimal(odds_dec: float) -> float:
    if odds_dec <= 1e-9:
        return 0.0
    return 1.0 / odds_dec

# prob -> decimal

def decimal_from_prob(p: float) -> float:
    p = max(min(p, 0.999999), 1e-6)
    return 1.0 / p

# American odds (para reporte)

def american_from_decimal(dec: float) -> str:
    if dec <= 1.0:
        return "N/A"
    if dec >= 2.0:
        return f"+{int(round((dec - 1) * 100))}"
    else:
        return f"-{int(round(100 / (dec - 1)))}"

# ============================= CLIENTES API ============================= #

class OddsClient:
    BASE = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str, regions: str):
        self.api_key = api_key
        self.regions = regions
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "error-odds-bot/1.3"})

    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        url = f"{self.BASE}/sports/{sport}/odds"
        params = {
            "regions": self.regions,
            "markets": ",".join(markets),
            "apiKey": self.api_key,
            "oddsFormat": "decimal"
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
            logging.warning("Telegram no configurado.")
            return
        url = f"{self.base}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": disable_preview}
        try:
            self.session.post(url, data=data, timeout=15)
        except Exception as e:
            logging.exception("Error enviando mensaje a Telegram: %s", e)

ODDS = OddsClient(CFG.odds_api_key, CFG.regions)
TG = Telegram(CFG.telegram_token, CFG.telegram_chat_id)

# ============================= ESTADO / CACHE ============================= #

class State:
    def __init__(self):
        # key: (sport_key, event_id, market, outcome_name, bookmaker)
        self.last_seen: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        self.alert_memory: Dict[str, float] = {}

    def update_price(self, key: Tuple[str, str, str, str, str], price: float):
        self.last_seen[key] = {"last_price": price, "ts": time.time()}

    def get_last_price(self, key: Tuple[str, str, str, str, str]) -> Optional[float]:
        info = self.last_seen.get(key)
        return info["last_price"] if info else None

    def should_suppress(self, dedup_key: str, ttl_sec: int = 600) -> bool:
        now = time.time()
        last = self.alert_memory.get(dedup_key)
        if last and (now - last) < ttl_sec:
            return True
        self.alert_memory[dedup_key] = now
        return False

STATE = State()

# ============================= DETECCIÓN ============================= #

@dataclass
class MarketPoint:
    bookmaker: str
    price_dec: float
    line: Optional[float]


def consensus_price(outcomes_prices: List[float], exclude: Optional[float] = None) -> float:
    prices = [p for p in outcomes_prices if p and p > 1.0 and p != exclude]
    if len(prices) >= 2:
        probs = [prob_from_decimal(p) for p in prices]
        p_bar = mean(probs)
        return decimal_from_prob(p_bar)
    elif prices:
        return mean(prices)
    return float('inf')


def detect_error_by_price(book_price: float, market_fair: float, umbral_pct: float) -> Tuple[bool, float]:
    if market_fair <= 1.0 or book_price <= 1.0:
        return False, 0.0
    diff = (book_price - market_fair) / market_fair * 100.0
    return (diff >= umbral_pct), diff


def detect_sudden_change(prev: Optional[float], current: float, umbral_pct: float) -> Tuple[bool, float]:
    if not prev or prev <= 0 or current <= 0:
        return False, 0.0
    change = (current - prev) / prev * 100.0
    return (abs(change) >= umbral_pct), change


def detect_line_outlier(this_line: Optional[float], peer_lines: List[float], max_dev: float) -> Tuple[bool, float]:
    if this_line is None:
        return False, 0.0
    peers = [l for l in peer_lines if l is not None]
    if len(peers) < 2:
        return False, 0.0
    med = median(peers)
    dev = this_line - med
    return (abs(dev) >= max_dev), dev

# ============================= ALERTAS ============================= #

def fmt_alert(
    sport: str,
    league: str,
    commence: str,
    event_name: str,
    market: str,
    selection: str,
    bookmaker: str,
    price_dec: float,
    market_fair: float,
    diff_pct: float,
    change_pct: Optional[float],
    line_info: Optional[str],
) -> str:
    price_amer = american_from_decimal(price_dec)
    fair_amer = american_from_decimal(market_fair)

    lines = [
        "🚨 <b>ERROR DE CUOTA DETECTADO</b>",
        f"Deporte: {sport.upper()} – {league}",
        f"Partido: {event_name}",
        f"Hora: {commence}",
        f"Mercado: {market}{' – ' + line_info if line_info else ''}",
        f"Casa: <b>{bookmaker}</b>",
        f"Cuota: <b>{price_dec:.2f}</b> ({price_amer})",
        f"Promedio de mercado: <b>{market_fair:.2f}</b> ({fair_amer})",
        f"Diferencia: <b>{diff_pct:.1f}%</b>",
    ]
    if change_pct is not None:
        lines.append(f"Cambio vs último visto: <b>{change_pct:+.1f}%</b>")

    lines.append("\n⚠️ Posible 'palp'. Stake moderado: la casa puede anular si confirma error.")
    return "\n".join(lines)

# ============================= PROCESAMIENTO ============================= #

def process_event(event: Dict[str, Any], sport_key: str):
    sport_title = event.get("sport_title", "")
    commence = to_local(event.get("commence_time", ""))
    event_id = event.get("id")

    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    event_name = f"{home} vs {away}" if home or away else event_id

    outcomes_map: Dict[Tuple[str, str], List[MarketPoint]] = {}
    lines_map: Dict[str, List[float]] = {"spreads": [], "totals": []}

    for bk in event.get("bookmakers", []):
        bk_key = str(bk.get("key", "")).lower()
        if bk_key not in BOOKMAKERS_ALLOW:
            continue
        bk_title = bk.get("title") or bk_key
        for mkt in bk.get("markets", []):
            mkt_key = mkt.get("key")  # h2h, spreads, totals
            outcomes = mkt.get("outcomes", [])
            for out in outcomes:
                name = out.get("name", "")
                price = out.get("price")  # decimal
                point = out.get("point")   # spread/total line
                if not price or price <= 1.0:
                    continue
                outcomes_map.setdefault((mkt_key, name), []).append(
                    MarketPoint(bookmaker=bk_title, price_dec=float(price), line=point)
                )
                if mkt_key in ("spreads", "totals") and point is not None:
                    lines_map[mkt_key].append(float(point))

    for (mkt_key, name), points in outcomes_map.items():
        all_prices = [p.price_dec for p in points]
        for mp in points:
            fair = consensus_price(all_prices, exclude=mp.price_dec)
            is_err, diff_pct = detect_error_by_price(mp.price_dec, fair, CFG.umbral_error_pct)

            prev_key = (sport_key, event_id, mkt_key, name, mp.bookmaker)
            prev_price = STATE.get_last_price(prev_key)
            changed, change_pct = detect_sudden_change(prev_price, mp.price_dec, CFG.umbral_cambio_brusco_pct)
            STATE.update_price(prev_key, mp.price_dec)

            line_info = None
            if mkt_key == "spreads":
                outlier, dev = detect_line_outlier(mp.line, lines_map.get("spreads", []), CFG.max_desvio_spread)
                line_info = f"Spread {mp.line:+.1f}{f' (desvío {dev:+.1f})' if outlier else ''}" if mp.line is not None else None
            elif mkt_key == "totals":
                outlier, dev = detect_line_outlier(mp.line, lines_map.get("totals", []), CFG.max_desvio_total)
                line_info = f"Total {mp.line:.1f}{f' (desvío {dev:+.1f})' if outlier else ''}" if mp.line is not None else None

            should_alert = is_err or changed or (line_info and "desvío" in line_info)
            if should_alert:
                dedup_key = f"{event_id}|{mkt_key}|{name}|{mp.bookmaker}|{round(mp.price_dec,2)}"
                if STATE.should_suppress(dedup_key):
                    continue
                msg = fmt_alert(
                    sport=sport_key,
                    league=sport_title,
                    commence=commence,
                    event_name=event_name,
                    market=mkt_key.upper(),
                    selection=name,
                    bookmaker=mp.bookmaker,
                    price_dec=mp.price_dec,
                    market_fair=fair,
                    diff_pct=diff_pct,
                    change_pct=(change_pct if changed else None),
                    line_info=line_info,
                )
                TG.send(msg)

# ============================= EJECUCIÓN ============================= #

def fetch_odds_wrapper(sport: str, markets: List[str]) -> List[Dict[str, Any]]:
    if LCFG.TEST_MODE:
        # Datos simulados mínimos (1 evento, h2h con una cuota errónea grande)
        return [
            {
                "id": "evt123",
                "sport_title": "ATP Sample",
                "commence_time": datetime.now(timezone.utc).isoformat(),
                "home_team": "Jugador A",
                "away_team": "Jugador B",
                "bookmakers": [
                    {"key": "fanduel", "title": "FanDuel", "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Jugador A", "price": 7.00},
                            {"name": "Jugador B", "price": 1.45},
                        ]}
                    ]},
                    {"key": "draftkings", "title": "DraftKings", "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Jugador A", "price": 3.80},
                            {"name": "Jugador B", "price": 1.28},
                        ]}
                    ]},
                    {"key": "betmgm", "title": "BetMGM", "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Jugador A", "price": 3.70},
                            {"name": "Jugador B", "price": 1.30},
                        ]}
                    ]},
                ]
            }
        ]
    else:
        return ODDS.fetch_odds(sport, markets)


def run_once():
    for sport in CFG.sports:
        try:
            data = fetch_odds_wrapper(sport, CFG.markets)
        except Exception as e:
            logging.exception("Error consultando cuotas para %s: %s", sport, e)
            continue
        for event in data:
            try:
                process_event(event, sport)
            except Exception:
                logging.exception("Error procesando evento en %s", sport)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info("Iniciando bot de errores de cuotas… TEST_MODE=%s | RUN_ONCE=%s", LCFG.TEST_MODE, LCFG.RUN_ONCE)
    logging.info("Sports: %s | Markets: %s | Casas: %s", ",".join(CFG.sports), ",".join(CFG.markets), ",".join(sorted(BOOKMAKERS_ALLOW)))

    if LCFG.RUN_ONCE:
        run_once()
        return

    while True:
        start = time.time()
        run_once()
        elapsed = time.time() - start
        sleep_for = max(1, CFG.scan_interval_sec - int(elapsed))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
