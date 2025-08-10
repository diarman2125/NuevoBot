# NuevoBot.py
from __future__ import annotations
"""
Bot de detección de ERRORES DE CUOTAS (“palp”) listo para Railway.

- Escanea tenis (ATP/WTA/ITF). Fácil de extender a otros deportes.
- Detecta: (a) precio desfasado vs. mercado, (b) cambio brusco, (c) línea atípica.
- Alertas a Telegram en HTML. Anti-duplicados (TTL 10 min).
- Normaliza claves de deportes para evitar 404.
- Valida ODDS_API_KEY antes de iniciar (evita 401 inútiles).
- **Nuevo:** Autodescubre los sports disponibles y filtra para evitar 404 masivos.
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
    TEST_MODE: bool = False   # True = usa datos simulados (no llama APIs)
    RUN_ONCE: bool = False    # True = una pasada y termina

    # (Solo para pruebas locales) — en Railway usa variables de entorno
    ODDS_API_KEY_INLINE: str = ""      # NO usar en prod
    TELEGRAM_TOKEN_INLINE: str = ""    # NO usar en prod
    TELEGRAM_CHAT_ID_INLINE: str = ""  # NO usar en prod

LCFG = LocalConfig()


@dataclass
class Config:
    odds_api_key: str = os.getenv("ODDS_API_KEY", "")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    scan_interval_sec: int = int(os.getenv("SCAN_INTERVAL_SEC", "60"))

    # Umbrales (tuneables)
    umbral_error_pct: float = float(os.getenv("UMBRAL_ERROR_PCT", "30"))
    umbral_cambio_brusco_pct: float = float(os.getenv("UMBRAL_CAMBIO_BRUSCO_PCT", "40"))

    # Umbral mínimo de diferencia de precio para alertar por outlier de línea
    min_price_diff_for_line_alert: float = float(os.getenv("MIN_PRICE_DIFF_FOR_LINE_ALERT", os.getenv("UMBRAL_ERROR_PCT", "30")))

    # Desvíos de líneas (tenis)
    max_desvio_spread: float = float(os.getenv("MAX_DESVIO_SPREAD", "1.5"))
    max_desvio_total: float = float(os.getenv("MAX_DESVIO_TOTAL", "2.0"))

    # Casas a incluir
    bookmakers_filter_csv: str = os.getenv(
        "BOOKMAKERS_FILTER",
        "fanduel,draftkings,betmgm,caesars,betrivers,espnbet,fanatics"
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
            ]
        if self.markets is None:
            self.markets = ["h2h", "spreads", "totals"]

CFG = Config()

# Para pruebas locales si faltan env vars (evitar en prod)
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
    if odds_dec <= 1e-9:
        return 0.0
    return 1.0 / odds_dec

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
        self.session.headers.update({"User-Agent": "error-odds-bot/1.6"})

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
            "regions": self.regions,
            "markets": ",".join(markets),
            "apiKey": self.api_key,
            "oddsFormat": "decimal",
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


# ============================= STATE / CACHE ============================= #

class State:
    def __init__(self):
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


# ============================= DETECTION ============================= #

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
    return float("inf")

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


# ============================= ALERTS ============================= #

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


# ============================= PROCESS ============================= #

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
                price = out.get("price")
                point = out.get("point")
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
            line_has_outlier = False
            if mkt_key == "spreads":
                outlier, dev = detect_line_outlier(mp.line, lines_map.get("spreads", []), CFG.max_desvio_spread)
                line_has_outlier = outlier
                line_info = f"Spread {mp.line:+.1f}{f' (desvío {dev:+.1f})' if outlier else ''}" if mp.line is not None else None
            elif mkt_key == "totals":
                outlier, dev = detect_line_outlier(mp.line, lines_map.get("totals", []), CFG.max_desvio_total)
                line_has_outlier = outlier
                line_info = f"Total {mp.line:.1f}{f' (desvío {dev:+.1f})' if outlier else ''}" if mp.line is not None else None

            line_outlier_alert = line_has_outlier and (diff_pct >= CFG.min_price_diff_for_line_alert)
            should_alert = is_err or changed or line_outlier_alert
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


# ============================= RUN ============================= #

class OddsClientWrapper:
    def __init__(self, odds_client: OddsClient):
        self.odds_client = odds_client

    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        sport = normalize_sport_key(sport)
        if LCFG.TEST_MODE:
            # Datos simulados (1 evento con precio anómalo) para pruebas locales
            return [{
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
            }]
        return self.odds_client.fetch_odds(sport, markets)


def run_once(odds_wrapper: OddsClientWrapper):
    for sport in CFG.sports:
        sport_norm = normalize_sport_key(sport)
        logging.info("Consultando cuotas para sport='%s' (normalizado='%s')", sport, sport_norm)
        try:
            data = odds_wrapper.fetch_odds(sport_norm, CFG.markets)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logging.info("Sin datos para %s (404). Lo omito por ahora.", sport_norm)
                continue
            logging.exception("Error consultando cuotas para %s: %s", sport_norm, e)
            continue
        except Exception as e:
            logging.exception("Error consultando cuotas para %s: %s", sport_norm, e)
            continue

        for event in data:
            try:
                process_event(event, sport_norm)
            except Exception:
                logging.exception("Error procesando evento en %s", sport_norm)


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info("Iniciando bot… TEST_MODE=%s | RUN_ONCE=%s", LCFG.TEST_MODE, LCFG.RUN_ONCE)
    logging.info("Sports config (antes de filtrar): %s", ",".join(CFG.sports))

    # Validación estricta de API key
    if not CFG.odds_api_key or len(CFG.odds_api_key.strip()) == 0:
        logging.critical("ODDS_API_KEY está vacía o no configurada. Configúrala en Railway → Variables.")
        raise SystemExit(1)

    # Autodescubrir sports y usar SOLO los que la API expose para TENIS
    try:
        available_list = ODDS.list_sports()
        # set de claves disponibles
        available = {s.get("key") for s in available_list if isinstance(s, dict) and s.get("key")}
        # 1) Descubrir todos los deportes de tenis vigentes según la API
        discovered_tennis = sorted(k for k in available if k.startswith("tennis"))
        # 2) Si la API devolvió claves de tenis, usarlas; si no, filtrar la lista actual
        if discovered_tennis:
            CFG.sports = discovered_tennis
            logging.info("Sports de TENIS detectados por la API: %s", ", ".join(CFG.sports))
        else:
            wanted = list(CFG.sports)
            CFG.sports = [s for s in wanted if s in available]
            missing = [s for s in wanted if s not in available]
            if missing:
                logging.info("Sports no disponibles (omitidos): %s", ", ".join(missing))
            logging.info("Sports finales: %s", ",".join(CFG.sports))
    except Exception as e:
        logging.exception("No se pudo listar sports; sigo con la lista estática. Error: %s", e)

    logging.info("Markets: %s | Casas: %s", ",".join(CFG.markets), ",".join(sorted(BOOKMAKERS_ALLOW)))
    logging.info("API key length: %s | Telegram token length: %s", len(CFG.odds_api_key or ''), len(CFG.telegram_token or ''))

    odds_wrapper = OddsClientWrapper(ODDS)

    if LCFG.RUN_ONCE:
        run_once(odds_wrapper)
        return

    while True:
        start = time.time()
        run_once(odds_wrapper)
        elapsed = time.time() - start
        sleep_for = max(1, CFG.scan_interval_sec - int(elapsed))
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
