
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
NuevoBot (Prematch Tenis) — Plantilla Diego [FIXED defaults]
- Corrige TypeError por CFG.markets None (usa default_factory).
- Solo Prematch.
- Value Bet + Arbitraje (H2H).
- Plantilla oficial de alertas.
"""
import os, time, logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from statistics import mean
from datetime import datetime, timezone
import requests
from requests import HTTPError
from dateutil import parser as dtparser
import pytz

# ============================= CONFIG ============================= #

@dataclass
class LocalConfig:
    TEST_MODE: bool = False
    RUN_ONCE: bool = False
    ODDS_API_KEY_INLINE: str = ""
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
    bookmakers_filter_csv: str = os.getenv(
        "BOOKMAKERS_FILTER",
        "fanduel,draftkings,betmgm,caesars,betrivers,espnbet,hardrockbet,fanatics"
    )
    # FIX: provide defaults via default_factory
    markets: List[str] = field(default_factory=lambda: ["h2h","spreads","totals"])
    sports: List[str]  = field(default_factory=lambda: ["tennis_atp","tennis_wta","tennis_itf_men","tennis_itf_women"])
    min_value_edge_pct: float = float(os.getenv("MIN_VALUE_EDGE_PCT", "15"))
    min_arbitrage_edge: float = float(os.getenv("MIN_ARBITRAGE_EDGE", "0.03"))
    umbral_cambio_brusco_pct: float = float(os.getenv("UMBRAL_CAMBIO_BRUSCO_PCT", "40"))
    max_desvio_spread: float = float(os.getenv("MAX_DESVIO_SPREAD", "1.5"))
    max_desvio_total: float = float(os.getenv("MAX_DESVIO_TOTAL", "2.0"))

CFG = Config()

# For local tests if envs missing
if LCFG.ODDS_API_KEY_INLINE and not CFG.odds_api_key:
    CFG.odds_api_key = LCFG.ODDS_API_KEY_INLINE
if LCFG.TELEGRAM_TOKEN_INLINE and not CFG.telegram_token:
    CFG.telegram_token = LCFG.TELEGRAM_TOKEN_INLINE
if LCFG.TELEGRAM_CHAT_ID_INLINE and not CFG.telegram_chat_id:
    CFG.telegram_chat_id = LCFG.TELEGRAM_CHAT_ID_INLINE

BOOKMAKERS_ALLOW = {bk.strip().lower() for bk in CFG.bookmakers_filter_csv.split(",") if bk.strip()}
LOCAL_TZ = pytz.timezone(CFG.timezone_name)

# ==== Prematch-only filters ====
import os as _os
PREMATCH_GRACE_SEC = int(_os.getenv("PREMATCH_GRACE_SEC", "300"))  # 5 min por defecto

def _parse_iso_utc(dt_str: str):
    try:
        dt = dtparser.isoparse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def is_prematch_event(ev: Dict[str, Any]) -> bool:
    # 1) Excluir si ya inició (con margen negativo) o si falta menos del GRACE (para evitar near-live)
    dt = _parse_iso_utc(ev.get("commence_time",""))
    now_utc = datetime.now(timezone.utc)
    if dt is None:
        return False
    secs_to_start = (dt - now_utc).total_seconds()
    if secs_to_start < PREMATCH_GRACE_SEC * -1:
        # empezó hace más de -grace
        return False
    if secs_to_start < PREMATCH_GRACE_SEC:
        # está por empezar o ya empezó dentro del margen → evitamos
        return False
    # 2) Excluir si el payload trae indicadores de live/in-progress
    for k in ("in_progress","live","completed","status"):
        v = ev.get(k)
        if isinstance(v, bool) and v:
            return False
        if isinstance(v, str) and v.lower() in ("live","in_progress","in-progress","started","ongoing","closed","completed","finished"):
            return False
    # 3) Algunas respuestas incluyen "scores" cuando está en vivo
    if ev.get("scores"):
        return False
    return True


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

class OddsClient:
    BASE = "https://api.the-odds-api.com/v4"
    def __init__(self, api_key: str, regions: str):
        self.api_key = api_key
        self.regions = regions
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "nuevo-bot-prematch/2.1"})
    def list_sports(self) -> List[Dict[str, Any]]:
        url = f"{self.BASE}/sports"
        params = {"apiKey": self.api_key}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        sport = normalize_sport_key(sport)
        url = f"{self.BASE}/sports/{sport}/odds"
        params = {"regions":"us","markets":",".join(markets),"apiKey":self.api_key,"oddsFormat":"decimal","dateFormat":"iso"}
        r = self.session.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token; self.chat_id = chat_id
        self.base = f"https://api.telegram.org/bot{token}"
        self.session = requests.Session()
    def send(self, text: str, disable_preview: bool = True):
        if not self.token or not self.chat_id:
            print("\n" + text + "\n"); return
        url = f"{self.base}/sendMessage"
        data = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": disable_preview}
        self.session.post(url, data=data, timeout=15)

ODDS = OddsClient(CFG.odds_api_key, "us")
TG = Telegram(CFG.telegram_token, CFG.telegram_chat_id)

class State:
    def __init__(self): self.alert_memory: Dict[str, float] = {}
    def should_suppress(self, k: str, ttl_sec: int = 600) -> bool:
        now = time.time(); last = self.alert_memory.get(k)
        if last and (now - last) < ttl_sec: return True
        self.alert_memory[k] = now; return False
STATE = State()

from dataclasses import dataclass
@dataclass
class Quote:
    bookmaker: str
    price_dec: float
    point: Optional[float]

def consensus_prob(prices: List[float]) -> float:
    arr = [p for p in prices if p and p > 1.0]
    if not arr: return 0.0
    probs = [prob_from_decimal(p) for p in arr]
    return sum(probs)/len(probs)

def find_extremes(quotes: List[Quote]) -> Tuple[Optional[Quote], Optional[Quote]]:
    if not quotes: return None, None
    best = max(quotes, key=lambda q: q.price_dec)
    worst = min(quotes, key=lambda q: q.price_dec)
    return best, worst

def build_value_msg(sport_title, event_name, commence_local, market, selection, best_q, worst_q, implied_best, real_prob) -> str:
    fair_dec = decimal_from_prob(real_prob)
    diff_pct = max(0.0, (best_q.price_dec - fair_dec) / max(fair_dec,1e-9) * 100.0)
    lines = [
        "📢 <b>ALERTA: VALUE BET (Prematch)</b>",
        f"🎾 {sport_title} — {event_name}",
        f"🕒 {commence_local}",
        f"📊 Mercado: {market} | Selección: <b>{selection}</b>",
        f"🏆 Casa con cuota más alta: <b>{best_q.bookmaker}</b> — {best_q.price_dec:.2f} ({american_from_decimal(best_q.price_dec)})",
    ]
    if worst_q:
        spread = (best_q.price_dec - worst_q.price_dec)/max(worst_q.price_dec,1e-9)*100.0
        lines.append(f"🏷️ Casa con cuota más baja: <b>{worst_q.bookmaker}</b> — {worst_q.price_dec:.2f} ({american_from_decimal(worst_q.price_dec)})")
        lines.append(f"🔁 Diferencia entre cuotas: <b>{spread:.1f}%</b>")
    lines += [
        f"🧮 Probabilidad implícita: <b>{prob_from_decimal(best_q.price_dec)*100:.1f}%</b>",
        f"📈 Probabilidad real estimada: <b>{real_prob*100:.1f}%</b> (consenso)",
        f"💡 Valor estimado: <b>{diff_pct:.1f}%</b>",
    ]
    return "\n".join(lines)

def build_arb_msg(sport_title, event_name, commence_local, outcome_a, outcome_b, edge, stake_base=100.0) -> str:
    name_a, qa = outcome_a; name_b, qb = outcome_b
    p_sum = prob_from_decimal(qa.price_dec) + prob_from_decimal(qb.price_dec)
    if p_sum <= 0: p_sum = 1.0
    stake_a = stake_base * (prob_from_decimal(qa.price_dec) / p_sum)
    stake_b = stake_base * (prob_from_decimal(qb.price_dec) / p_sum)
    payoff_a = stake_a * (qa.price_dec - 1.0) - stake_b
    payoff_b = stake_b * (qb.price_dec - 1.0) - stake_a
    guaranteed = min(payoff_a, payoff_b)
    return "\n".join([
        "🧩 <b>ARBITRAJE (Prematch)</b>",
        f"🎾 {sport_title} — {event_name}",
        f"🕒 {commence_local}",
        f"A: <b>{name_a}</b> — {qa.bookmaker} {qa.price_dec:.2f} ({american_from_decimal(qa.price_dec)})",
        f"B: <b>{name_b}</b> — {qb.bookmaker} {qb.price_dec:.2f} ({american_from_decimal(qb.price_dec)})",
        f"📈 Beneficio teórico: <b>{edge*100:.1f}%</b>",
        f"💵 Inversión sugerida (base ${stake_base:.0f}): A = ${stake_a:.2f} | B = ${stake_b:.2f}",
        f"✅ Ganancia neta estimada: <b>${guaranteed:.2f}</b>",
    ])

def process_event(event: Dict[str, Any]):
    sport_title = event.get("sport_title", "Tennis")
    commence_local = to_local(event.get("commence_time", ""))
    home = event.get("home_team") or ""
    away = event.get("away_team") or ""
    event_name = f"{away} vs {home}" if (home or away) else (event.get("id") or "Partido")
    omap: Dict[Tuple[str,str], List[Quote]] = {}
    for bk in event.get("bookmakers", []):
        bk_key = str(bk.get("key","")).lower()
        if bk_key not in BOOKMAKERS_ALLOW: continue
        bk_title = bk.get("title") or bk_key
        for mkt in bk.get("markets", []):
            mkey = mkt.get("key")
            if mkey not in ("h2h","spreads","totals"): continue
            for out in mkt.get("outcomes", []):
                name = out.get("name",""); price = out.get("price"); point = out.get("point")
                if not price or price <= 1.0: continue
                omap.setdefault((mkey,name), []).append(Quote(bk_title, float(price), point))
    # VALUE
    for (mkey, name), quotes in omap.items():
        prices = [q.price_dec for q in quotes]
        real_p = consensus_prob(prices)
        fair_dec = decimal_from_prob(real_p)
        best, worst = find_extremes(quotes)
        if not best: continue
        value_edge = (best.price_dec - fair_dec)/max(fair_dec,1e-9)
        if value_edge >= (CFG.min_value_edge_pct/100.0):
            dedup = f"VALUE|{event.get('id')}|{mkey}|{name}|{best.bookmaker}|{round(best.price_dec,2)}"
            if not STATE.should_suppress(dedup, 600):
                TG.send(build_value_msg(sport_title, event_name, commence_local, mkey.upper(), name, best, worst, prob_from_decimal(best.price_dec), real_p))
    # ARB H2H
    h2h = {(k[1]):v for k,v in omap.items() if k[0]=="h2h"}
    if len(h2h)==2:
        (name_a, qa_list), (name_b, qb_list) = list(h2h.items())
        qa, _ = find_extremes(qa_list); qb, _ = find_extremes(qb_list)
        if qa and qb:
            edge = 1.0 - (prob_from_decimal(qa.price_dec) + prob_from_decimal(qb.price_dec))
            if edge >= CFG.min_arbitrage_edge:
                dedup = f"ARB|{event.get('id')}|{round(qa.price_dec,2)}|{qa.bookmaker}|{round(qb.price_dec,2)}|{qb.bookmaker}"
                if not STATE.should_suppress(dedup, 600):
                    TG.send(build_arb_msg(sport_title, event_name, commence_local, (name_a, qa), (name_b, qb), edge, stake_base=100.0))

class OddsClientWrapper:
    def __init__(self, odds_client: OddsClient): self.odds_client = odds_client
    def discover_tennis(self) -> List[str]:
        try:
            lst = self.odds_client.list_sports()
            keys = [s.get("key") for s in lst if isinstance(s, dict) and s.get("key")]
            tennis = sorted([k for k in keys if str(k).startswith("tennis")])
            return tennis or CFG.sports
        except Exception:
            return CFG.sports
    def fetch_odds(self, sport: str, markets: List[str]) -> List[Dict[str, Any]]:
        return self.odds_client.fetch_odds(sport, markets)

def run_once(wrapper: OddsClientWrapper, sports: List[str]):
    for sport in sports:
        try:
            data = wrapper.fetch_odds(sport, CFG.markets)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404: continue
            continue
        except Exception:
            continue
        for ev in data:
            try:
                if not is_prematch_event(ev):
                    continue
                process_event(ev)
            except Exception:
                pass

def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
    logging.info("Iniciando NuevoBot (Prematch Tenis) — Plantilla Diego")
    if not CFG.odds_api_key.strip():
        logging.critical("ODDS_API_KEY no configurada."); raise SystemExit(1)
    wrapper = OddsClientWrapper(ODDS)
    sports = wrapper.discover_tennis()
    # FIX: safe join
    logging.info("Sports a escanear: %s", ", ".join(sports))
    logging.info("Markets: %s | Casas: %s", ",".join(CFG.markets or []), ",".join(sorted({*BOOKMAKERS_ALLOW})))
    while True:
        start = time.time()
        run_once(wrapper, sports)
        elapsed = time.time() - start
        time.sleep(max(1, CFG.scan_interval_sec - int(elapsed)))

if __name__ == "__main__":
    main()
