from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_TZ = "Europe/Paris"
BARCHART_BASE = "https://ondemand.websol.barchart.com"
TIMEOUT = 20

COMMODITIES: Dict[str, Dict[str, str]] = {
    "WTI Crude": {"root": "CL", "sector": "Energy", "unit": "$/bbl", "symbol": "$", "color": "#f6ad55"},
    "Brent Crude": {"root": "CB", "sector": "Energy", "unit": "$/bbl", "symbol": "$", "color": "#ed8936"},
    "Natural Gas": {"root": "NG", "sector": "Energy", "unit": "$/MMBtu", "symbol": "$", "color": "#4299e1"},
    "RBOB Gasoline": {"root": "RB", "sector": "Energy", "unit": "$/gal", "symbol": "$", "color": "#e53e3e"},
    "Heating Oil": {"root": "HO", "sector": "Energy", "unit": "$/gal", "symbol": "$", "color": "#dd6b20"},
    "Gold": {"root": "GC", "sector": "Metals", "unit": "$/oz", "symbol": "$", "color": "#f0b429"},
    "Silver": {"root": "SI", "sector": "Metals", "unit": "$/oz", "symbol": "$", "color": "#a0aec0"},
    "Copper": {"root": "HG", "sector": "Metals", "unit": "$/lb", "symbol": "$", "color": "#fc8181"},
    "Platinum": {"root": "PL", "sector": "Metals", "unit": "$/oz", "symbol": "$", "color": "#b794f4"},
    "Palladium": {"root": "PA", "sector": "Metals", "unit": "$/oz", "symbol": "$", "color": "#4fd1c5"},
    "Corn": {"root": "ZC", "sector": "Agriculture", "unit": "¢/bu", "symbol": "¢", "color": "#68d391"},
    "Wheat": {"root": "ZW", "sector": "Agriculture", "unit": "¢/bu", "symbol": "¢", "color": "#48bb78"},
    "Soybeans": {"root": "ZS", "sector": "Agriculture", "unit": "¢/bu", "symbol": "¢", "color": "#9ae6b4"},
    "Cotton": {"root": "CT", "sector": "Agriculture", "unit": "¢/lb", "symbol": "¢", "color": "#fefcbf"},
}


@dataclass
class ContractRow:
    symbol: str
    contract: str
    expiration_date: Optional[str]
    last_trading_day: Optional[str]
    price: Optional[float]
    settlement: Optional[float]
    previous_close: Optional[float]
    open_interest: Optional[float]
    volume: Optional[float]
    exchange: Optional[str]
    datasource: Optional[str]
    trade_timestamp: Optional[str]

    @property
    def display_price(self) -> Optional[float]:
        if self.price is not None:
            return self.price
        return self.settlement


class BarchartClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _get(self, endpoint: str, **params: Any) -> Dict[str, Any]:
        url = f"{BARCHART_BASE}/{endpoint}.json"
        response = requests.get(
            url,
            params={"apikey": self.api_key, **params},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status", {}).get("code") not in (200, None):
            raise RuntimeError(payload.get("status", {}).get("message", "Unknown Barchart API error"))
        return payload

    def get_expirations(self, root: str) -> List[Dict[str, Any]]:
        payload = self._get(
            "getFuturesExpirations",
            roots=root,
            fields="symbol,contract,exchange,expirationDate,lastTradingDay,firstNoticeDate",
        )
        return payload.get("results", [])

    def get_quotes(self, symbols: List[str], realtime: bool = False) -> List[Dict[str, Any]]:
        if not symbols:
            return []
        endpoint = "getQuote" if realtime else "getQuoteEod"
        payload = self._get(
            endpoint,
            symbols=",".join(symbols),
            fields="symbol,name,lastPrice,close,settlement,previousClose,openInterest,volume,tradeTimestamp,month,year,exchange,datasource",
        )
        return payload.get("results", [])

    def get_history(self, symbol: str, max_records: int = 90) -> List[Dict[str, Any]]:
        payload = self._get(
            "getHistory",
            symbol=symbol,
            type="daily",
            maxRecords=max_records,
            order="asc",
        )
        return payload.get("results", [])


@st.cache_data(ttl=1800, show_spinner=False)
def load_curve(api_key: str, commodity: str, max_contracts: int, realtime: bool) -> pd.DataFrame:
    client = BarchartClient(api_key)
    root = COMMODITIES[commodity]["root"]

    expirations = client.get_expirations(root)
    if not expirations:
        raise RuntimeError(f"No expiration data returned for root {root}.")

    exp_df = pd.DataFrame(expirations)
    keep_cols = [c for c in ["symbol", "contract", "exchange", "expirationDate", "lastTradingDay", "firstNoticeDate"] if c in exp_df.columns]
    exp_df = exp_df[keep_cols].copy()
    if "expirationDate" in exp_df.columns:
        exp_df["expirationDate"] = pd.to_datetime(exp_df["expirationDate"], errors="coerce")
        exp_df = exp_df.sort_values("expirationDate")
    exp_df = exp_df.head(max_contracts)

    symbols = exp_df["symbol"].dropna().astype(str).tolist()
    quotes = client.get_quotes(symbols, realtime=realtime)
    quote_df = pd.DataFrame(quotes)
    if quote_df.empty:
        raise RuntimeError("Quote request returned no results. Check your Barchart subscription permissions.")

    merged = exp_df.merge(quote_df, on="symbol", how="left", suffixes=("_exp", ""))

    if "close" in merged.columns and "lastPrice" not in merged.columns:
        merged["lastPrice"] = merged["close"]

    merged["display_price"] = merged["lastPrice"].combine_first(merged.get("settlement"))
    merged = merged[merged["display_price"].notna()].copy()
    if merged.empty:
        raise RuntimeError("No contracts had a usable price field (lastPrice or settlement).")

    merged = merged.sort_values("expirationDate")
    merged["months_forward"] = range(1, len(merged) + 1)
    merged["curve_pct_vs_front"] = (merged["display_price"] / float(merged.iloc[0]["display_price"]) - 1.0) * 100.0
    return merged.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_front_history(api_key: str, symbol: str) -> pd.DataFrame:
    client = BarchartClient(api_key)
    rows = client.get_history(symbol)
    hist = pd.DataFrame(rows)
    if hist.empty:
        return hist
    date_col = "tradingDay" if "tradingDay" in hist.columns else "timestamp" if "timestamp" in hist.columns else "date"
    hist["date"] = pd.to_datetime(hist[date_col], errors="coerce")
    price_col = "close" if "close" in hist.columns else "lastPrice" if "lastPrice" in hist.columns else None
    if price_col is None:
        return pd.DataFrame()
    hist["price"] = pd.to_numeric(hist[price_col], errors="coerce")
    return hist[["date", "price"]].dropna().sort_values("date")


def compute_structure(prices: List[float]) -> str:
    if len(prices) < 3:
        return "INSUFFICIENT DATA"
    front_slope = (prices[2] - prices[0]) / prices[0]
    if front_slope < -0.005:
        return "BACKWARDATION"
    if front_slope > 0.005:
        return "CONTANGO"
    return "FLAT"



def annualized_roll(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p1 = float(df.iloc[0]["display_price"])
    p2 = float(df.iloc[1]["display_price"])
    return ((p1 - p2) / p2) * 12 * 100.0



def curve_slope(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 0.0
    p1 = float(df.iloc[0]["display_price"])
    pn = float(df.iloc[-1]["display_price"])
    return ((pn - p1) / p1) * 100.0



def spread_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i in range(len(df) - 1):
        near = df.iloc[i]
        far = df.iloc[i + 1]
        near_px = float(near["display_price"])
        far_px = float(far["display_price"])
        spread = far_px - near_px
        annualized = ((near_px - far_px) / far_px) * 12 * 100 if far_px else None
        rows.append(
            {
                "Roll": f"M{i+1}→M{i+2}",
                "Near": near["symbol"],
                "Far": far["symbol"],
                "Near Px": near_px,
                "Far Px": far_px,
                "Spread": spread,
                "Annualized Roll %": annualized,
            }
        )
    return pd.DataFrame(rows)



def style_app() -> None:
    st.set_page_config(page_title="Live Futures Curve Terminal", page_icon="◈", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background-color:#06080f; color:#e2e8f0;}
        [data-testid="stSidebar"] {background-color:#0b0f1a;}
        .metric-card {background:#0b0f1a;border:1px solid rgba(255,255,255,.08);padding:14px;border-radius:8px;text-align:center;}
        .metric-label {font-size:11px;color:#94a3b8;margin-bottom:4px;}
        .metric-value {font-size:22px;font-weight:700;}
        .section {font-size:12px;color:#94a3b8;letter-spacing:.12em;margin:10px 0 8px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )



def metric_card(label: str, value: str, color: str = "#e2e8f0") -> str:
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value" style="color:{color}">{value}</div></div>'



def get_api_key() -> str:
    if "BARCHART_API_KEY" in st.secrets:
        return str(st.secrets["BARCHART_API_KEY"])
    return ""



def main() -> None:
    style_app()

    now = datetime.now(ZoneInfo(APP_TZ))
    default_api_key = get_api_key()

    with st.sidebar:
        st.title("◈ Live Futures Curve")
        st.caption("Barchart-powered Streamlit terminal")
        api_key = st.text_input("Barchart API key", value=default_api_key, type="password")
        sector = st.selectbox("Sector", ["All", "Energy", "Metals", "Agriculture"])
        names = [n for n, meta in COMMODITIES.items() if sector == "All" or meta["sector"] == sector]
        commodity = st.selectbox("Commodity", names, index=0)
        max_contracts = st.slider("How many contracts", 4, 12, 8)
        realtime = st.toggle("Use real-time endpoint", value=False)
        st.caption("If your Barchart plan is delayed/EOD only, leave real-time off.")

    st.markdown(
        f"## Live Futures Curve Terminal\n"
        f"**Time:** {now.strftime('%Y-%m-%d %H:%M %Z')}  \\n"
        f"**Mode:** {'Real-time quotes' if realtime else 'End-of-day / delayed quotes'}"
    )

    st.info(
        "This version uses Barchart contract expirations plus quote data to build a real futures curve. "
        "If you do not add a Barchart API key, it cannot load market data."
    )

    if not api_key:
        st.warning("Add your Barchart API key in the sidebar or Streamlit secrets to load live data.")
        st.stop()

    try:
        curve_df = load_curve(api_key, commodity, max_contracts, realtime)
    except Exception as exc:
        st.error(f"Live data load failed: {exc}")
        st.stop()

    meta = COMMODITIES[commodity]
    prices = curve_df["display_price"].astype(float).tolist()
    structure = compute_structure(prices)
    roll = annualized_roll(curve_df)
    slope = curve_slope(curve_df)
    front_price = float(curve_df.iloc[0]["display_price"])

    structure_color = "#48bb78" if structure == "BACKWARDATION" else "#fc8181" if structure == "CONTANGO" else "#f0b429"
    roll_color = "#48bb78" if roll > 0 else "#fc8181" if roll < 0 else "#f0b429"

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(metric_card("FRONT PRICE", f"{meta['symbol']}{front_price:,.2f}"), unsafe_allow_html=True)
    c2.markdown(metric_card("STRUCTURE", structure, structure_color), unsafe_allow_html=True)
    c3.markdown(metric_card("ANNUALIZED ROLL", f"{roll:+.2f}%", roll_color), unsafe_allow_html=True)
    c4.markdown(metric_card("CURVE SLOPE", f"{slope:+.2f}%", "#94a3b8"), unsafe_allow_html=True)

    left, right = st.columns([2.2, 1.2])

    with left:
        st.markdown("### Forward Curve")
        labels = [f"M{int(x)}" for x in curve_df["months_forward"]]
        fill_color = "rgba(72,187,120,0.08)" if structure == "BACKWARDATION" else "rgba(252,129,129,0.08)"
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=curve_df["display_price"],
                fill="tozeroy",
                fillcolor=fill_color,
                mode="lines",
                line=dict(color="rgba(0,0,0,0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=curve_df["display_price"],
                mode="lines+markers",
                name=commodity,
                line=dict(color=meta["color"], width=3),
                marker=dict(size=7, color=meta["color"]),
                hovertemplate=f"<b>%{{x}}</b><br>{meta['symbol']}%{{y:,.2f}} {meta['unit']}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="#0b0f1a",
            plot_bgcolor="#06080f",
            font=dict(color="#e2e8f0"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=420,
            yaxis_title=meta["unit"],
            xaxis_title="Nearby contracts",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Contract Table")
        show_cols = [
            c for c in [
                "months_forward", "symbol", "contract", "expirationDate", "lastTradingDay", "display_price",
                "settlement", "previousClose", "openInterest", "volume", "exchange", "datasource", "tradeTimestamp"
            ] if c in curve_df.columns
        ]
        display_df = curve_df[show_cols].copy()
        if "expirationDate" in display_df.columns:
            display_df["expirationDate"] = pd.to_datetime(display_df["expirationDate"], errors="coerce").dt.date
        if "lastTradingDay" in display_df.columns:
            display_df["lastTradingDay"] = pd.to_datetime(display_df["lastTradingDay"], errors="coerce").dt.date
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown("### Roll Schedule")
        rolls_df = spread_table(curve_df)
        if not rolls_df.empty:
            styled = rolls_df.copy()
            st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("### Market Read")
        if structure == "BACKWARDATION":
            st.success("Front contracts are priced above later contracts. That usually means the prompt market is tighter than the deferred market.")
        elif structure == "CONTANGO":
            st.error("Later contracts are priced above the front. That usually means carry, storage, or looser prompt conditions are dominating.")
        else:
            st.warning("The front of the curve is relatively flat. The market is not showing a strong carry signal right now.")

        if len(curve_df) >= 2:
            first = curve_df.iloc[0]
            second = curve_df.iloc[1]
            spread = float(second["display_price"]) - float(first["display_price"])
            st.write(f"**M1→M2 spread:** {spread:+.4f}")
        st.write(f"**Contracts loaded:** {len(curve_df)}")
        ds = curve_df.iloc[0].get("datasource")
        if pd.notna(ds):
            st.write(f"**Datasource:** {ds}")

    st.markdown("### Front Contract History")
    front_symbol = str(curve_df.iloc[0]["symbol"])
    try:
        hist = load_front_history(api_key, front_symbol)
        if hist.empty:
            st.info("History endpoint returned no rows for the front contract.")
        else:
            hist_fig = go.Figure()
            hist_fig.add_trace(
                go.Scatter(
                    x=hist["date"],
                    y=hist["price"],
                    mode="lines",
                    line=dict(color=meta["color"], width=2),
                    name=front_symbol,
                )
            )
            hist_fig.update_layout(
                paper_bgcolor="#0b0f1a",
                plot_bgcolor="#06080f",
                font=dict(color="#e2e8f0"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                xaxis_title="Date",
                yaxis_title=meta["unit"],
            )
            st.plotly_chart(hist_fig, use_container_width=True)
    except Exception as exc:
        st.info(f"History section skipped: {exc}")

    st.markdown("### Setup Notes")
    st.write(
        "This app uses Barchart futures expiration data to discover live contracts, then uses Barchart quote data to build the curve from actual symbols. "
        "That means the curve is no longer synthetic."
    )


if __name__ == "__main__":
    main()
