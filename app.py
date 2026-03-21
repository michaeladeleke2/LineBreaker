"""
linebreaker/app.py — LineBreaker: NBA & NFL Prop Predictor
Run: streamlit run app.py
"""
import sys, warnings, math
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from models.predict import predict, get_players_for_ui, get_teams_for_ui, get_available_targets, TARGET_DISPLAY, TARGET_GROUPS
from features.engineer import DEFAULT_THRESHOLDS, COMBO_TARGETS
from data.fetch_data import fetch_today_slate, fetch_team_records, refresh_current_season
try:
    from data.fetch_lines import compute_edge
    from data.accuracy_tracker import log_prediction, resolve_predictions, get_accuracy_stats, get_recent_predictions
    TRACKING_ENABLED = True
except ImportError:
    TRACKING_ENABLED = False
    def compute_edge(*a, **k): return {}
    def log_prediction(*a, **k): return {}
    def resolve_predictions(*a, **k): return 0
    def get_accuracy_stats(*a, **k): return {'total':0,'correct':0,'accuracy':0.0,'streak':0,'by_target':{},'recent_30d':{'total':0,'correct':0,'accuracy':0.0}}
    def get_recent_predictions(*a, **k): return []
from data.fetch_injuries import fetch_injury_report, get_player_injury
from data.fetch_lineups import fetch_all_lineups, get_player_lineup_status
try:
    from data.fetch_nfl import (
        fetch_nfl_teams, fetch_all_nfl_players, fetch_nfl_gamelog,
        predict_nfl_prop, get_nfl_player_headshot, fetch_nfl_today_slate,
        NFL_TARGET_DISPLAY, NFL_TARGET_GROUPS, NFL_THRESHOLDS, NFL_TEAM_COLORS, NFL_SEASON,
    )
    NFL_ENABLED = True
except ImportError:
    NFL_ENABLED = False

TEAM_COLORS = {
    "ATL":"#E03A3E","BOS":"#007A33","BKN":"#AAAAAA","CHA":"#00788C",
    "CHI":"#CE1141","CLE":"#860038","DAL":"#00538C","DEN":"#FEC524",
    "DET":"#C8102E","GSW":"#1D428A","HOU":"#CE1141","IND":"#002D62",
    "LAC":"#C8102E","LAL":"#FDB927","MEM":"#5D76A9","MIA":"#98002E",
    "MIL":"#00471B","MIN":"#236192","NOP":"#0C2340","NYK":"#F58426",
    "OKC":"#007AC1","ORL":"#0077C0","PHI":"#006BB6","PHX":"#E56020",
    "POR":"#E03A3E","SAC":"#5A2D81","SAS":"#C4CED4","TOR":"#CE1141",
    "UTA":"#002B5C","WAS":"#002B5C",
}

def _headshot(pid): return f"https://cdn.nba.com/headshots/nba/latest/1040x760/{pid}.png"

st.set_page_config(page_title="LineBreaker", page_icon="🏀",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap');
*, html, body, [class*="css"] { font-family:'Inter',sans-serif; box-sizing:border-box; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:0 !important; max-width:100% !important; }

/* Force dark everywhere */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stMain"] { background:#080810 !important; }

/* Nav */
.lb-nav {
    display:flex; align-items:center; justify-content:space-between;
    background:#080810; border-bottom:1px solid #13131f;
    padding:0 2rem; height:52px; position:sticky; top:0; z-index:100;
}
.lb-logo { font-family:'Bebas Neue',sans-serif; font-size:1.9rem; color:#f0672a; letter-spacing:0.05em; }
.lb-ticker-wrap { flex:1; overflow:hidden; margin:0 1.5rem;
    mask-image:linear-gradient(90deg,transparent,black 8%,black 92%,transparent); }
.lb-ticker-track { display:inline-flex; gap:2rem; white-space:nowrap; animation:tick 40s linear infinite; }
@keyframes tick { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
.lb-tick { font-size:0.68rem; font-weight:500; color:#252535; letter-spacing:0.05em; }
.lb-tick .h { color:#f0672a; font-weight:700; }
.lb-tick .u { color:#4caf82; font-weight:700; }
.lb-nav-right { display:flex; flex-direction:column; align-items:flex-end; gap:2px; }
.lb-badge { font-size:0.58rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
    color:#f0672a; border:1px solid rgba(240,103,42,0.35); border-radius:4px; padding:0.18rem 0.5rem; }
.lb-tagline { font-size:0.48rem; font-weight:600; letter-spacing:0.18em; text-transform:uppercase; color:#1a1a28; }

/* Tabs flush */
[data-testid="stTabs"] { margin-top:0 !important; }
[data-testid="stTabs"] > div:first-child {
    background:#080810 !important; border-bottom:1px solid #13131f;
    padding:0 2rem; margin-top:0 !important; }
div[data-testid="stTabsContent"] > div { padding-top:0 !important; margin-top:0 !important; }
[data-testid="stTabsContent"] { padding:0 !important; }
[data-testid="stVerticalBlock"] > div:empty { display:none !important; }
button[data-baseweb="tab"] {
    font-size:0.72rem !important; font-weight:600 !important; letter-spacing:0.1em !important;
    text-transform:uppercase !important; color:#2a2a3a !important;
    padding:0.75rem 1.1rem !important; border-radius:0 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f0672a !important; border-bottom:2px solid #f0672a !important; }

/* Body */
.lb-body { padding:1.2rem 2rem 3rem; background:#080810; }

/* Slate */
.slate-hdr { font-size:0.58rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase;
    color:#2a2a3a; margin-bottom:0.6rem; display:block; }

/* Controls panel */
.ctrl-panel { background:#0d0d15; border:1px solid #13131f; border-radius:14px; padding:1.3rem 1.4rem 1.5rem; }
.ctrl-title { font-size:0.58rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase;
    color:#f0672a; margin-bottom:1rem; padding-bottom:0.6rem; border-bottom:1px solid #111118; }
.ctrl-label { font-size:0.6rem; font-weight:700; letter-spacing:0.14em; text-transform:uppercase;
    color:#252535; margin-bottom:0.2rem; display:block; margin-top:0.8rem; }

/* Widget overrides */
div[data-baseweb="select"] > div {
    background:#111118 !important; border-color:#1c1c28 !important;
    border-radius:8px !important; color:#e8e6e0 !important; font-size:0.85rem !important; }
.stSelectbox label, .stTextInput label, .stRadio label, .stSlider label {
    font-size:0.6rem !important; font-weight:700 !important; letter-spacing:0.14em !important;
    text-transform:uppercase !important; color:#252535 !important; }
[data-testid="stRadio"] label p { color:#c8c6c0 !important; }
[data-testid="stSlider"] > div > div > div > div { background:#f0672a !important; }
[data-testid="stTickBarMin"],[data-testid="stTickBarMax"] { color:#c8c6c0 !important; background:transparent !important; }
[data-testid="stTextInput"] input { background:#111118 !important; border-color:#1c1c28 !important; color:#e8e6e0 !important; border-radius:8px !important; }
[data-testid="stExpander"] { background:#080810 !important; border:1px solid #0d0d15 !important; border-radius:8px !important; }

/* Buttons — primary (main actions) */
[data-testid="stButton"] button[kind="primary"] {
    background:#f0672a !important; border:none !important;
    font-family:'Bebas Neue',sans-serif !important; font-size:1.15rem !important;
    letter-spacing:0.12em !important; border-radius:10px !important;
    width:100% !important; padding:0.65rem !important; margin-top:0.6rem !important; }
/* Secondary buttons — visible subtle dark style */
[data-testid="stButton"] button[kind="secondary"] {
    background:#0d0d15 !important; border:1px solid #1a1a28 !important;
    color:#2e2e42 !important; border-radius:6px !important;
    font-size:0.65rem !important; font-weight:600 !important;
    letter-spacing:0.08em !important; padding:0.35rem 0.5rem !important;
    margin-top:0.25rem !important; width:100% !important; }
[data-testid="stButton"] button[kind="secondary"]:hover {
    background:#111118 !important; border-color:#252535 !important;
    color:#505068 !important; }

/* Sidebar off */
[data-testid="stSidebar"] { display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _lineup_badge(lineup_info):
    """Show starter/bench badge next to player name."""
    if not lineup_info: return ""
    role      = lineup_info.get("role","unknown")
    confirmed = lineup_info.get("confirmed", False)
    if role in ("unknown","starter"): return ""  # starters need no badge, unknown = no info
    colors = {
        "bench":    ("#4a4a5a","rgba(74,74,90,0.12)"),
        "sixth_man":("#d4b44a","rgba(212,180,74,0.1)"),
        "inactive": ("#e05a5a","rgba(224,90,90,0.15)"),
    }
    tc,bg  = colors.get(role,("#4a4a5a","rgba(74,74,90,0.1)"))
    lbl    = ("6th Man" if role=="sixth_man" else role.title())
    proj   = "" if confirmed else " (proj)"
    return (f'&nbsp;&middot;&nbsp;<span style="background:{bg};color:{tc};border:1px solid {tc}44;'
            f'border-radius:4px;padding:1px 5px;font-size:9px;font-weight:700;letter-spacing:1px;'
            f'text-transform:uppercase;vertical-align:middle;white-space:nowrap;">{lbl}{proj}</span>')


def _inj_badge(inj_info):
    if not inj_info: return ""
    status = inj_info.get("status","active").lower()
    desc   = inj_info.get("description","")
    if status in ("active","healthy","unknown",""): return ""
    c = {"out":("#e05a5a","rgba(224,90,90,0.15)"),
         "doubtful":("#e05a5a","rgba(224,90,90,0.1)"),
         "questionable":("#d4b44a","rgba(212,180,74,0.12)"),
         "day-to-day":("#d4b44a","rgba(212,180,74,0.1)"),
         "probable":("#4caf82","rgba(76,175,130,0.1)")}
    tc,bg = c.get(status,("#d4b44a","rgba(212,180,74,0.1)"))
    lbl   = status.upper().replace("-"," ")
    tip   = f' title="{desc}"' if desc else ""
    return (f'&nbsp;&middot;&nbsp;<span{tip} style="background:{bg};color:{tc};border:1px solid {tc}44;'
            f'border-radius:4px;padding:1px 5px;font-size:9px;font-weight:700;letter-spacing:1px;'
            f'text-transform:uppercase;vertical-align:middle;white-space:nowrap;">{lbl}</span>')

def _svg_court():
    return """<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg" style="width:100%;opacity:0.035;">
  <rect width="400" height="220" fill="none" stroke="#e8e6e0" stroke-width="2"/>
  <line x1="200" y1="0" x2="200" y2="220" stroke="#e8e6e0" stroke-width="1.5"/>
  <circle cx="200" cy="110" r="45" fill="none" stroke="#e8e6e0" stroke-width="1.5"/>
  <circle cx="200" cy="110" r="6" fill="#e8e6e0"/>
  <rect x="0" y="60" width="90" height="100" fill="none" stroke="#e8e6e0" stroke-width="1.5"/>
  <rect x="310" y="60" width="90" height="100" fill="none" stroke="#e8e6e0" stroke-width="1.5"/>
  <path d="M90 60 Q140 60 140 110 Q140 160 90 160" fill="none" stroke="#e8e6e0" stroke-width="1.5"/>
  <path d="M310 60 Q260 60 260 110 Q260 160 310 160" fill="none" stroke="#e8e6e0" stroke-width="1.5"/>
</svg>"""

def _svg_field():
    return """<svg viewBox="0 0 360 200" xmlns="http://www.w3.org/2000/svg" style="width:100%;opacity:0.06;">
  <rect width="360" height="200" fill="none" stroke="#4caf82" stroke-width="2"/>
  <rect x="0" y="0" width="36" height="200" fill="none" stroke="#4caf82" stroke-width="1.5"/>
  <rect x="324" y="0" width="36" height="200" fill="none" stroke="#4caf82" stroke-width="1.5"/>
  <line x1="180" y1="0" x2="180" y2="200" stroke="#4caf82" stroke-width="1"/>
  <line x1="72" y1="0" x2="72" y2="200" stroke="#4caf82" stroke-width="0.5" opacity="0.5"/>
  <line x1="144" y1="0" x2="144" y2="200" stroke="#4caf82" stroke-width="0.5" opacity="0.5"/>
  <line x1="216" y1="0" x2="216" y2="200" stroke="#4caf82" stroke-width="0.5" opacity="0.5"/>
  <line x1="288" y1="0" x2="288" y2="200" stroke="#4caf82" stroke-width="0.5" opacity="0.5"/>
</svg>"""

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_players(): return get_players_for_ui(active_only=True)
@st.cache_data(show_spinner=False)
def load_teams(): return get_teams_for_ui()
@st.cache_data(show_spinner=False)
def load_targets(): return get_available_targets()
@st.cache_data(show_spinner=False, ttl=300)
def load_slate(): return fetch_today_slate()
@st.cache_data(show_spinner=False, ttl=3600)
def load_records(): return fetch_team_records()
@st.cache_data(show_spinner=False, ttl=1800)
def load_injuries():
    try: return fetch_injury_report()
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=1800)
def load_lineups():
    try: return fetch_all_lineups()
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_nfl_teams():
    try: return fetch_nfl_teams() if NFL_ENABLED else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def load_nfl_players():
    try: return fetch_all_nfl_players() if NFL_ENABLED else pd.DataFrame()
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=300)
def load_nfl_slate():
    try: return fetch_nfl_today_slate() if NFL_ENABLED else []
    except: return []

@st.cache_data(show_spinner=False)
def _compute_picks_cached(conf: str, top_n: int, seed: int) -> pd.DataFrame:
    """Cache picks by seed so reruns don't restart computation."""
    try:
        from models.quick_picks import generate_quick_picks
        min_conf = None if conf == "All" else conf
        return generate_quick_picks(top_n=top_n, min_confidence=min_conf)
    except Exception as e:
        return pd.DataFrame()


if "season_refreshed" not in st.session_state:
    # Show branded overlay — hides all Streamlit content behind it
    overlay = st.empty()
    overlay.markdown(
        "<style>"
        "@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@500;700&display=swap');"
        "*, *::before, *::after { visibility:hidden !important; }"
        ".lb-ov, .lb-ov *, .lb-ov *::before, .lb-ov *::after { visibility:visible !important; }"
        "[data-testid='stSpinner'] { display:none !important; }"
        ".lb-ov{position:fixed;top:0;left:0;width:100vw;height:100vh;background:#080810;z-index:9999;"
        "display:flex;flex-direction:column;align-items:center;justify-content:center;}"
        ".lb-ov-logo{font-family:'Bebas Neue',sans-serif;font-size:3.5rem;color:#f0672a;"
        "letter-spacing:0.08em;line-height:1;margin-bottom:0.4rem;}"
        ".lb-ov-sub{font-size:0.58rem;font-weight:700;letter-spacing:0.22em;text-transform:uppercase;"
        "color:#1e1e28;margin-bottom:2.5rem;}"
        ".lb-ov-track{width:240px;height:2px;background:#13131f;border-radius:2px;overflow:hidden;margin-bottom:0.9rem;}"
        ".lb-ov-fill{height:100%;width:0%;background:#f0672a;border-radius:2px;animation:lbfill 4s ease-in-out forwards;}"
        "@keyframes lbfill{0%{width:0%}15%{width:18%}40%{width:45%}70%{width:72%}90%{width:88%}100%{width:100%}}"
        ".lb-ov-msg{font-size:0.6rem;font-weight:500;letter-spacing:0.1em;color:#252535;animation:lbpulse 1.8s ease-in-out infinite;}"
        "@keyframes lbpulse{0%,100%{opacity:1}50%{opacity:0.35}}"
        "</style>"
        "<div class='lb-ov'>"
        "<div class='lb-ov-logo'>LineBreaker</div>"
        "<div class='lb-ov-sub'>Beat the Line. Break the Line.</div>"
        "<div class='lb-ov-track'><div class='lb-ov-fill'></div></div>"
        "<div class='lb-ov-msg'>Syncing latest game data&hellip;</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    # Do ALL loading while overlay is showing
    refresh_current_season()
    players_df    = load_players()
    teams_df      = load_teams()
    avail_targets = load_targets()
    today_slate   = load_slate()
    team_records  = load_records()
    injury_df     = load_injuries()
    lineup_df     = load_lineups()
    st.session_state["season_refreshed"] = True
    overlay.empty()  # Remove overlay — page appears fully loaded
else:
    players_df    = load_players()
    teams_df      = load_teams()
    avail_targets = load_targets()
    today_slate   = load_slate()
    team_records  = load_records()
    injury_df     = load_injuries()
    lineup_df     = load_lineups()

player_options = players_df["full_name"].tolist()
team_options   = teams_df["team_abbreviation"].tolist()

stat_options, stat_map = [], {}
for group in TARGET_GROUPS:
    for target, info in TARGET_DISPLAY.items():
        if info["group"] == group and target in avail_targets:
            d = f"{info['label']} ({info['short']})"
            stat_options.append(d)
            stat_map[d] = target

# ── Ticker — built from real player averages ─────────────────────────────────
def _build_ticker(_players_df):
    """Build ticker from actual current-season averages read from the feature matrix."""
    stars = [
        ("Shai Gilgeous-Alexander", "pts"),
        ("Nikola Jokic",            "pra"),
        ("Luka Doncic",             "pts"),
        ("Giannis Antetokounmpo",   "reb"),
        ("Jayson Tatum",            "pts"),
        ("Stephen Curry",           "fg3m"),
        ("LeBron James",            "pts"),
        ("Victor Wembanyama",       "blk"),
        ("Anthony Davis",           "blk"),
        ("Kevin Durant",            "pts"),
        ("Tyrese Haliburton",       "ast"),
        ("Donovan Mitchell",        "pts"),
    ]
    LABELS = {"pts":"PTS","reb":"REB","ast":"AST","blk":"BLK","fg3m":"3PM","pra":"PRA"}
    ticks = []
    try:
        fm_path = ROOT / "data" / "cache" / "feature_matrix.csv"
        if fm_path.exists():
            fm = pd.read_csv(fm_path, usecols=["season_year","player_name","pts","reb","ast","blk","fg3m"])
            cur = fm["season_year"].max()
            fm  = fm[fm["season_year"] == cur].copy()
            fm["pra"] = fm["pts"] + fm["reb"] + fm["ast"]
            for name, stat in stars:
                last = name.split()[-1]
                sub  = fm[fm["player_name"].str.contains(last, case=False, na=False)]
                if sub.empty or stat not in sub.columns:
                    continue
                val  = round(float(sub[stat].mean()), 1)
                lbl  = LABELS.get(stat, stat.upper())
                ticks.append((last, f"{lbl} {val}", "u"))
    except Exception:
        pass
    if not ticks:
        ticks = [
            ("SGA","PTS 31.2","h"),("Jokic","PRA 52.1","h"),
            ("Doncic","PTS 27.3","u"),("Giannis","REB 10.8","h"),
            ("Tatum","PTS 25.4","u"),("Curry","3PM 3.8","u"),
            ("LeBron","PTS 18.6","h"),("Wemby","BLK 3.2","h"),
            ("A.Davis","BLK 2.4","u"),("Durant","PTS 27.3","u"),
        ]
    return ticks

try:
    TICKS = _build_ticker(players_df)
except Exception:
    TICKS = [
        ("SGA","PTS 31.2","h"),("Jokic","PRA 52.1","h"),
        ("Doncic","PTS 27.3","u"),("Giannis","REB 10.8","h"),
        ("Tatum","PTS 25.4","u"),("Curry","3PM 3.8","u"),
        ("LeBron","PTS 18.6","h"),("Wemby","BLK 3.2","h"),
    ]
ti  = "".join(f'<span class="lb-tick">{n}&nbsp;<span class="{c}">{s}</span></span>' for n,s,c in TICKS)
td  = ti * 2

# ── Nav ───────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="lb-nav">
    <div class="lb-logo">LineBreaker</div>
    <div class="lb-ticker-wrap"><div class="lb-ticker-track">{td}</div></div>
    <div class="lb-nav-right">
        <div class="lb-badge">NBA &amp; NFL Props</div>
        <div class="lb-tagline">Beat the Line. Break the Line.</div>
    </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
nba_tab, picks_tab, accuracy_tab, nfl_tab = st.tabs(["🏀  NBA Props", "⚡  Quick Picks", "📈  Accuracy", "🏈  NFL Props"])

# ══════════════════════════════════════════════════════════════════════════════
with nba_tab:
    # ── Slate (full width, horizontal) ────────────────────────────────────────
    if today_slate:
        st.markdown('<span class="slate-hdr">Today\'s Games</span>', unsafe_allow_html=True)
        n    = min(len(today_slate), 8)
        cols = st.columns(n)
        cur  = st.session_state.get("slate_opp")
        for i, g in enumerate(today_slate[:n]):
            ha,aa = g["home_abbr"],g["away_abbr"]
            hc = TEAM_COLORS.get(ha,"#888"); ac = TEAM_COLORS.get(aa,"#888")
            hr = team_records.get(ha, g.get("home_record","")); ar = team_records.get(aa, g.get("away_record",""))
            gst = g["status"]
            if gst == "live":
                hp,ap = g.get("home_pts") or "",g.get("away_pts") or ""
                ts,tc = (f"LIVE {ap}-{hp}" if hp or ap else "LIVE"),"#e05a5a"
            elif gst == "final":
                hp,ap = g.get("home_pts") or "",g.get("away_pts") or ""
                ts,tc = f"Final {ap}-{hp}","#3a3a50"
            else:
                ts,tc = g.get("game_time","TBD"),"#3a3a50"
            sel = (cur == ha)
            try: r2,g2,b2=int(hc[1:3],16),int(hc[3:5],16),int(hc[5:7],16); bg=f"rgba({r2},{g2},{b2},0.08)" if sel else "#0d0d15"
            except: bg="#0d0d15"
            bdr = hc if sel else "#1a1a28"
            fl = "<link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@500;600&display=swap' rel='stylesheet'>"
            cs = (f"*{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;font-family:'Inter',sans-serif;}}"
                  f".c{{border:2px solid {bdr};border-radius:8px;padding:7px 8px;background:{bg};}}"
                  f".t{{font-size:9px;font-weight:600;color:{tc};letter-spacing:1px;text-transform:uppercase;margin-bottom:3px;}}"
                  f".r{{display:flex;align-items:center;justify-content:center;gap:5px;}}"
                  f".a{{font-family:'Bebas Neue',sans-serif;font-size:16px;line-height:1;}}"
                  f".s{{font-size:8px;color:#2a2a3a;text-align:center;margin-top:1px;}}"
                  f".at{{font-size:9px;color:#1e1e2e;font-weight:700;}}")
            bd = (f"<div class='c'><div class='t'>{ts}</div><div class='r'>"
                  f"<div><div class='a' style='color:{ac}'>{aa}</div><div class='s'>{ar}</div></div>"
                  f"<div class='at'>@</div>"
                  f"<div><div class='a' style='color:{hc}'>{ha}</div><div class='s'>{hr}</div></div>"
                  f"</div></div>")
            with cols[i]:
                components.html(f"<!DOCTYPE html><html><head>{fl}<style>{cs}</style></head><body>{bd}</body></html>",
                               height=60, scrolling=False)
                if __import__('streamlit').button(f"{aa}@{ha}",key=f"sg_{i}",use_container_width=True,help=f"Set opponent: {ha}"):
                    __import__('streamlit').session_state["slate_opp"] = ha
                    __import__('streamlit').rerun()
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Two columns ───────────────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)

        # ── Player search (always visible, no expander) ───────────────────────
        search_q = st.text_input("🔍  Search player", placeholder="Name...", label_visibility="visible")

        with st.expander("Filter", expanded=False):
            all_teams = sorted([t for t in players_df.get("team_abbreviation",pd.Series()).dropna().unique() if t not in ("UNK","FA","")])
            f_team = st.selectbox("Team", ["All teams"]+all_teams)
            all_pos = sorted([p for p in players_df.get("position",pd.Series()).dropna().unique() if p not in ("Unknown","")])
            f_pos  = st.selectbox("Position", ["All positions"]+all_pos)

        filt = players_df.copy()
        if search_q: filt = filt[filt["full_name"].str.contains(search_q,case=False,na=False)]
        if f_team != "All teams" and "team_abbreviation" in filt.columns: filt = filt[filt["team_abbreviation"]==f_team]
        if all_pos and f_pos != "All positions" and "position" in filt.columns: filt = filt[filt["position"]==f_pos]
        fopts = filt["full_name"].tolist() or player_options

        st.markdown('<span class="ctrl-label">Player</span>', unsafe_allow_html=True)
        sel_player = st.selectbox("Player", options=fopts, index=0, label_visibility="collapsed")

        st.markdown('<span class="ctrl-label">Opponent</span>', unsafe_allow_html=True)
        sl_opp   = st.session_state.get("slate_opp")
        opp_idx  = team_options.index(sl_opp) if sl_opp and sl_opp in team_options else 0
        sel_opp  = st.selectbox("Opponent", options=team_options, index=opp_idx, label_visibility="collapsed")

        lc, rc = st.columns(2)
        with lc:
            st.markdown('<span class="ctrl-label">Location</span>', unsafe_allow_html=True)
            location = st.radio("loc", ["Home","Away"], horizontal=True, label_visibility="collapsed")
        with rc:
            st.markdown('<span class="ctrl-label">Rest Days</span>', unsafe_allow_html=True)
            rest_days = st.slider("rest", 0, 7, 2, label_visibility="collapsed")

        st.markdown('<span class="ctrl-label">Stat</span>', unsafe_allow_html=True)
        sel_stat_d  = st.selectbox("Stat", options=stat_options, label_visibility="collapsed")
        sel_target  = stat_map.get(sel_stat_d, "pts")
        target_info = TARGET_DISPLAY.get(sel_target, {"label":sel_target,"short":sel_target,"group":"Core"})

        st.markdown('<span class="ctrl-label">Over/Under Line</span>', unsafe_allow_html=True)
        default_line = DEFAULT_THRESHOLDS.get(sel_target, 10)
        custom_line  = st.slider("line", 1, 60, value=int(default_line), label_visibility="collapsed")

        run_btn = st.button("▶  Run Prediction", use_container_width=True, type="primary")

        # ── System buttons ────────────────────────────────────────────────────
        st.divider()
        st.caption("SYSTEM")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔄 Sync Data", use_container_width=True,
                         help="Re-fetch current season from NBA API"):
                with st.spinner("Syncing..."):
                    ok = refresh_current_season()
                    st.cache_data.clear()
                    st.session_state.pop("season_refreshed", None)
                if ok: st.success("✅ Game data synced!")
                else:  st.error("❌ Sync failed")

            if st.button("🏥 Injuries", use_container_width=True,
                         help="Refresh ESPN injury report"):
                with st.spinner("Fetching..."):
                    try:
                        from data.fetch_injuries import fetch_injury_report
                        fetch_injury_report(force_refresh=True)
                        load_injuries.clear()
                        st.success("✅ Injuries updated!")
                    except Exception as e:
                        st.error(f"❌ {e}")

        with c2:
            if st.button("📋 Lineups", use_container_width=True,
                         help="Refresh starting lineups from ESPN"):
                with st.spinner("Fetching..."):
                    try:
                        from data.fetch_lineups import fetch_all_lineups
                        fetch_all_lineups(force_refresh=True)
                        load_lineups.clear()
                        st.success("✅ Lineups updated!")
                    except Exception as e:
                        st.error(f"❌ {e}")

            if st.button("🤖 Retrain", use_container_width=True,
                         help="Retrain all models (~15 min, runs in background)"):
                try:
                    import subprocess
                    retrain_path = str(ROOT / "retrain.py")
                    (ROOT / "logs").mkdir(exist_ok=True)
                    subprocess.Popen(
                        [sys.executable, retrain_path, "--force"],
                        stdout=open(ROOT / "logs" / "retrain.log", "a"),
                        stderr=subprocess.STDOUT, cwd=str(ROOT))
                    st.info("🤖 Retraining started in background")
                except Exception as e:
                    st.error(f"❌ {e}")

        if st.button("📊 Backtest (30d)", use_container_width=True,
                     help="Check prediction accuracy vs actual results"):
            with st.spinner("Running backtest..."):
                try:
                    from models.backtest import backtest_vs_actuals
                    df_bt = backtest_vs_actuals(target="pts", days_back=30, verbose=False)
                    if not df_bt.empty:
                        acc = df_bt["correct"].mean()*100
                        hi  = df_bt[df_bt["confidence"]=="High"]["correct"].mean()*100 if len(df_bt[df_bt["confidence"]=="High"]) else 0
                        st.success(f"Pts accuracy: {acc:.1f}% overall | {hi:.1f}% high-conf")
                    else:
                        st.warning("No backtest data.")
                except Exception as e:
                    st.error(f"Backtest failed: {e}")

        try:
            import json as _json
            _mp = ROOT / "models" / "saved" / "metadata.json"
            if _mp.exists():
                _m = _json.load(open(_mp))
                st.markdown(
                    f'<div style="font-size:0.52rem;color:#2a2a3a;margin-top:0.5rem;">'
                    f'Models: {_m.get("trained_at","?")[:10]} &nbsp;·&nbsp; {_m.get("n_models","?")} targets</div>',
                    unsafe_allow_html=True)
        except: pass

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Output ────────────────────────────────────────────────────────────────
    with right:
        if not run_btn:
            ph = f"""<!DOCTYPE html><html><head>
            <link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@600&display=swap' rel='stylesheet'>
            <style>*{{box-sizing:border-box;margin:0;}} body{{background:transparent;}}
            .w{{position:relative;overflow:hidden;border-radius:16px;background:#0a0a12;border:1px solid #13131f;
                 min-height:480px;display:flex;align-items:center;justify-content:center;}}
            .bg{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}}
            .i{{position:relative;z-index:1;text-align:center;padding:2rem;}}
            .lg{{font-family:'Bebas Neue',sans-serif;font-size:5rem;color:#f0672a;opacity:0.07;line-height:1;margin-bottom:1rem;}}
            .t{{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1a1a28;}}
            .s{{font-size:10px;color:#111118;letter-spacing:1px;margin-top:6px;}}
            </style></head><body>
            <div class='w'><div class='bg'>{_svg_court()}</div>
            <div class='i'><div class='lg'>LineBreaker</div>
            <div class='t'>Select a player &amp; run prediction</div>
            <div class='s'>Beat the Line. Break the Line.</div>
            </div></div></body></html>"""
            components.html(ph, height=480, scrolling=False)

        else:
            p_row = players_df[players_df["full_name"]==sel_player].iloc[0]
            o_row = teams_df[teams_df["team_abbreviation"]==sel_opp].iloc[0]

            with st.spinner(f"Predicting {sel_player}..."):
                try:
                    result = predict(
                        player_id=int(p_row["id"]),
                        opponent_team_id=int(o_row["team_id"]),
                        opponent_name=sel_opp,
                        is_home=(location=="Home"),
                        rest_days=rest_days,
                    )
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    st.stop()

            tr = result.targets.get(sel_target)
            if tr is None:
                st.warning(f"No model for {target_info['label']}.")
                st.stop()

            pt  = str(p_row.get("team_abbreviation","")).upper()
            pc  = TEAM_COLORS.get(pt,"#f0672a")
            oc  = TEAM_COLORS.get(sel_opp,"#888888")
            loc = "vs." if location=="Home" else "@"
            hs  = _headshot(int(p_row["id"]))
            inj     = _inj_badge(result.injury_info or {})
            lu_info = result.lineup_info or {}
            lu_badge = _lineup_badge(lu_info)

            # Over/under
            mae  = tr.model_mae if tr.model_mae>0 else 1.0
            diff = (tr.predicted_value - custom_line)/mae
            bw   = 0.25 if abs(tr.threshold-custom_line)<2 else 0.5
            ro   = 1/(1+math.exp(-1.2*diff))
            op   = round((bw*ro+(1-bw)*tr.over_prob)*100,1)
            up   = round(100-op,1)

            # Confidence
            cc = {"High":"#4caf82","Medium":"#d4b44a","Low":"#e05a5a"}
            cb = {"High":"rgba(76,175,130,0.12)","Medium":"rgba(212,180,74,0.12)","Low":"rgba(224,90,90,0.12)"}
            cbo= {"High":"rgba(76,175,130,0.3)","Medium":"rgba(212,180,74,0.3)","Low":"rgba(224,90,90,0.3)"}
            cco= cc.get(tr.confidence_label,"#f0672a")
            ca  = {"High":150,"Medium":90,"Low":30}.get(tr.confidence_label,90)
            cnx = 100+65*math.cos(math.radians(180-ca))
            cny = 100-65*math.sin(math.radians(180-ca))

            # Page tint
            try:
                r,gv,bv=int(pc[1:3],16),int(pc[3:5],16),int(pc[5:7],16)
                st.markdown(f"""<style>
                .lb-body {{ background: radial-gradient(ellipse 70% 50% at 80% 15%,
                    rgba({r},{gv},{bv},0.1) 0%,transparent 55%) !important; }}
                [data-testid="stSlider"] > div > div > div > div {{ background:{pc} !important; }}
                [data-testid="stButton"] button[kind="primary"] {{ background:{pc} !important; }}
                </style>""", unsafe_allow_html=True)
            except: pass

            # Chart
            rg = result.recent_games
            cv = []
            if sel_target in COMBO_TARGETS:
                pts = COMBO_TARGETS[sel_target]
                av  = [p for p in pts if p in rg.columns]
                if av: cv = rg[av].apply(pd.to_numeric,errors="coerce").sum(axis=1).tolist()
            elif sel_target in ("double_double","triple_double"):
                dc  = [c for c in ["pts","reb","ast","stl","blk"] if c in rg.columns]
                ten = (rg[dc].apply(pd.to_numeric,errors="coerce")>=10).sum(axis=1)
                cv  = (ten>=(2 if sel_target=="double_double" else 3)).astype(int).tolist()
            elif sel_target in rg.columns:
                cv = rg[sel_target].apply(pd.to_numeric,errors="coerce").tolist()

            gd = []
            if "game_date" in rg.columns and cv:
                try: gd = pd.to_datetime(rg["game_date"]).dt.strftime("%-m/%-d").tolist()
                except: gd = pd.to_datetime(rg["game_date"]).dt.strftime("%m/%d").tolist()

            # ── Chart: area + line ──────────────────────────────────────────────
            W,H,PL,PR,PT,PB = 600,160,10,55,20,32
            chart_block = ""
            if cv:
                vld  = [v for v in cv if v==v]
                mx   = max(vld)*1.15 if vld else 1
                mn   = max(0,min(vld)*0.7) if vld else 0
                rng  = mx-mn if mx!=mn else 1
                av_v = sum(vld)/len(vld) if vld else 0
                n    = len(cv)
                def yx(v): return PT+(1-(v-mn)/rng)*(H-PT-PB)
                def xx(i): return PL+i*(W-PL-PR)/max(n-1,1)
                pts = " ".join(f"{xx(i):.1f},{yx(v):.1f}" for i,v in enumerate(cv) if v==v)
                fx,lx,by2 = xx(0),xx(n-1),H-PB
                ap = f"M {fx:.1f},{by2} "+" ".join(f"L {xx(i):.1f},{yx(v):.1f}" for i,v in enumerate(cv) if v==v)+f" L {lx:.1f},{by2} Z"
                dots=""
                for i,v in enumerate(cv):
                    if v!=v: continue
                    cx2,cy2=xx(i),yx(v); ab=v>=custom_line
                    dc=pc if ab else "#1e2535"
                    dots+=f'<circle cx="{cx2:.1f}" cy="{cy2:.1f}" r="4" fill="{dc}" stroke="#080810" stroke-width="1.5"/>'
                    ly3=cy2-10 if cy2>PT+15 else cy2+17
                    fc="#c8c6c0" if ab else "#2a3040"
                    dots+=f'<text x="{cx2:.1f}" y="{ly3:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="{fc}" font-family="Inter,sans-serif">{int(v)}</text>'
                    if i<len(gd): dots+=f'<text x="{cx2:.1f}" y="{H-PB+14}" text-anchor="middle" font-size="9" fill="#252535" font-family="Inter,sans-serif">{gd[i]}</text>'
                ay2=yx(av_v); ly4=yx(custom_line)
                rl=f'<line x1="{PL}" y1="{ay2:.1f}" x2="{W-PR}" y2="{ay2:.1f}" stroke="#1e1e2e" stroke-dasharray="3,4" stroke-width="1"/>'
                rl+=f'<text x="{W-PR+4}" y="{ay2+4:.1f}" font-size="9" fill="#252535" font-family="Inter,sans-serif">avg {av_v:.1f}</text>'
                if PT<=ly4<=H-PB:
                    rl+=f'<line x1="{PL}" y1="{ly4:.1f}" x2="{W-PR}" y2="{ly4:.1f}" stroke="{pc}" stroke-dasharray="4,3" stroke-width="1.5" opacity="0.6"/>'
                    rl+=f'<text x="{W-PR+4}" y="{ly4+4:.1f}" font-size="9" fill="{pc}" font-family="Inter,sans-serif" opacity="0.9">{custom_line}</text>'
                chart_svg=f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;display:block;"><defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{pc}" stop-opacity="0.2"/><stop offset="100%" stop-color="{pc}" stop-opacity="0"/></linearGradient></defs>{rl}<path d="{ap}" fill="url(#ag)"/><polyline points="{pts}" fill="none" stroke="{pc}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>{dots}</svg>'
                chart_block=f'<div><div style="font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1e1e28;margin-bottom:10px;">Last {n} games &mdash; {target_info["label"]}</div>{chart_svg}</div>'


            html = f"""<!DOCTYPE html><html><head>
            <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <style>
            *{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;font-family:'Inter',sans-serif;color:#e8e6e0;}}
            .card{{background:#0a0a12;border-radius:16px;overflow:hidden;}}
            .top{{height:3px;background:linear-gradient(90deg,{pc} 0%,transparent 100%);}}
            .hdr{{display:flex;align-items:center;justify-content:space-between;padding:16px 20px 14px;border-bottom:1px solid #0d0d15;gap:12px;}}
            .hl{{display:flex;align-items:center;gap:12px;flex:1;min-width:0;}}
            .av{{width:52px;height:52px;border-radius:50%;border:2px solid {pc};object-fit:cover;object-position:top;background:#0d0d15;flex-shrink:0;}}
            .nm{{font-size:18px;font-weight:700;letter-spacing:-0.02em;color:#f0ede8;line-height:1.1;}}
            .sb{{font-size:11px;color:#252535;margin-top:3px;}} .opp{{color:{oc};font-weight:600;}}
            .cf{{display:inline-flex;align-items:center;gap:4px;border-radius:20px;padding:3px 10px;
                  font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;flex-shrink:0;
                  background:{cb.get(tr.confidence_label,"rgba(240,103,42,0.12)")};
                  color:{cco};border:1px solid {cbo.get(tr.confidence_label,"rgba(240,103,42,0.3)")};}}
            .cd{{width:5px;height:5px;border-radius:50%;background:{cco};}}
            .body{{padding:18px 20px 16px;}}
            /* Hero number row */
            .hero-row{{display:flex;align-items:flex-end;gap:20px;margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid #0d0d15;}}
            .proj{{flex:0 0 auto;}}
            .pnum{{font-family:'Bebas Neue',sans-serif;font-size:min(96px,18vw);line-height:0.85;color:{pc};
                    text-shadow:0 0 60px {pc}44;letter-spacing:0.01em;}}
            .plbl{{font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#1e1e28;margin-top:6px;}}
            /* Stats grid */
            .sgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;flex:1;}}
            .sc{{background:#0d0d15;border-radius:10px;padding:10px 12px;}}
            .sv{{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#e8e6e0;line-height:1;margin-bottom:2px;}}
            .sl{{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#1a1a28;}}
            /* Confidence meter */
            .meter-col{{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;gap:4px;padding-bottom:4px;}}
            .meter-lbl{{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#1a1a28;}}
            .meter-val{{font-family:'Bebas Neue',sans-serif;font-size:1rem;color:{cco};}}
            /* Over/under */
            .ou{{margin-bottom:16px;}}
            .ot{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1a1a28;margin-bottom:10px;}}
            .or{{display:flex;align-items:center;gap:10px;margin-bottom:7px;}}
            .or:last-child{{margin-bottom:0;}}
            .ol{{font-size:11px;font-weight:500;color:#2a2a3a;min-width:72px;}}
            .ob{{flex:1;height:5px;background:#0d0d15;border-radius:3px;overflow:hidden;position:relative;}}
            .fo{{height:100%;background:{pc};border-radius:3px;width:{op:.1f}%;}}
            .fu{{height:100%;background:#2a5a9f;border-radius:3px;width:{up:.1f}%;}}
            .op2{{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;min-width:48px;text-align:right;line-height:1;}}
            /* Chart */
            .chart-wrap{{padding-top:14px;border-top:1px solid #0d0d15;}}
            .ct{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1a1a28;margin-bottom:10px;}}
            .meta{{font-size:9px;color:#111118;margin-top:8px;padding-top:10px;border-top:1px solid #0a0a12;}}
            </style></head><body>
            <div class="card">
                <div class="top"></div>
                <div class="hdr">
                    <div class="hl">
                        <img class="av" src="{hs}" onerror="this.style.display='none'"/>
                        <div>
                            <div class="nm">{result.player_name}</div>
                            <div class="sb">{loc}&nbsp;<span class="opp">{sel_opp}</span>&nbsp;&middot;&nbsp;{rest_days}d rest&nbsp;&middot;&nbsp;{location}{inj}{lu_badge}</div>
                        </div>
                    </div>
                    <span class="cf"><span class="cd"></span>{tr.confidence_label}</span>
                </div>
                <div class="body">
                    <div class="hero-row">
                        <div class="proj">
                            <div class="pnum">{tr.predicted_value}</div>
                            <div class="plbl">Projected {target_info["label"]}</div>
                        </div>
                        <div class="meter-col">
                            <svg width="90" height="52" viewBox="0 0 200 110" xmlns="http://www.w3.org/2000/svg">
                                <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#111118" stroke-width="18" stroke-linecap="round"/>
                                <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#cg)" stroke-width="18" stroke-linecap="round"
                                      stroke-dasharray="251" stroke-dashoffset="{int(251*(1-op/100))}"/>
                                <defs><linearGradient id="cg" x1="0" y1="0" x2="1" y2="0">
                                    <stop offset="0%" stop-color="#e05a5a"/>
                                    <stop offset="50%" stop-color="#d4b44a"/>
                                    <stop offset="100%" stop-color="#4caf82"/>
                                </linearGradient></defs>
                                <line x1="100" y1="100" x2="{cnx:.1f}" y2="{cny:.1f}" stroke="#e8e6e0" stroke-width="2.5" stroke-linecap="round"/>
                                <circle cx="100" cy="100" r="5" fill="#e8e6e0"/>
                            </svg>
                            <div class="meter-val">{op:.0f}%</div>
                            <div class="meter-lbl">{tr.confidence_label}</div>
                        </div>
                        <div class="sgrid">
                            <div class="sc"><div class="sv">{tr.recent_avg_5}</div><div class="sl">L5 Avg</div></div>
                            <div class="sc"><div class="sv">{tr.recent_avg_10}</div><div class="sl">L10 Avg</div></div>
                            <div class="sc"><div class="sv">{custom_line}</div><div class="sl">Line</div></div>
                        </div>
                    </div>
                    <div class="ou">
                        <div class="ot">Over / Under {custom_line} {target_info["short"]}</div>
                        <div class="or">
                            <div class="ol">Over {custom_line}</div>
                            <div class="ob"><div class="fo"></div></div>
                            <div class="op2" style="color:{pc};">{op:.1f}%</div>
                        </div>
                        <div class="or">
                            <div class="ol">Under {custom_line}</div>
                            <div class="ob"><div class="fu"></div></div>
                            <div class="op2" style="color:#4a9eff;">{up:.1f}%</div>
                        </div>
                    </div>
                    {"" if not chart_block else f'<div class="chart-wrap"><div class="ct">Last {n if cv else ""} games &mdash; {target_info["label"]}</div>{chart_svg if cv else ""}</div>'}
                    <div class="meta">Model AUC {tr.model_auc}&nbsp;&middot;&nbsp;Trained on 5 seasons</div>
                </div>
            </div></body></html>"""

            components.html(html, height=630, scrolling=False)

            # ── Log this prediction ───────────────────────────────────────────────
            log_col, share_col = st.columns(2)
            with log_col:
                if st.button("📝 Log This Pick", use_container_width=True,
                             help="Save prediction to track accuracy"):
                    log_prediction(
                        player_name=result.player_name,
                        player_id=int(p_row["id"]),
                        target=sel_target,
                        predicted=float(tr.predicted_value),
                        custom_line=float(custom_line),
                        over_prob=float(op),
                        opponent=sel_opp,
                        is_home=(location=="Home"),
                    )
                    st.success("✅ Pick logged! Check Accuracy tab to track results.")
            with share_col:
                share_text = (
                    f"🏀 LineBreaker Prediction\n"
                    f"{result.player_name} — {target_info['label']}\n"
                    f"Projected: {tr.predicted_value} | Line: {custom_line}\n"
                    f"{'OVER' if op>50 else 'UNDER'} {custom_line} ({op:.1f}%)\n"
                    f"breaktheline.streamlit.app"
                )
                st.code(share_text, language=None)

            with st.expander("📊  All props — this matchup", expanded=False):
                    rows=[]
                    for t,ta in result.targets.items():
                        inf=TARGET_DISPLAY.get(t,{"label":t,"short":t,"group":"-"})
                        rows.append({"Stat":inf["label"],"Group":inf["group"],"Projected":ta.predicted_value,
                                     "Line":ta.threshold,"Over %":f"{ta.over_prob*100:.1f}%",
                                     "Confidence":ta.confidence_label,"MAE":ta.model_mae})
                    if rows:
                        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
                            column_config={"Projected":st.column_config.NumberColumn("Projected",format="%.1f"),
                                           "MAE":st.column_config.NumberColumn("MAE",format="%.3f")})

# ══════════════════════════════════════════════════════════════════════════════
# QUICK PICKS TAB
# ══════════════════════════════════════════════════════════════════════════════
with picks_tab:
    st.markdown("""
    <div style="margin-bottom:1.2rem;">
        <div style="font-size:1.4rem;font-weight:700;color:#f0ede8;margin-bottom:0.3rem;">⚡ Today's Best Picks</div>
        <div style="font-size:0.72rem;color:#2a2a3a;">Highest-edge plays ranked by model confidence. Scan the full slate in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        qp_conf = st.selectbox("Confidence", ["All","High","Medium"], key="qp_conf")
    with col_f2:
        qp_dir = st.selectbox("Direction", ["All","OVER","UNDER"], key="qp_dir")
    with col_f3:
        qp_n = st.slider("Top N", 5, 25, 10, key="qp_n")

    gen_col, ref_col = st.columns([3, 1])
    with gen_col:
        gen_btn = st.button("⚡ Generate Quick Picks", use_container_width=True, type="primary", key="qp_gen")
    with ref_col:
        clear_btn = st.button("🗑 Clear", use_container_width=True, key="qp_clear")

    if gen_btn:
        import time as _time
        st.session_state["qp_seed"] = int(_time.time())
        st.session_state.pop("qp_result", None)
        st.rerun()  # clean rerun so computation starts fresh, button state cleared

    if clear_btn:
        st.session_state.pop("qp_result", None)
        st.session_state.pop("qp_seed", None)
        st.rerun()

    seed = st.session_state.get("qp_seed", 0)
    if seed > 0 and "qp_result" not in st.session_state:
        with st.spinner("Scanning today's slate… this takes about 30–60 seconds"):
            df = _compute_picks_cached(qp_conf, qp_n, seed)
        # store OUTSIDE spinner so session state persists even if UI redraws
        st.session_state["qp_result"] = df
        st.session_state["qp_dir_filter"] = qp_dir
        st.rerun()  # force clean display rerun — prevents loop

    # Display from session state (survives tab switches)
    qp_df = st.session_state.get("qp_result")
    qp_dir_filter = st.session_state.get("qp_dir_filter", "All")

    if qp_df is None:
        # Empty state — show a clean prompt
        components.html(f"""<!DOCTYPE html><html><head>
        <link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@500;600&display=swap' rel='stylesheet'>
        <style>*{{box-sizing:border-box;margin:0;}} body{{background:transparent;}}
        .w{{border-radius:16px;background:#0a0a12;border:1px solid #13131f;min-height:340px;
             display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden;}}
        .i{{text-align:center;padding:2rem;position:relative;z-index:1;}}
        .ico{{font-size:3rem;line-height:1;margin-bottom:1rem;opacity:0.3;}}
        .t{{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1a1a28;}}
        .s{{font-size:10px;color:#111118;letter-spacing:1px;margin-top:8px;line-height:1.6;}}
        .hl{{color:#f0672a;}}
        </style></head><body>
        <div class='w'><div class='bg'>{_svg_court()}</div>
        <div class='i'><div class='ico'>⚡</div>
        <div class='t'>No picks generated yet</div>
        <div class='s'>Hit <span class='hl'>Generate Quick Picks</span> to scan today's slate.<br>
        Scans all players &amp; targets — takes ~30–60 seconds.</div>
        </div></div></body></html>""", height=340, scrolling=False)

    elif qp_df.empty:
        st.warning("⚠️ No picks found. Either no NBA games are scheduled today, or player data needs to be synced (use Sync Data in the NBA tab).")
    else:
        # Apply direction filter
        display_df = qp_df.copy()
        if qp_dir_filter != "All":
            display_df = display_df[display_df["direction"] == qp_dir_filter]
        if display_df.empty:
            st.info(f"No {qp_dir_filter} picks in today's slate. Try removing the direction filter.")
        else:
            # ── Top 3 Hero Picks ──────────────────────────────────────────
            top3 = display_df.head(3)
            hero_cols = st.columns(len(top3))
            for idx, (_, row) in enumerate(top3.iterrows()):
                pc_h    = TEAM_COLORS.get(row["team"], "#f0672a")
                conf_c  = {"High":"#4caf82","Medium":"#d4b44a","Low":"#e05a5a"}.get(row["confidence"],"#f0672a")
                arrow_h = "⬆" if row["direction"] == "OVER" else "⬇"
                try:
                    r2,g2,b2 = int(pc_h[1:3],16),int(pc_h[3:5],16),int(pc_h[5:7],16)
                    bg_h = f"rgba({r2},{g2},{b2},0.08)"
                except: bg_h = "#0a0a12"
                # get player ID for headshot
                p_match = players_df[players_df["full_name"] == row["player"]]
                hs_h = _headshot(int(p_match.iloc[0]["id"])) if not p_match.empty else ""
                with hero_cols[idx]:
                    components.html(f"""<!DOCTYPE html><html><head>
                    <link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@500;600;700&display=swap' rel='stylesheet'>
                    <style>*{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;font-family:'Inter',sans-serif;}}
                    .c{{background:{bg_h};border:1px solid {pc_h}33;border-radius:12px;border-top:2px solid {pc_h};
                         padding:14px;overflow:hidden;position:relative;}}
                    .rank{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:{pc_h};margin-bottom:8px;}}
                    .hs{{width:42px;height:42px;border-radius:50%;border:2px solid {pc_h};object-fit:cover;object-position:top;
                          background:#0d0d15;float:right;margin-left:8px;}}
                    .nm{{font-size:13px;font-weight:700;color:#f0ede8;line-height:1.2;}}
                    .tm{{font-size:9px;color:#2a2a3a;margin-top:2px;}}
                    .prop{{font-family:'Bebas Neue',sans-serif;font-size:28px;color:{pc_h};line-height:1;margin:10px 0 4px;}}
                    .det{{font-size:9px;color:#252535;}}
                    .prob{{font-size:11px;font-weight:700;color:{conf_c};margin-top:6px;}}
                    </style></head><body>
                    <div class='c'>
                        <div class='rank'>#{idx+1} Best Edge</div>
                        {"<img class='hs' src='" + hs_h + "' onerror=\"this.style.display='none'\"/>" if hs_h else ""}
                        <div class='nm'>{row["player"].split()[-1]}</div>
                        <div class='tm'>{row["team"]} &middot; {row["opponent"]}</div>
                        <div class='prop'>{arrow_h} {row["short"]}</div>
                        <div class='det'>Proj {row["predicted"]} &middot; Line {row["line"]}</div>
                        <div class='prob'>{row["direction"]} {row["over_prob"]}%</div>
                    </div></body></html>""", height=160, scrolling=False)

            st.markdown(f'<div style="font-size:0.58rem;font-weight:700;letter-spacing:0.18em;text-transform:uppercase;color:#1e1e2e;margin:1rem 0 0.6rem;">All {len(display_df)} picks — ranked by edge</div>', unsafe_allow_html=True)

            # ── Full ranked list ──────────────────────────────────────────
            try:
                for rank, (_, row) in enumerate(display_df.iterrows(), 1):
                    arrow    = "⬆" if row["direction"] == "OVER" else "⬇"
                    pc       = TEAM_COLORS.get(row["team"], "#f0672a")
                    conf_c   = {"High":"#4caf82","Medium":"#d4b44a","Low":"#e05a5a"}.get(row["confidence"],"#f0672a")
                    edge_w   = min(100, int(row["edge"] * 12))
                    trend    = row.get("trend", "→")
                    l5       = row.get("l5_avg", 0)
                    l10      = row.get("l10_avg", 0)
                    # get headshot
                    p_match  = players_df[players_df["full_name"] == row["player"]]
                    hs_url   = _headshot(int(p_match.iloc[0]["id"])) if not p_match.empty else ""
                    try:
                        r2,g2,b2 = int(pc[1:3],16),int(pc[3:5],16),int(pc[5:7],16)
                        pc_bg = f"rgba({r2},{g2},{b2},0.05)"
                    except: pc_bg = "#0a0a12"

                    hs_img = (f'<img src="{hs_url}" style="width:38px;height:38px;border-radius:50%;'
                              f'border:1.5px solid {pc};object-fit:cover;object-position:top;'
                              f'background:#0d0d15;flex-shrink:0;" onerror="this.style.display=\'none\'"/>') if hs_url else ""

                    st.markdown(f"""
                    <div style="background:{pc_bg};border:1px solid #0d0d15;border-radius:10px;
                                padding:0.75rem 1rem;margin-bottom:0.4rem;border-left:2px solid {pc};
                                display:flex;align-items:center;gap:0.9rem;">
                        <div style="font-size:0.6rem;font-weight:700;color:#1e1e2e;min-width:16px;text-align:center;">{rank}</div>
                        {hs_img}
                        <div style="flex:1;min-width:0;">
                            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">
                                <span style="font-size:0.88rem;font-weight:700;color:#f0ede8;">{row["player"]}</span>
                                <span style="background:{conf_c}18;color:{conf_c};border:1px solid {conf_c}33;
                                             border-radius:20px;padding:1px 7px;font-size:8px;font-weight:700;
                                             letter-spacing:1px;flex-shrink:0;">{row["confidence"].upper()}</span>
                            </div>
                            <div style="display:flex;align-items:center;gap:0.8rem;">
                                <span style="font-family:'Bebas Neue',sans-serif;font-size:1.6rem;color:{pc};line-height:1;">{arrow} {row["short"]}</span>
                                <div style="flex:1;">
                                    <div style="font-size:0.68rem;color:#2a2a3a;">
                                        {row["team"]}&nbsp;vs&nbsp;{row["opponent"]}&nbsp;&middot;&nbsp;
                                        Proj <strong style="color:#e8e6e0;">{row["predicted"]}</strong>&nbsp;&middot;&nbsp;
                                        Line <strong style="color:#e8e6e0;">{row["line"]}</strong>&nbsp;&middot;&nbsp;
                                        <strong style="color:{pc};">{row["direction"]} {row["over_prob"]}%</strong>
                                        &nbsp;&middot;&nbsp;<span style="color:#2a2a3a;">{trend}</span>
                                    </div>
                                    <div style="display:flex;align-items:center;gap:6px;margin-top:4px;">
                                        <span style="font-size:8px;color:#1a1a28;text-transform:uppercase;letter-spacing:1px;">Edge</span>
                                        <div style="flex:1;height:2px;background:#1a1a28;border-radius:2px;overflow:hidden;">
                                            <div style="height:100%;width:{edge_w}%;background:{pc};border-radius:2px;"></div>
                                        </div>
                                        <span style="font-size:9px;font-weight:700;color:{pc};">{row["edge"]:.1f}x</span>
                                        <span style="font-size:8px;color:#1a1a28;">L5 {l5} &middot; L10 {l10}</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                import traceback
                st.error(f"Display error: {e}")
                with st.expander("Details"):
                    st.code(traceback.format_exc())

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY TAB
# ══════════════════════════════════════════════════════════════════════════════
with accuracy_tab:
    # Resolve any pending predictions
    try:
        resolved = resolve_predictions()
        if resolved > 0:
            st.success(f"✅ Resolved {resolved} predictions with actual results!")
    except Exception:
        pass

    stats = get_accuracy_stats()

    if stats["total"] == 0:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#2a2a3a;">
            <div style="font-size:3rem;margin-bottom:1rem;">📈</div>
            <div style="font-size:1rem;font-weight:600;color:#3a3a4a;">No predictions logged yet</div>
            <div style="font-size:0.72rem;margin-top:0.5rem;">
                Run a prediction and click "Log This Pick" to start tracking accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Overall stats
        acc     = stats["accuracy"]
        acc_c   = "#4caf82" if acc >= 55 else "#d4b44a" if acc >= 50 else "#e05a5a"
        r_acc   = stats["recent_30d"]["accuracy"]
        r_acc_c = "#4caf82" if r_acc >= 55 else "#d4b44a" if r_acc >= 50 else "#e05a5a"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.8rem;margin-bottom:1.5rem;">
            <div style="background:#0a0a12;border:1px solid #13131f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:{acc_c};line-height:1;">{acc:.1f}%</div>
                <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#2a2a3a;margin-top:4px;">Overall</div>
            </div>
            <div style="background:#0a0a12;border:1px solid #13131f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:{r_acc_c};line-height:1;">{r_acc:.1f}%</div>
                <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#2a2a3a;margin-top:4px;">Last 30 Days</div>
            </div>
            <div style="background:#0a0a12;border:1px solid #13131f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:#f0672a;line-height:1;">{stats["total"]}</div>
                <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#2a2a3a;margin-top:4px;">Total Picks</div>
            </div>
            <div style="background:#0a0a12;border:1px solid #13131f;border-radius:12px;padding:1rem;text-align:center;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;color:#4caf82;line-height:1;">{stats["streak"]}</div>
                <div style="font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#2a2a3a;margin-top:4px;">Win Streak</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # By stat accuracy
        if stats["by_target"]:
            st.markdown('<div style="font-size:0.6rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#2a2a3a;margin-bottom:0.6rem;">Accuracy by Stat</div>', unsafe_allow_html=True)
            for tgt, tdata in sorted(stats["by_target"].items(), key=lambda x: -x[1]["accuracy"]):
                tinfo = TARGET_DISPLAY.get(tgt, {"short": tgt})
                tacc  = tdata["accuracy"]
                tc    = "#4caf82" if tacc >= 55 else "#d4b44a" if tacc >= 50 else "#e05a5a"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.4rem;">
                    <div style="font-size:0.72rem;color:#3a3a4a;min-width:60px;">{tinfo.get("short","?")}</div>
                    <div style="flex:1;height:4px;background:#1a1a28;border-radius:2px;overflow:hidden;">
                        <div style="height:100%;width:{tacc}%;background:{tc};border-radius:2px;"></div>
                    </div>
                    <div style="font-size:0.72rem;font-weight:600;color:{tc};min-width:40px;">{tacc:.0f}%</div>
                    <div style="font-size:0.62rem;color:#2a2a3a;">{tdata["total"]} picks</div>
                </div>
                """, unsafe_allow_html=True)

        # Recent picks log
        st.markdown('<div style="margin-top:1.5rem;font-size:0.6rem;font-weight:700;letter-spacing:0.16em;text-transform:uppercase;color:#2a2a3a;margin-bottom:0.6rem;">Recent Picks</div>', unsafe_allow_html=True)
        recent = get_recent_predictions(15)
        for p in recent:
            if p.get("resolved") and p.get("correct") is not None:
                icon   = "✅" if p["correct"] else "❌"
                actual = f"Actual: {p['actual']}" if p["actual"] is not None else "No result"
            else:
                icon   = "⏳"
                actual = "Pending"
            tinfo = TARGET_DISPLAY.get(p["target"], {"short": p["target"]})
            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;
                        padding:0.5rem 0;border-bottom:1px solid #0d0d15;font-size:0.72rem;">
                <div>{icon} <strong style="color:#e8e6e0;">{p["player_name"]}</strong>
                    <span style="color:#2a2a3a;margin-left:4px;">{tinfo.get("short","?")} {p["pick"]} {p["custom_line"]}</span>
                </div>
                <div style="color:#2a2a3a;">{actual} · {p.get("game_date","")}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# NFL TAB
# ══════════════════════════════════════════════════════════════════════════════
with nfl_tab:
    if not NFL_ENABLED:
        st.error("NFL module not found. Make sure `data/fetch_nfl.py` exists.")
        st.stop()

    # ── Load NFL data (cached) ─────────────────────────────────────────────
    nfl_teams_df   = load_nfl_teams()
    nfl_players_df = load_nfl_players()
    nfl_slate      = load_nfl_slate()

    # ── Today's NFL Slate ──────────────────────────────────────────────────
    if nfl_slate:
        st.markdown('<span class="slate-hdr">Today\'s NFL Games</span>', unsafe_allow_html=True)
        ns = min(len(nfl_slate), 8)
        ncols = st.columns(ns)
        for i, g in enumerate(nfl_slate[:ns]):
            ha, aa = g["home_abbr"], g["away_abbr"]
            hc = NFL_TEAM_COLORS.get(ha, "#888")
            ac = NFL_TEAM_COLORS.get(aa, "#888")
            gst = g["status"]
            if gst == "live":
                hp, ap = g.get("home_pts",""), g.get("away_pts","")
                ts, tc = (f"LIVE {ap}-{hp}" if (hp or ap) else "LIVE"), "#e05a5a"
            elif gst == "final":
                hp, ap = g.get("home_pts",""), g.get("away_pts","")
                ts, tc = f"Final {ap}-{hp}", "#3a3a50"
            else:
                ts, tc = g.get("game_time","TBD")[:5], "#3a3a50"
            fl = "<link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@500;600&display=swap' rel='stylesheet'>"
            cs = (f"*{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;font-family:'Inter',sans-serif;}}"
                  f".c{{border:2px solid #1a2a1a;border-radius:8px;padding:7px 8px;background:#060e06;}}"
                  f".t{{font-size:9px;font-weight:600;color:{tc};letter-spacing:1px;text-transform:uppercase;margin-bottom:3px;}}"
                  f".r{{display:flex;align-items:center;justify-content:center;gap:5px;}}"
                  f".a{{font-family:'Bebas Neue',sans-serif;font-size:16px;line-height:1;}}"
                  f".s{{font-size:8px;color:#2a5a3a;text-align:center;margin-top:1px;}}"
                  f".at{{font-size:9px;color:#1a3a1a;font-weight:700;}}")
            bd = (f"<div class='c'><div class='t'>{ts}</div><div class='r'>"
                  f"<div><div class='a' style='color:{ac}'>{aa}</div></div>"
                  f"<div class='at'>@</div>"
                  f"<div><div class='a' style='color:{hc}'>{ha}</div></div>"
                  f"</div></div>")
            with ncols[i]:
                components.html(f"<!DOCTYPE html><html><head>{fl}<style>{cs}</style></head><body>{bd}</body></html>",
                               height=55, scrolling=False)
        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#060e06;border:1px solid #143214;border-radius:10px;
                    padding:0.9rem 1.2rem;margin-bottom:1.2rem;display:flex;align-items:center;gap:0.8rem;">
            <div style="font-size:1.3rem;">🏈</div>
            <div>
                <div style="font-size:0.75rem;font-weight:700;color:#2a5a3a;letter-spacing:0.1em;text-transform:uppercase;">NFL Offseason</div>
                <div style="font-size:0.65rem;color:#1a3a1a;margin-top:2px;">No games today — search any player to view 2025 season projections</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Layout ─────────────────────────────────────────────────────────────
    nfl_left, nfl_right = st.columns([1, 2], gap="large")

    with nfl_left:
        st.markdown('<div class="ctrl-panel">', unsafe_allow_html=True)

        nfl_search = st.text_input("🔍  Search player", placeholder="Name…", key="nfl_search",
                                   label_visibility="visible")

        if not nfl_players_df.empty:
            with st.expander("Filter", expanded=False):
                nfl_pos_opts = ["All"] + sorted(nfl_players_df["position"].dropna().unique().tolist())
                nfl_team_opts = ["All"] + sorted(nfl_players_df["team_abbreviation"].dropna().unique().tolist())
                nfl_f_pos  = st.selectbox("Position", nfl_pos_opts, key="nfl_fpos")
                nfl_f_team = st.selectbox("Team",     nfl_team_opts, key="nfl_fteam")

            nfl_filt = nfl_players_df.copy()
            if nfl_search:
                nfl_filt = nfl_filt[nfl_filt["full_name"].str.contains(nfl_search, case=False, na=False)]
            if nfl_f_pos  != "All": nfl_filt = nfl_filt[nfl_filt["position"] == nfl_f_pos]
            if nfl_f_team != "All": nfl_filt = nfl_filt[nfl_filt["team_abbreviation"] == nfl_f_team]
            nfl_opts = nfl_filt["full_name"].tolist() or nfl_players_df["full_name"].tolist()
        else:
            nfl_opts = []

        nfl_run_btn = False  # default so right column can always reference it
        if not nfl_opts:
            st.warning("No players loaded. Click **Load Rosters** below.")
        else:
            st.markdown('<span class="ctrl-label">Player</span>', unsafe_allow_html=True)
            nfl_sel_player = st.selectbox("NFL Player", options=nfl_opts, key="nfl_player",
                                          label_visibility="collapsed")

            # Determine position for stat filtering
            p_match = nfl_players_df[nfl_players_df["full_name"] == nfl_sel_player]
            nfl_pos = p_match.iloc[0]["position"] if not p_match.empty else "QB"

            # Filter targets by position
            pos_targets = {k: v for k, v in NFL_TARGET_DISPLAY.items() if nfl_pos in v.get("pos", [])}
            if not pos_targets:
                pos_targets = NFL_TARGET_DISPLAY

            nfl_stat_opts = [f"{v['label']} ({v['short']})" for k, v in pos_targets.items()]
            nfl_stat_map  = {f"{v['label']} ({v['short']})": k for k, v in pos_targets.items()}

            st.markdown('<span class="ctrl-label">Location</span>', unsafe_allow_html=True)
            nfl_location = st.radio("nfl_loc", ["Home", "Away"], horizontal=True, key="nfl_loc",
                                    label_visibility="collapsed")

            st.markdown('<span class="ctrl-label">Stat</span>', unsafe_allow_html=True)
            nfl_sel_stat_d = st.selectbox("NFL Stat", options=nfl_stat_opts, key="nfl_stat",
                                          label_visibility="collapsed")
            nfl_target = nfl_stat_map.get(nfl_sel_stat_d, list(pos_targets.keys())[0])
            nfl_tinfo  = NFL_TARGET_DISPLAY.get(nfl_target, {"label": nfl_target, "short": nfl_target})

            st.markdown('<span class="ctrl-label">Over/Under Line</span>', unsafe_allow_html=True)
            nfl_default_line = NFL_THRESHOLDS.get(nfl_target, 50)
            nfl_line = st.slider("nfl_line", 0, 500, value=int(nfl_default_line), key="nfl_line",
                                 label_visibility="collapsed")

            nfl_run_btn = st.button("▶  Run NFL Prediction", use_container_width=True,
                                    type="primary", key="nfl_run")

        # ── Data management ────────────────────────────────────────────────
        st.divider()
        st.caption("NFL DATA")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📋 Load Rosters", use_container_width=True, key="nfl_load",
                         help="Fetch all 32 NFL team rosters from ESPN (~2 min)"):
                with st.spinner("Fetching NFL rosters…"):
                    try:
                        fetch_all_nfl_players(force_refresh=True)
                        load_nfl_players.clear()
                        st.success("✅ Rosters loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
        with c2:
            if st.button("🔄 Refresh", use_container_width=True, key="nfl_refresh",
                         help="Clear NFL cache and reload"):
                load_nfl_players.clear()
                load_nfl_teams.clear()
                load_nfl_slate.clear()
                st.success("✅ Cache cleared!")
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ── NFL Prediction Output ──────────────────────────────────────────────
    with nfl_right:
        if nfl_opts and nfl_run_btn:
            p_row_nfl = nfl_players_df[nfl_players_df["full_name"] == nfl_sel_player]
            if p_row_nfl.empty:
                st.error("Player not found.")
            else:
                p_row_nfl = p_row_nfl.iloc[0]
                pid_nfl   = str(p_row_nfl["player_id"])
                pt_nfl    = str(p_row_nfl.get("team_abbreviation", "")).upper()
                pc_nfl    = NFL_TEAM_COLORS.get(pt_nfl, "#4caf82")
                hs_nfl    = str(p_row_nfl.get("headshot_url", ""))

                with st.spinner(f"Fetching {nfl_sel_player} game logs…"):
                    nfl_pred = predict_nfl_prop(
                        player_id   = pid_nfl,
                        target      = nfl_target,
                        custom_line = float(nfl_line),
                        is_home     = (nfl_location == "Home"),
                        season      = NFL_SEASON,
                    )

                op_nfl  = round(nfl_pred["over_prob"]  * 100, 1)
                up_nfl  = round(nfl_pred["under_prob"] * 100, 1)
                cc_nfl  = {"High":"#4caf82","Medium":"#d4b44a","Low":"#e05a5a"}
                cb_nfl  = {"High":"rgba(76,175,130,0.12)","Medium":"rgba(212,180,74,0.12)","Low":"rgba(224,90,90,0.12)"}
                cbo_nfl = {"High":"rgba(76,175,130,0.3)","Medium":"rgba(212,180,74,0.3)","Low":"rgba(224,90,90,0.3)"}
                conf    = nfl_pred["confidence"]
                cco_nfl = cc_nfl.get(conf, "#4caf82")
                loc_nfl = "vs." if nfl_location == "Home" else "@"

                # Trend
                l5_nfl  = nfl_pred["recent_avg_5"]
                l10_nfl = nfl_pred["recent_avg_10"]
                if l10_nfl > 0:
                    ratio_nfl = l5_nfl / l10_nfl
                    trend_nfl = "🔥 HOT" if ratio_nfl >= 1.10 else "🥶 COLD" if ratio_nfl <= 0.90 else "→ STEADY"
                else:
                    trend_nfl = "→ STEADY"

                # Edge
                edge_nfl     = round(nfl_pred["predicted_value"] - nfl_line, 1)
                edge_pct_nfl = round(abs(edge_nfl) / max(nfl_line, 1) * 100, 1)
                edge_dir_nfl = "OVER" if edge_nfl >= 0 else "UNDER"

                # Mini chart (recent games)
                rg_nfl = nfl_pred.get("recent_games", [])
                chart_nfl = ""
                if rg_nfl:
                    W,H,PL,PR,PT,PB = 560,130,10,50,15,28
                    vld  = [v for v in rg_nfl if v==v and v is not None]
                    mx   = max(vld)*1.2 if vld else 1
                    mn   = max(0, min(vld)*0.8) if vld else 0
                    rng  = mx - mn if mx != mn else 1
                    av_v = sum(vld)/len(vld) if vld else 0
                    n_nfl = len(rg_nfl)
                    def yx2(v): return PT+(1-(v-mn)/rng)*(H-PT-PB)
                    def xx2(i): return PL+i*(W-PL-PR)/max(n_nfl-1,1)
                    pts2 = " ".join(f"{xx2(i):.1f},{yx2(v):.1f}" for i,v in enumerate(rg_nfl) if v==v)
                    fx2,lx2,by2 = xx2(0),xx2(n_nfl-1),H-PB
                    ap2 = f"M {fx2:.1f},{by2} "+" ".join(f"L {xx2(i):.1f},{yx2(v):.1f}" for i,v in enumerate(rg_nfl) if v==v)+f" L {lx2:.1f},{by2} Z"
                    dots2 = ""
                    for i2,v2 in enumerate(rg_nfl):
                        if v2 != v2: continue
                        cx3,cy3 = xx2(i2),yx2(v2)
                        col3 = pc_nfl if v2 >= nfl_line else "#1e2535"
                        dots2 += f'<circle cx="{cx3:.1f}" cy="{cy3:.1f}" r="4" fill="{col3}" stroke="#080810" stroke-width="1.5"/>'
                        ly5 = cy3-10 if cy3>PT+15 else cy3+17
                        fc3 = "#c8c6c0" if v2 >= nfl_line else "#2a3040"
                        dots2 += f'<text x="{cx3:.1f}" y="{ly5:.1f}" text-anchor="middle" font-size="10" font-weight="600" fill="{fc3}" font-family="Inter,sans-serif">{int(v2)}</text>'
                    ay3 = yx2(av_v); ly6 = yx2(nfl_line)
                    rl2 = f'<line x1="{PL}" y1="{ay3:.1f}" x2="{W-PR}" y2="{ay3:.1f}" stroke="#1e1e2e" stroke-dasharray="3,4" stroke-width="1"/>'
                    rl2 += f'<text x="{W-PR+4}" y="{ay3+4:.1f}" font-size="9" fill="#252535" font-family="Inter,sans-serif">avg {av_v:.1f}</text>'
                    if PT<=ly6<=H-PB:
                        rl2 += f'<line x1="{PL}" y1="{ly6:.1f}" x2="{W-PR}" y2="{ly6:.1f}" stroke="{pc_nfl}" stroke-dasharray="4,3" stroke-width="1.5" opacity="0.6"/>'
                        rl2 += f'<text x="{W-PR+4}" y="{ly6+4:.1f}" font-size="9" fill="{pc_nfl}" font-family="Inter,sans-serif">{nfl_line}</text>'
                    chart_nfl = (f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="overflow:visible;display:block;">'
                                 f'<defs><linearGradient id="ng" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="{pc_nfl}" stop-opacity="0.2"/><stop offset="100%" stop-color="{pc_nfl}" stop-opacity="0"/></linearGradient></defs>'
                                 f'{rl2}<path d="{ap2}" fill="url(#ng)"/>'
                                 f'<polyline points="{pts2}" fill="none" stroke="{pc_nfl}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.9"/>{dots2}</svg>')

                nfl_html = f"""<!DOCTYPE html><html><head>
                <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
                <style>
                *{{box-sizing:border-box;margin:0;padding:0;}} body{{background:transparent;font-family:'Inter',sans-serif;color:#e8e6e0;}}
                .card{{background:#060e06;border-radius:16px;overflow:hidden;border:1px solid #143214;}}
                .top{{height:3px;background:linear-gradient(90deg,{pc_nfl} 0%,transparent 100%);}}
                .hdr{{display:flex;align-items:center;justify-content:space-between;padding:16px 20px 14px;border-bottom:1px solid #0d1a0d;gap:12px;}}
                .hl{{display:flex;align-items:center;gap:12px;flex:1;min-width:0;}}
                .av{{width:52px;height:52px;border-radius:50%;border:2px solid {pc_nfl};object-fit:cover;object-position:top;background:#0d1a0d;flex-shrink:0;}}
                .nm{{font-size:18px;font-weight:700;letter-spacing:-0.02em;color:#f0ede8;line-height:1.1;}}
                .sb{{font-size:11px;color:#2a5a3a;margin-top:3px;}}
                .opp{{color:{pc_nfl};font-weight:600;}}
                .cf{{display:inline-flex;align-items:center;gap:4px;border-radius:20px;padding:3px 10px;
                      font-size:10px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                      background:{cb_nfl.get(conf,"rgba(76,175,130,0.12)")};
                      color:{cco_nfl};border:1px solid {cbo_nfl.get(conf,"rgba(76,175,130,0.3)")};}}
                .cd{{width:5px;height:5px;border-radius:50%;background:{cco_nfl};}}
                .body{{padding:18px 20px 16px;}}
                .hero-row{{display:flex;align-items:flex-end;gap:20px;margin-bottom:18px;padding-bottom:18px;border-bottom:1px solid #0d1a0d;}}
                .pnum{{font-family:'Bebas Neue',sans-serif;font-size:min(96px,18vw);line-height:0.85;color:{pc_nfl};
                        text-shadow:0 0 60px {pc_nfl}44;letter-spacing:0.01em;}}
                .plbl{{font-size:9px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:#1e3a1e;margin-top:6px;}}
                .sgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;flex:1;}}
                .sc{{background:#0d1a0d;border-radius:10px;padding:10px 12px;}}
                .sv{{font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:#e8e6e0;line-height:1;margin-bottom:2px;}}
                .sl{{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#1a3a1a;}}
                .ou{{margin-bottom:16px;}}
                .ot{{font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1a3a1a;margin-bottom:10px;}}
                .or{{display:flex;align-items:center;gap:10px;margin-bottom:7px;}}
                .ol{{font-size:11px;font-weight:500;color:#2a5a3a;min-width:72px;}}
                .ob{{flex:1;height:5px;background:#0d1a0d;border-radius:3px;overflow:hidden;}}
                .fo{{height:100%;background:{pc_nfl};border-radius:3px;width:{op_nfl:.1f}%;}}
                .fu{{height:100%;background:#2a5a9f;border-radius:3px;width:{up_nfl:.1f}%;}}
                .op2{{font-family:'Bebas Neue',sans-serif;font-size:1.3rem;min-width:48px;text-align:right;line-height:1;}}
                .trend{{font-size:10px;font-weight:600;color:#2a5a3a;margin-bottom:14px;}}
                .chart-wrap{{padding-top:14px;border-top:1px solid #0d1a0d;}}
                .meta{{font-size:9px;color:#1a3a1a;margin-top:8px;padding-top:10px;border-top:1px solid #060e06;}}
                </style></head><body>
                <div class="card">
                    <div class="top"></div>
                    <div class="hdr">
                        <div class="hl">
                            {"<img class='av' src='" + hs_nfl + "' onerror=\"this.style.display='none'\"/>" if hs_nfl else ""}
                            <div>
                                <div class="nm">{nfl_sel_player}</div>
                                <div class="sb">{pt_nfl}&nbsp;&middot;&nbsp;{loc_nfl}&nbsp;<span class="opp">NFL</span>&nbsp;&middot;&nbsp;{nfl_location}</div>
                            </div>
                        </div>
                        <span class="cf"><span class="cd"></span>{conf}</span>
                    </div>
                    <div class="body">
                        <div class="hero-row">
                            <div class="proj">
                                <div class="pnum">{nfl_pred["predicted_value"]}</div>
                                <div class="plbl">Projected {nfl_tinfo["label"]}</div>
                            </div>
                            <div class="sgrid">
                                <div class="sc"><div class="sv">{l5_nfl}</div><div class="sl">L5 Avg</div></div>
                                <div class="sc"><div class="sv">{l10_nfl}</div><div class="sl">L10 Avg</div></div>
                                <div class="sc"><div class="sv">{nfl_line}</div><div class="sl">Line</div></div>
                            </div>
                        </div>
                        <div class="trend">{trend_nfl}&nbsp;&nbsp;Edge:&nbsp;
                            <strong style="color:{'#4caf82' if edge_nfl >= 0 else '#e05a5a'}">
                                {'▲' if edge_nfl >= 0 else '▼'} {edge_dir_nfl} {abs(edge_nfl)} ({edge_pct_nfl}%)
                            </strong>
                        </div>
                        <div class="ou">
                            <div class="ot">Over / Under {nfl_line} {nfl_tinfo["short"]}</div>
                            <div class="or">
                                <div class="ol">Over {nfl_line}</div>
                                <div class="ob"><div class="fo"></div></div>
                                <div class="op2" style="color:{pc_nfl};">{op_nfl:.1f}%</div>
                            </div>
                            <div class="or">
                                <div class="ol">Under {nfl_line}</div>
                                <div class="ob"><div class="fu"></div></div>
                                <div class="op2" style="color:#4a9eff;">{up_nfl:.1f}%</div>
                            </div>
                        </div>
                        {"<div class='chart-wrap'><div style='font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#1a3a1a;margin-bottom:10px;'>Last " + str(n_nfl) + " games &mdash; " + nfl_tinfo["label"] + "</div>" + chart_nfl + "</div>" if chart_nfl else ""}
                        <div class="meta">Rolling-average model &nbsp;&middot;&nbsp; {nfl_pred["n_games"]} games &nbsp;&middot;&nbsp; {NFL_SEASON} season &nbsp;&middot;&nbsp; Std dev {nfl_pred["std_dev"]}</div>
                    </div>
                </div></body></html>"""

                components.html(nfl_html, height=580, scrolling=False)

                nfl_share = (
                    f"🏈 LineBreaker NFL Prediction\n"
                    f"{nfl_sel_player} — {nfl_tinfo['label']}\n"
                    f"Projected: {nfl_pred['predicted_value']} | Line: {nfl_line}\n"
                    f"{'OVER' if op_nfl > 50 else 'UNDER'} {nfl_line} ({op_nfl:.1f}%)\n"
                    f"Confidence: {conf}"
                )
                st.code(nfl_share, language=None)

        elif not nfl_opts:
            pass  # warning already shown in left panel
        else:
            # Placeholder
            ph_nfl = f"""<!DOCTYPE html><html><head>
            <link href='https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@600&display=swap' rel='stylesheet'>
            <style>*{{box-sizing:border-box;margin:0;}} body{{background:transparent;}}
            .w{{position:relative;overflow:hidden;border-radius:16px;background:#060e06;border:1px solid #143214;
                 min-height:420px;display:flex;align-items:center;justify-content:center;}}
            .bg{{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}}
            .i{{position:relative;z-index:1;text-align:center;padding:2rem;}}
            .lg{{font-family:'Bebas Neue',sans-serif;font-size:5rem;color:#4caf82;opacity:0.07;line-height:1;margin-bottom:1rem;}}
            .t{{font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1a3a1a;}}
            .s{{font-size:10px;color:#111a11;letter-spacing:1px;margin-top:6px;}}
            </style></head><body>
            <div class='w'><div class='bg'>{_svg_field()}</div>
            <div class='i'><div class='lg'>NFL</div>
            <div class='t'>Select a player &amp; run prediction</div>
            <div class='s'>Beat the Line. Break the Line.</div>
            </div></div></body></html>"""
            components.html(ph_nfl, height=420, scrolling=False)