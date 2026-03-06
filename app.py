import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
import re

st.set_page_config(
    page_title="AERIE Risk Intelligence",
    page_icon="assets/icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --white:       #FFFFFF;
    --bg:          #F7F8FA;
    --surface:     #FFFFFF;
    --border:      #E2E6EC;
    --border-dark: #C8CDD6;
    --navy:        #0F1C2E;
    --navy-mid:    #1E3A5F;
    --blue:        #1D6FA4;
    --blue-light:  #EBF4FB;
    --muted:       #6B7A90;
    --muted-light: #9BA8B8;
    --green:       #1A7F5A;
    --green-bg:    #EAF7F2;
    --amber:       #A05C00;
    --amber-bg:    #FEF3E2;
    --red:         #B91C1C;
    --red-bg:      #FEF2F2;
    --divider:     #E2E6EC;
}

* { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: var(--navy) !important;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background-color: var(--navy) !important;
    border-right: none !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: rgba(255,255,255,0.65) !important;
    font-size: 0.78rem !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255,255,255,0.5) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0 !important;
    text-transform: none !important;
    padding: 7px 10px !important;
    border-radius: 6px !important;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #FFFFFF !important;
}

/* ---- TYPOGRAPHY ---- */
h1 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.65rem !important;
    color: var(--navy) !important;
    letter-spacing: -0.01em !important;
    margin-bottom: 0.25rem !important;
    line-height: 1.2 !important;
}
h2 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    color: var(--navy) !important;
    margin-bottom: 0.75rem !important;
}
h3 {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--navy-mid) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
p, li { color: var(--muted) !important; font-size: 0.85rem !important; line-height: 1.6 !important; }

/* ---- PAGE HEADER CUSTOM ---- */
.ph-wrap {
    padding: 24px 0 20px 0;
    border-bottom: 1px solid var(--divider);
    margin-bottom: 28px;
}
.ph-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue);
    margin-bottom: 6px;
    font-weight: 500;
}
.ph-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.7rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.ph-sub {
    font-size: 0.83rem;
    color: var(--muted);
    margin-top: 5px;
    font-weight: 400;
}

/* ---- SECTION LABELS ---- */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.64rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted-light);
    font-weight: 500;
    display: block;
    margin-bottom: 10px;
    margin-top: 4px;
}

/* ---- METRIC CARDS ---- */
div[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-top: 3px solid var(--blue) !important;
    border-radius: 8px !important;
    padding: 18px 20px !important;
}
div[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 1.9rem !important;
    font-weight: 600 !important;
    color: var(--navy) !important;
    line-height: 1.15 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ---- BUTTONS ---- */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: var(--navy) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s, transform 0.1s !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: var(--navy-mid) !important;
    transform: translateY(-1px) !important;
}

/* ---- FORM ---- */
div[data-testid="stForm"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 24px !important;
}

/* ---- INPUTS ---- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    background: var(--white) !important;
    border-color: var(--border-dark) !important;
    border-radius: 6px !important;
    color: var(--navy) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 3px rgba(29,111,164,0.1) !important;
}
input[type="number"] {
    background: var(--white) !important;
    color: var(--navy) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border: 1px solid var(--border-dark) !important;
    border-radius: 6px !important;
}
.stNumberInput label,
.stSlider label,
.stSelectbox label,
.stTextInput label,
.stTextArea label,
.stFileUploader label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}

/* ---- SLIDER ---- */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--blue) !important;
    border-color: var(--white) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}
.stSlider > div > div > div:first-child { background: var(--blue) !important; }

/* ---- ALERTS ---- */
.stSuccess {
    background: var(--green-bg) !important;
    border-left: 3px solid var(--green) !important;
    color: var(--green) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stError {
    background: var(--red-bg) !important;
    border-left: 3px solid var(--red) !important;
    color: var(--red) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
.stWarning {
    background: var(--amber-bg) !important;
    border-left: 3px solid var(--amber) !important;
    color: var(--amber) !important;
    border-radius: 6px !important;
}
.stInfo {
    background: var(--blue-light) !important;
    border-left: 3px solid var(--blue) !important;
    color: var(--navy-mid) !important;
    border-radius: 6px !important;
}

/* Force alert text colours */
.stSuccess p, .stSuccess span { color: var(--green) !important; font-size: 0.85rem !important; }
.stError   p, .stError   span { color: var(--red)   !important; font-size: 0.85rem !important; }
.stWarning p, .stWarning span { color: var(--amber) !important; font-size: 0.85rem !important; }
.stInfo    p, .stInfo    span { color: var(--navy-mid) !important; font-size: 0.85rem !important; }

/* ---- DATAFRAMES ---- */
.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ---- FILE UPLOADER ---- */
div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1.5px dashed var(--border-dark) !important;
    border-radius: 8px !important;
}
div[data-testid="stFileUploader"]:hover {
    border-color: var(--blue) !important;
}

/* ---- CAPTION / MONO ---- */
.stCaption, small {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--muted-light) !important;
    font-size: 0.7rem !important;
}

/* ---- JSON ---- */
.stJson {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ---- CODE ---- */
code, pre {
    background: var(--bg) !important;
    color: var(--navy-mid) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    border: 1px solid var(--border) !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}

/* ---- SPINNER ---- */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* ---- DIVIDER ---- */
hr { border-color: var(--divider) !important; margin: 1.5rem 0 !important; }

/* ---- RISK BADGE ---- */
.risk-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}
.risk-low    { background: var(--green-bg); color: var(--green);     border: 1px solid #A7D7C5; }
.risk-medium { background: var(--amber-bg); color: var(--amber);     border: 1px solid #F6C980; }
.risk-high   { background: var(--red-bg);   color: var(--red);       border: 1px solid #FCA5A5; }
.risk-online { background: #EAF7F2; color: #1A7F5A; border: 1px solid #A7D7C5; }

/* ---- INFO PANEL ---- */
.info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.82rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--muted); font-weight: 400; }
.info-val { color: var(--navy); font-weight: 500; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }

/* ---- USAGE CARD ---- */
.usage-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--blue);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.usage-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--navy);
    margin-bottom: 4px;
}
.usage-desc {
    font-size: 0.8rem;
    color: var(--muted);
    line-height: 1.5;
}

/* ---- SIDEBAR IDENTITY ---- */
.sidebar-id {
    padding: 28px 18px 20px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 8px;
}
.sidebar-name {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: 0.04em;
}
.sidebar-full {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: rgba(255,255,255,0.38);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 4px;
}
.sidebar-feat-item {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: rgba(255,255,255,0.4);
    padding: 3px 0;
    display: flex;
    gap: 10px;
}
.sidebar-feat-num { color: rgba(255,255,255,0.2); min-width: 18px; }
.sidebar-feat-name { color: rgba(255,255,255,0.5); }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ================================================================
# PLOTLY THEME
# ================================================================
_AXIS = dict(gridcolor='#E2E6EC', linecolor='#E2E6EC', tickfont=dict(color='#6B7A90', size=10))

PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#FAFBFC',
    font=dict(family='IBM Plex Mono, monospace', color='#6B7A90', size=11),
    title=dict(font=dict(family='IBM Plex Sans, sans-serif', color='#0F1C2E', size=13), x=0),
    margin=dict(t=44, b=36, l=16, r=16),
)

def pt(**overrides):
    layout = dict(PT)
    layout['xaxis'] = {**_AXIS, **overrides.pop('xaxis', {})}
    layout['yaxis'] = {**_AXIS, **overrides.pop('yaxis', {})}
    layout.update(overrides)
    return layout

# ================================================================
# MODEL
# ================================================================
@st.cache_resource
def load_model():
    m = joblib.load('aerie_model.pkl')
    s = joblib.load('aerie_scaler.pkl')
    with open('feature_list.pkl', 'rb') as f:
        fl = pickle.load(f)
    return m, s, fl

try:
    model, scaler, feature_list = load_model()
    model_ok = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()
    model_ok = False

FEATURE_META = {
    'severity':                   {"label": "Severity",               "min": 1,   "max": 5,      "default": 3,     "step": 1,    "fmt": ".0f"},
    'downtime':                   {"label": "Downtime (hrs)",         "min": 0.0, "max": 100.0,  "default": 5.0,   "step": 1.0,  "fmt": ".1f"},
    'financial_impact':           {"label": "Financial Impact ($)",   "min": 0,   "max": 500000, "default": 50000, "step": 5000, "fmt": ",.0f"},
    'regulatory_flag':            {"label": "Regulatory Flag",        "min": 0,   "max": 1,      "default": 0,     "step": 1,    "fmt": ".0f"},
    'data_sensitivity':           {"label": "Data Sensitivity",       "min": 0.0, "max": 1.0,    "default": 0.5,   "step": 0.05, "fmt": ".2f"},
    'criticality':                {"label": "Criticality",            "min": 1,   "max": 5,      "default": 3,     "step": 1,    "fmt": ".0f"},
    'severity_x_data_sensitivity':{"label": "Severity x Sensitivity", "min": 0.0, "max": 5.0,    "default": 1.5,   "step": 0.1,  "fmt": ".2f"},
    'asset_incident_prev_count':  {"label": "Prior Incidents",        "min": 0,   "max": 20,     "default": 0,     "step": 1,    "fmt": ".0f"},
    'days_since_audit':           {"label": "Days Since Audit",       "min": 0,   "max": 365,    "default": 30,    "step": 5,    "fmt": ".0f"},
}

# ================================================================
# HELPERS
# ================================================================
def predict_single(d):
    df = pd.DataFrame([d])[feature_list]
    return model.predict(scaler.transform(df))[0], model.predict_proba(scaler.transform(df))[0][1]

def predict_batch(df):
    missing = set(feature_list) - set(df.columns)
    if missing:
        return None, f"Missing columns: {missing}"
    df_in = df[feature_list].copy().fillna(df[feature_list].median())
    scaled = scaler.transform(df_in)
    return pd.DataFrame({
        'predicted_major_event': model.predict(scaled),
        'probability': model.predict_proba(scaled)[:,1]
    }), None

def risk_badge(p):
    if p < 0.3:   return '<span class="risk-badge risk-low">Low Risk</span>'
    elif p < 0.7: return '<span class="risk-badge risk-medium">Medium Risk</span>'
    return            '<span class="risk-badge risk-high">High Risk</span>'

def gauge_chart(proba):
    if proba < 0.3:   bar_color, label = "#1A7F5A", "LOW"
    elif proba < 0.7: bar_color, label = "#A05C00", "MEDIUM"
    else:             bar_color, label = "#B91C1C", "HIGH"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 38, "family": "IBM Plex Sans", "color": "#0F1C2E"}},
        gauge={
            "axis": {"range": [0,100], "tickcolor":"#C8CDD6",
                     "tickfont": {"color":"#9BA8B8","size":10,"family":"IBM Plex Mono"}},
            "bar": {"color": bar_color, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range":[0,30],  "color":"rgba(26,127,90,0.08)"},
                {"range":[30,70], "color":"rgba(160,92,0,0.07)"},
                {"range":[70,100],"color":"rgba(185,28,28,0.08)"},
            ],
            "threshold":{"line":{"color":bar_color,"width":2},"thickness":0.78,"value":proba*100},
        },
        title={"text": f"MAJOR EVENT PROBABILITY — {label}",
               "font":{"size":9,"family":"IBM Plex Mono","color":"#9BA8B8"}},
    ))
    fig.update_layout(height=240, paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=40, b=8, l=20, r=20))
    return fig

def sweep_chart(base_dict, sweep_feature):
    meta = FEATURE_META[sweep_feature]
    vals = np.linspace(meta["min"], meta["max"], 50)
    probas = []
    for v in vals:
        d = base_dict.copy()
        d[sweep_feature] = v
        d['severity_x_data_sensitivity'] = d['severity'] * d['data_sensitivity']
        _, p = predict_single(d)
        probas.append(p * 100)

    cv = base_dict[sweep_feature]
    _, cp = predict_single(base_dict)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=probas, mode='lines',
        line=dict(color='#1D6FA4', width=2),
        fill='tozeroy', fillcolor='rgba(29,111,164,0.07)',
        name="Risk %", showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[cv], y=[cp*100], mode='markers',
        marker=dict(size=9, color='#1D6FA4', symbol='circle',
                    line=dict(color='white', width=2)),
        name="Current value", showlegend=False
    ))
    fig.add_vline(x=cv, line_dash="dash", line_color="#C8CDD6", line_width=1.5,
        annotation_text=f"Current: {cv:{meta['fmt']}}",
        annotation_font_color="#6B7A90", annotation_font_size=10,
        annotation_bgcolor="white", annotation_bordercolor="#E2E6EC")
    fig.add_hline(y=50, line_dash="dot", line_color="#E2E6EC", line_width=1,
        annotation_text="50%", annotation_font_color="#C8CDD6",
        annotation_font_size=9, annotation_position="right")

    fig.update_layout(**pt(
        title=f"Sensitivity — {meta['label']}",
        xaxis_title=meta['label'],
        yaxis_title="Risk Probability (%)",
        yaxis=dict(range=[0,108]),
        height=320,
    ))
    return fig

def get_available_gemini_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        usable = [
            m["name"].replace("models/","")
            for m in r.json().get("models",[])
            if "generateContent" in m.get("supportedGenerationMethods",[])
            and "gemini" in m.get("name","").lower()
            and "vision" not in m.get("name","").lower()
            and "embedding" not in m.get("name","").lower()
        ]
        usable.sort(key=lambda x: (0 if "flash" in x else 1, x))
        return usable or ["gemini-2.0-flash","gemini-1.5-flash-8b","gemini-1.5-flash"]
    except Exception:
        return ["gemini-2.0-flash","gemini-1.5-flash-8b","gemini-1.5-flash"]

def call_gemini_api(prompt, api_key):
    last_error = None
    for mn in get_available_gemini_models(api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{mn}:generateContent?key={api_key}"
        try:
            r = requests.post(url, json={
                "contents":[{"parts":[{"text":prompt}]}],
                "generationConfig":{"temperature":0.7,"maxOutputTokens":1024}
            }, timeout=60)
            if r.status_code in (429,404,403):
                last_error = f"{r.status_code} on {mn}"; continue
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"], mn
        except Exception as e:
            last_error = str(e); continue
    raise Exception(f"No working Gemini model found.\nLast error: {last_error}")

def parse_csv_block(text):
    m = re.search(r'```(?:csv)?\s*\n(.*?)```', text, re.DOTALL|re.IGNORECASE)
    if m: return m.group(1).strip()
    for i, line in enumerate(text.splitlines()):
        if 'severity' in line.lower() and ',' in line:
            return "\n".join(text.splitlines()[i:])
    return text.strip()

def page_header(label, title, sub):
    st.markdown(f"""
    <div class="ph-wrap">
        <div class="ph-label">{label}</div>
        <div class="ph-title">{title}</div>
        <div class="ph-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown(f"""
<div class="sidebar-id">
    <div class="sidebar-name">AERIE</div>
    <div class="sidebar-full">Adaptive Enterprise Risk Intelligence Engine</div>
    <div style="margin-top:14px;">
        <span class="risk-badge risk-online">Model Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", [
    "Single Prediction",
    "Batch Upload",
    "Scenario Simulator",
    "Model Info",
    "AI Scenario Generator"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-family:IBM Plex Mono,monospace;font-size:0.6rem;letter-spacing:0.14em;'
    'text-transform:uppercase;color:rgba(255,255,255,0.25);padding:0 0 8px 0;">Feature Index</div>',
    unsafe_allow_html=True
)
for i, f in enumerate(feature_list):
    st.sidebar.markdown(
        f'<div class="sidebar-feat-item">'
        f'<span class="sidebar-feat-num">{i+1:02d}</span>'
        f'<span class="sidebar-feat-name">{f}</span></div>',
        unsafe_allow_html=True
    )

# ================================================================
# SINGLE PREDICTION
# ================================================================
if page == "Single Prediction":
    page_header("Incident Analysis", "Single Incident Predictor",
                "Enter incident parameters to forecast major event probability")

    with st.form("pred_form"):
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            st.markdown('<span class="sec-label">Incident Severity</span>', unsafe_allow_html=True)
            severity = st.slider("Severity (1–5)", 1, 5, 3)
            downtime = st.number_input("Downtime (hours)", 0.0, 100.0, 5.0)
            financial_impact = st.number_input("Financial Impact ($)", 0, 500000, 50000, step=5000)
            regulatory_flag = st.selectbox("Regulatory Flag", [0, 1])
        with c2:
            st.markdown('<span class="sec-label">Asset Profile</span>', unsafe_allow_html=True)
            data_sensitivity = st.slider("Data Sensitivity (0–1)", 0.0, 1.0, 0.5)
            criticality = st.slider("Criticality (1–5)", 1, 5, 3)
            asset_incident_prev_count = st.number_input("Prior Incidents on Asset", 0, 20, 0)
            days_since_audit = st.number_input("Days Since Last Audit", 0, 365, 30)
        with c3:
            st.markdown('<span class="sec-label">Input Summary</span>', unsafe_allow_html=True)
            sxd = severity * data_sensitivity
            st.metric("Severity x Sensitivity", f"{sxd:.2f}")
            st.json({
                "severity": severity, "downtime": downtime,
                "financial_impact": financial_impact, "regulatory_flag": regulatory_flag,
                "data_sensitivity": data_sensitivity, "criticality": criticality,
                "sev_x_sens": round(sxd,3), "prior_incidents": asset_incident_prev_count,
                "days_since_audit": days_since_audit
            })
        st.form_submit_button("Run Prediction", use_container_width=True)

    # Run prediction on every form state (not just submit) by reading values
    d = {
        'severity': severity, 'downtime': downtime, 'financial_impact': financial_impact,
        'regulatory_flag': regulatory_flag, 'data_sensitivity': data_sensitivity,
        'criticality': criticality, 'severity_x_data_sensitivity': sxd,
        'asset_incident_prev_count': asset_incident_prev_count, 'days_since_audit': days_since_audit
    }
    pred, proba = predict_single(d)

    st.markdown("---")
    r1, r2, r3 = st.columns([1,1,2])
    with r1:
        if pred == 1:
            st.error("Major Event Predicted")
        else:
            st.success("Minor / Routine Incident")
    with r2:
        st.metric("Risk Probability", f"{proba:.1%}")
        st.markdown(risk_badge(proba), unsafe_allow_html=True)
    with r3:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)

    st.markdown('<span class="sec-label">Feature Importance</span>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker=dict(color=imp_df['Importance'],
                    colorscale=[[0,'#EBF4FB'],[1,'#1D6FA4']],
                    showscale=False)
    ))
    fig.update_layout(**pt(height=280, xaxis_title="Importance Score"))
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# BATCH UPLOAD
# ================================================================
elif page == "Batch Upload":
    page_header("Batch Processing", "Batch Risk Scoring",
                "Upload a CSV to score multiple incidents simultaneously")

    st.download_button(
        "Download Template CSV",
        pd.DataFrame(columns=feature_list).to_csv(index=False),
        "aerie_template.csv", "text/csv"
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload incidents CSV", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown('<span class="sec-label">Data Preview</span>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Score All Incidents", use_container_width=True):
            with st.spinner("Running batch inference..."):
                results, err = predict_batch(df)
            if err:
                st.error(err)
            else:
                out = pd.concat([df, results], axis=1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Major Events", int(results['predicted_major_event'].sum()))
                m2.metric("Avg Probability", f"{results['probability'].mean():.1%}")
                m3.metric("High Risk  >70%", int((results['probability'] > 0.7).sum()))
                st.markdown("---")
                st.markdown('<span class="sec-label">Scored Results</span>', unsafe_allow_html=True)
                st.dataframe(out, use_container_width=True)

                fig = px.histogram(results, x='probability', nbins=20,
                    title="Risk Probability Distribution",
                    color_discrete_sequence=['#1D6FA4'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="#A05C00",
                    annotation_text="Decision boundary",
                    annotation_font_color="#A05C00", annotation_font_size=10)
                fig.update_layout(**pt())
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download Results", out.to_csv(index=False),
                                   "aerie_predictions.csv", "text/csv")

# ================================================================
# SCENARIO SIMULATOR
# ================================================================
elif page == "Scenario Simulator":
    page_header("What-If Analysis", "Scenario Simulator",
                "Adjust parameters and sweep any variable to see its effect on predicted risk")

    c1, c2 = st.columns(2)
    with c1:
        severity   = st.slider("Severity", 1, 5, 3)
        downtime   = st.slider("Downtime (hrs)", 0.0, 100.0, 5.0)
        financial  = st.slider("Financial Impact ($)", 0, 500000, 50000, step=5000)
        reg_flag   = st.selectbox("Regulatory Flag", [0, 1])
    with c2:
        data_sens  = st.slider("Data Sensitivity", 0.0, 1.0, 0.5)
        crit       = st.slider("Criticality", 1, 5, 3)
        prev_count = st.slider("Prior Incidents", 0, 20, 0)
        audit_days = st.slider("Days Since Audit", 0, 365, 30)

    base = {
        'severity': severity, 'downtime': downtime, 'financial_impact': financial,
        'regulatory_flag': reg_flag, 'data_sensitivity': data_sens, 'criticality': crit,
        'severity_x_data_sensitivity': severity * data_sens,
        'asset_incident_prev_count': prev_count, 'days_since_audit': audit_days,
    }
    pred, proba = predict_single(base)

    st.markdown("---")
    gc, vc = st.columns([1.6, 1])
    with gc:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)
    with vc:
        st.markdown('<span class="sec-label">Assessment</span>', unsafe_allow_html=True)
        if pred == 1:
            st.error("Major Event Predicted")
        else:
            st.success("Minor / Routine Incident")
        st.metric("Risk Probability", f"{proba:.1%}")
        st.markdown(risk_badge(proba), unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.markdown('<span class="sec-label">Current Parameters</span>', unsafe_allow_html=True)
        rows = [
            ("Severity", severity), ("Downtime", f"{downtime:.1f}h"),
            ("Financial Impact", f"${financial:,}"), ("Criticality", crit),
            ("Prior Incidents", prev_count), ("Days Since Audit", audit_days),
        ]
        for k, v in rows:
            st.markdown(
                f'<div class="info-row"><span class="info-key">{k}</span>'
                f'<span class="info-val">{v}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<span class="sec-label">Sensitivity Analysis</span>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#6B7A90;margin:-2px 0 14px 0;">'
        'Select a variable to sweep across its full range. All others remain fixed.</p>',
        unsafe_allow_html=True
    )
    sweep_feat = st.selectbox(
        "Variable to sweep",
        [f for f in feature_list if f != 'severity_x_data_sensitivity'],
        format_func=lambda x: FEATURE_META[x]["label"]
    )
    with st.spinner("Computing sensitivity..."):
        st.plotly_chart(sweep_chart(base, sweep_feat), use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="sec-label">Worst-Case Risk Delta</span>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#6B7A90;margin:-2px 0 14px 0;">'
        'Risk increase (percentage points) when each factor is pushed to its maximum independently.</p>',
        unsafe_allow_html=True
    )
    deltas = {}
    for feat in feature_list:
        if feat == 'severity_x_data_sensitivity': continue
        w = base.copy()
        w[feat] = FEATURE_META[feat]["max"]
        w['severity_x_data_sensitivity'] = w['severity'] * w['data_sensitivity']
        _, wp = predict_single(w)
        deltas[FEATURE_META[feat]["label"]] = round((wp - proba) * 100, 2)

    ddf = pd.DataFrame(list(deltas.items()), columns=["Feature","Delta (pp)"]).sort_values("Delta (pp)")
    fig2 = go.Figure(go.Bar(
        x=ddf["Delta (pp)"], y=ddf["Feature"], orientation='h',
        marker_color=['#B91C1C' if v > 0 else '#1A7F5A' for v in ddf["Delta (pp)"]],
        marker_opacity=0.8
    ))
    fig2.update_layout(**pt(
        title="Risk delta when each feature is maximised",
        xaxis_title="Percentage-point change", height=300,
        xaxis=dict(zeroline=True, zerolinecolor='#C8CDD6', zerolinewidth=1.5),
    ))
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# MODEL INFO
# ================================================================
elif page == "Model Info":
    page_header("System Overview", "Model Information",
                "Architecture, feature weights, and usage reference")

    m1, m2, m3 = st.columns(3)
    m1.metric("Model Type", "Random Forest")
    m2.metric("Decision Trees", model.n_estimators)
    m3.metric("Input Features", len(feature_list))

    st.markdown("---")
    st.markdown('<span class="sec-label">Feature Importances</span>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker=dict(color=imp_df['Importance'],
                    colorscale=[[0,'#EBF4FB'],[0.5,'#6BAED6'],[1,'#1D6FA4']],
                    showscale=False)
    ))
    fig.update_layout(**pt(height=300, xaxis_title="Importance Score"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="sec-label">Usage Reference</span>', unsafe_allow_html=True)
    for mode, desc in [
        ("Single Prediction", "Enter one incident manually and get an instant risk score with feature breakdown"),
        ("Batch Upload", "Score multiple incidents from a CSV file in a single pass"),
        ("Scenario Simulator", "Sweep any input variable and observe how the risk curve responds"),
        ("AI Generator", "Use Gemini to write structured incident scenarios, then auto-score them"),
    ]:
        st.markdown(
            f'<div class="usage-card">'
            f'<div class="usage-title">{mode}</div>'
            f'<div class="usage-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# ================================================================
# AI SCENARIO GENERATOR
# ================================================================
elif page == "AI Scenario Generator":
    page_header("Generative Analysis", "AI Scenario Generator",
                "Gemini generates structured incident scenarios — AERIE scores them automatically")

    try:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
        st.sidebar.markdown(
            '<div style="margin-top:8px;"><span class="risk-badge risk-online">Gemini Connected</span></div>',
            unsafe_allow_html=True
        )
    except:
        GEMINI_KEY = st.text_input(
            "Google Gemini API Key", type="password",
            help="Free key at aistudio.google.com/app/apikey — add to Streamlit secrets as GEMINI_API_KEY to avoid re-entering"
        )

    if not GEMINI_KEY:
        st.markdown("""
        <div class="usage-card" style="border-left-color:#A05C00;">
            <div class="usage-title" style="color:#A05C00;">API Key Required</div>
            <div class="usage-desc">
            Get a free key at <code>aistudio.google.com/app/apikey</code>.
            Add it to Streamlit secrets as <code>GEMINI_API_KEY</code> to avoid entering it each session.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    a1, a2 = st.columns(2)
    with a1:
        industry = st.selectbox("Industry / Context", [
            "Financial Services","Healthcare","Manufacturing",
            "Government","Retail","Energy and Utilities"
        ])
        n_scenarios = st.slider("Number of Scenarios", 3, 10, 5)
    with a2:
        threat = st.selectbox("Threat Focus", [
            "Mixed / Varied","Cybersecurity","Operational Failures",
            "Data Breaches","Regulatory Incidents","Third-party / Supply Chain"
        ])
        sev_bias = st.selectbox("Severity Bias", [
            "Realistic mix","Mostly high-severity","Mostly low-severity"
        ])

    extra = st.text_area(
        "Additional Context (optional)",
        placeholder="E.g. 'Focus on cloud infrastructure' or 'servers are wet from flood'",
        height=80
    )

    prompt = f"""You are a risk analyst for {industry}. Generate exactly {n_scenarios} incident scenarios. Focus: {threat}. Severity bias: {sev_bias}.{' Context: ' + extra if extra else ''}

Output ONLY a valid CSV block. Use these exact headers:
severity,downtime,financial_impact,regulatory_flag,data_sensitivity,criticality,asset_incident_prev_count,days_since_audit,description

Constraints:
- severity: int 1-5
- downtime: float hours 0-100
- financial_impact: int 0-500000
- regulatory_flag: 0 or 1
- data_sensitivity: float 0.0-1.0
- criticality: int 1-5
- asset_incident_prev_count: int 0-20
- days_since_audit: int 0-365
- description: one sentence, NO commas inside it

Example:
4,36.5,180000,1,0.85,4,3,120,Ransomware attack encrypting finance department servers

CSV:"""

    if st.button("Generate and Score Scenarios", use_container_width=True):
        with st.spinner("Generating scenarios..."):
            try:
                text, used_model = call_gemini_api(prompt, GEMINI_KEY)
                st.caption(f"Generated by {used_model}")

                csv_text = parse_csv_block(text)
                try:
                    scenarios_df = pd.read_csv(io.StringIO(csv_text))
                except Exception:
                    st.error("Could not parse structured data from AI output.")
                    st.code(text)
                    st.stop()

                scored, err = predict_batch(scenarios_df)
                if err:
                    st.warning(f"Scoring issue: {err}")
                    st.dataframe(scenarios_df)
                else:
                    out = pd.concat([scenarios_df, scored], axis=1)\
                            .sort_values('probability', ascending=False)\
                            .reset_index(drop=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Major Events", int(scored['predicted_major_event'].sum()))
                    m2.metric("Avg Risk", f"{scored['probability'].mean():.1%}")
                    m3.metric("High Risk  >70%", int((scored['probability'] > 0.7).sum()))

                    st.markdown("---")
                    st.markdown('<span class="sec-label">Scored Scenarios</span>', unsafe_allow_html=True)

                    def highlight(row):
                        p = row['probability']
                        c = ('rgba(185,28,28,0.05)' if p >= 0.7
                             else 'rgba(160,92,0,0.05)' if p >= 0.3
                             else 'rgba(26,127,90,0.05)')
                        return [f'background-color:{c}'] * len(row)

                    disp = [c for c in [
                        'description','severity','downtime','financial_impact',
                        'regulatory_flag','data_sensitivity','criticality',
                        'probability','predicted_major_event'
                    ] if c in out.columns]

                    st.dataframe(
                        out[disp].style.apply(highlight, axis=1).format({
                            'probability': '{:.1%}',
                            'financial_impact': '${:,.0f}',
                            'data_sensitivity': '{:.2f}',
                            'downtime': '{:.1f}h'
                        }),
                        use_container_width=True
                    )

                    colors = ['#B91C1C' if p >= 0.7 else ('#A05C00' if p >= 0.3 else '#1A7F5A')
                              for p in out['probability']]
                    fig = go.Figure(go.Bar(
                        x=list(range(len(out))), y=out['probability'],
                        marker_color=colors, marker_opacity=0.8,
                        text=[f"{p:.0%}" for p in out['probability']],
                        textposition='outside',
                        textfont=dict(family='IBM Plex Mono', size=10, color='#6B7A90')
                    ))
                    fig.update_layout(**pt(
                        title="Risk Probability by Scenario (sorted highest to lowest)",
                        xaxis_title="Scenario Index",
                        yaxis_title="Risk Probability",
                        yaxis=dict(range=[0,1.18], tickformat='.0%'),
                        height=300, showlegend=False
                    ))
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "Download Scored Scenarios",
                        out.to_csv(index=False),
                        "aerie_ai_scenarios.csv", "text/csv"
                    )

            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again in 30 seconds.")
            except Exception as e:
                st.error(f"Error: {e}")
