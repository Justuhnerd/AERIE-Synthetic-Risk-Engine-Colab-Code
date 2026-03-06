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
    page_title="AERIE — Risk Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# CSS — Clean Editorial Theme
# Fonts: Fraunces (display) + IBM Plex Sans (body) + IBM Plex Mono (data)
# Palette: Pure white, near-black, warm gray, single red accent
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@300;400&display=swap');

:root {
    --white:       #FFFFFF;
    --off-white:   #F9F8F6;
    --rule:        #E5E2DC;
    --muted:       #C4BFB8;
    --body:        #5C5750;
    --strong:      #1A1714;
    --accent:      #C0392B;
    --accent-soft: #F5EAE9;
    --mono-bg:     #F4F2EF;
}

/* ---------- GLOBAL ---------- */
html, body, .stApp {
    background: var(--white) !important;
    color: var(--strong) !important;
}

/* ---------- MAIN CONTENT AREA ---------- */
section.main > div { padding-top: 2.5rem !important; }

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {
    background: var(--off-white) !important;
    border-right: 1px solid var(--rule) !important;
}
section[data-testid="stSidebar"] * { color: var(--body) !important; }
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    color: var(--body) !important;
    padding: 5px 8px !important;
    border-radius: 3px !important;
    transition: background 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: var(--rule) !important;
}

/* ---------- HEADINGS ---------- */
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 400 !important;
    color: var(--strong) !important;
    letter-spacing: -0.02em !important;
}
h1 {
    font-size: 2.2rem !important;
    font-weight: 300 !important;
    border-bottom: 1px solid var(--rule) !important;
    padding-bottom: 0.75rem !important;
    margin-bottom: 0.5rem !important;
}
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.1rem !important; }

/* ---------- BODY TEXT ---------- */
p, li, .stMarkdown p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 300 !important;
    color: var(--body) !important;
    line-height: 1.65 !important;
}

/* ---------- LABELS (sliders, inputs, selects) ---------- */
label, .stSlider label, .stSelectbox label,
.stNumberInput label, .stTextInput label, .stTextArea label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}

/* ---------- METRICS ---------- */
div[data-testid="metric-container"] {
    background: var(--off-white) !important;
    border: 1px solid var(--rule) !important;
    border-top: 2px solid var(--strong) !important;
    border-radius: 0 !important;
    padding: 20px 22px 18px !important;
}
div[data-testid="stMetricLabel"] > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    font-weight: 400 !important;
}
div[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 2rem !important;
    font-weight: 400 !important;
    color: var(--strong) !important;
}

/* ---------- BUTTONS ---------- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: var(--strong) !important;
    color: var(--white) !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.6rem !important;
    transition: background 0.15s !important;
    box-shadow: none !important;
}
.stButton > button:hover, .stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: var(--accent) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ---------- INPUTS ---------- */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    background: var(--white) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    color: var(--strong) !important;
    transition: border-color 0.15s !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--strong) !important;
    box-shadow: none !important;
}
input[type="number"] {
    background: var(--white) !important;
    color: var(--strong) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid var(--rule) !important;
    border-radius: 2px !important;
}

/* ---------- SLIDER ---------- */
.stSlider > div > div > div {
    background: var(--strong) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--white) !important;
    border: 2px solid var(--strong) !important;
    box-shadow: none !important;
    width: 14px !important; height: 14px !important;
}

/* ---------- FORM CONTAINER ---------- */
div[data-testid="stForm"] {
    background: var(--off-white) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    padding: 28px !important;
}

/* ---------- ALERTS ---------- */
div[data-testid="stAlert"] {
    border-radius: 0 !important;
    border-left-width: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.02em !important;
}
.stSuccess {
    background: #F0F7F4 !important;
    border-left: 3px solid #2D8C6A !important;
    color: var(--strong) !important;
}
.stError {
    background: var(--accent-soft) !important;
    border-left: 3px solid var(--accent) !important;
    color: var(--strong) !important;
}
.stWarning {
    background: #FBF6EC !important;
    border-left: 3px solid #C9860A !important;
    color: var(--strong) !important;
}
.stInfo {
    background: var(--off-white) !important;
    border-left: 3px solid var(--muted) !important;
    color: var(--strong) !important;
}

/* ---------- DATAFRAMES ---------- */
.stDataFrame { border: 1px solid var(--rule) !important; border-radius: 0 !important; }
.stDataFrame thead th {
    background: var(--off-white) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--rule) !important;
}
.stDataFrame tbody td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
}

/* ---------- CODE / JSON ---------- */
code, pre, .stJson {
    background: var(--mono-bg) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    border: 1px solid var(--rule) !important;
    border-radius: 0 !important;
    color: var(--strong) !important;
}

/* ---------- FILE UPLOADER ---------- */
div[data-testid="stFileUploader"] {
    background: var(--off-white) !important;
    border: 1.5px dashed var(--muted) !important;
    border-radius: 0 !important;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--strong) !important; }

/* ---------- DIVIDER ---------- */
hr {
    border: none !important;
    border-top: 1px solid var(--rule) !important;
    margin: 2rem 0 !important;
}

/* ---------- CAPTION ---------- */
.stCaption, small {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.67rem !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
}

/* ---------- SPINNER ---------- */
.stSpinner > div { border-top-color: var(--strong) !important; }

/* ---------- CUSTOM COMPONENTS ---------- */
.page-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.page-title {
    font-family: 'Fraunces', serif;
    font-size: 2.1rem;
    font-weight: 300;
    color: var(--strong);
    letter-spacing: -0.025em;
    line-height: 1.1;
    margin-bottom: 0.35rem;
}
.page-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: var(--body);
    line-height: 1.5;
    margin-bottom: 1.8rem;
}

.section-rule {
    border: none;
    border-top: 1px solid var(--rule);
    margin: 1.8rem 0 1.2rem 0;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--muted);
    display: block;
    margin-bottom: 0.8rem;
}

.risk-pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    font-weight: 400;
}
.risk-low    { background: #EAF5F0; color: #2D8C6A; }
.risk-medium { background: #FBF6EC; color: #C9860A; }
.risk-high   { background: var(--accent-soft); color: var(--accent); }

.data-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--body);
    padding: 5px 0;
    border-bottom: 1px solid var(--rule);
    display: flex;
    justify-content: space-between;
}
.data-row span:first-child { color: var(--muted); }

.sidebar-wordmark {
    font-family: 'Fraunces', serif;
    font-size: 1.25rem;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: var(--strong) !important;
    -webkit-text-fill-color: var(--strong) !important;
}
.sidebar-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted) !important;
    -webkit-text-fill-color: var(--muted) !important;
}
.verdict-major {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent);
    font-weight: 400;
}
.verdict-minor {
    background: #EAF5F0;
    border-left: 3px solid #2D8C6A;
    padding: 14px 18px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #2D8C6A;
    font-weight: 400;
}

.stat-block {
    border-top: 2px solid var(--strong);
    padding: 14px 0 10px 0;
}
.stat-block .stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 4px;
}
.stat-block .stat-value {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem;
    font-weight: 300;
    color: var(--strong);
    letter-spacing: -0.02em;
    line-height: 1;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# PLOTLY THEME — clean, light
# ================================================================
PT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#F9F8F6',
    font=dict(family='IBM Plex Mono, monospace', color='#9C9590', size=10),
    xaxis=dict(gridcolor='#E5E2DC', linecolor='#E5E2DC', zeroline=False,
               tickfont=dict(color='#9C9590', size=9)),
    yaxis=dict(gridcolor='#E5E2DC', linecolor='#E5E2DC', zeroline=False,
               tickfont=dict(color='#9C9590', size=9)),
    title=dict(font=dict(family='Fraunces, serif', color='#1A1714', size=13), x=0),
    margin=dict(t=48, b=36, l=56, r=24),
)

# ================================================================
# MODEL
# ================================================================
@st.cache_resource
def load_model():
    model = joblib.load('aerie_model.pkl')
    scaler = joblib.load('aerie_scaler.pkl')
    with open('feature_list.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, feature_list = load_model()
    st.sidebar.markdown(
        '<span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        'letter-spacing:0.1em;text-transform:uppercase;color:#2D8C6A;">Model Active</span>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

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
    scaled = scaler.transform(df)
    return model.predict(scaled)[0], model.predict_proba(scaled)[0][1]

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

def risk_pill(p):
    if p < 0.3:
        return '<span class="risk-pill risk-low">Low</span>'
    elif p < 0.7:
        return '<span class="risk-pill risk-medium">Medium</span>'
    return '<span class="risk-pill risk-high">High</span>'

def gauge_chart(proba):
    color = "#2D8C6A" if proba < 0.3 else ("#C9860A" if proba < 0.7 else "#C0392B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 44, "family": "Fraunces, serif", "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#C4BFB8",
                     "tickfont": {"color": "#C4BFB8", "size": 9, "family": "IBM Plex Mono, monospace"},
                     "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.18},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30],  "color": "#EAF5F0"},
                {"range": [30, 70], "color": "#FBF6EC"},
                {"range": [70, 100],"color": "#F5EAE9"},
            ],
        },
        title={"text": "PROBABILITY OF MAJOR EVENT",
               "font": {"size": 9, "family": "IBM Plex Mono, monospace", "color": "#C4BFB8"}},
    ))
    fig.update_layout(height=240, paper_bgcolor='rgba(0,0,0,0)',
                      margin=dict(t=40, b=0, l=20, r=20))
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

    _, current_p = predict_single(base_dict)
    current_val = base_dict[sweep_feature]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vals, y=probas, mode='lines',
        line=dict(color='#1A1714', width=1.8),
        fill='tozeroy', fillcolor='rgba(26,23,20,0.05)',
        name='Risk'
    ))
    fig.add_trace(go.Scatter(
        x=[current_val], y=[current_p * 100],
        mode='markers',
        marker=dict(size=9, color='#C0392B', symbol='circle',
                    line=dict(color='#FFFFFF', width=2)),
        name='Current'
    ))
    fig.add_vline(x=current_val, line_dash="dot", line_color="#C4BFB8", line_width=1,
                  annotation_text=f"{current_val:{meta['fmt']}}",
                  annotation_font=dict(color="#9C9590", size=9,
                                       family="IBM Plex Mono, monospace"),
                  annotation_position="top")
    fig.add_hline(y=50, line_dash="dot", line_color="#C4BFB8", line_width=1,
                  annotation_text="50%",
                  annotation_font=dict(color="#9C9590", size=9,
                                       family="IBM Plex Mono, monospace"),
                  annotation_position="right")
    fig.update_layout(
        **PT,
        title=f"Sensitivity — {meta['label']}",
        xaxis_title=meta['label'],
        yaxis_title="Risk (%)",
        yaxis=dict(range=[0, 108], gridcolor='#E5E2DC'),
        showlegend=False,
        height=320,
    )
    return fig

def get_available_gemini_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        usable = [
            m["name"].replace("models/", "")
            for m in r.json().get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "gemini" in m.get("name", "").lower()
            and "vision" not in m.get("name", "").lower()
            and "embedding" not in m.get("name", "").lower()
        ]
        usable.sort(key=lambda x: (0 if "flash" in x else 1, x))
        return usable or ["gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash"]
    except Exception:
        return ["gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash"]

def call_gemini_api(prompt, api_key):
    last_error = None
    for model_name in get_available_gemini_models(api_key):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            r = requests.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
            }, timeout=60)
            if r.status_code in (429, 404, 403):
                last_error = f"{r.status_code} on {model_name}"
                continue
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"], model_name
        except Exception as e:
            last_error = str(e)
    raise Exception(f"No working Gemini model found. Last error: {last_error}")

def parse_csv_block(text):
    m = re.search(r'```(?:csv)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for i, line in enumerate(text.splitlines()):
        if 'severity' in line.lower() and ',' in line:
            return "\n".join(text.splitlines()[i:])
    return text.strip()

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.markdown("""
<div style="padding: 6px 0 22px 0;">
    <div class="sidebar-wordmark">Aerie</div>
    <div class="sidebar-tagline">Adaptive Enterprise Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "Single Prediction",
    "Batch Upload",
    "Scenario Simulator",
    "Model Info",
    "AI Scenario Generator"
])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span style="font-family:IBM Plex Mono,monospace;font-size:0.62rem;'
    'letter-spacing:0.12em;text-transform:uppercase;color:#C4BFB8;">Features</span>',
    unsafe_allow_html=True
)
for i, f in enumerate(feature_list):
    st.sidebar.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        f'color:#C4BFB8;padding:2px 0;">{i+1:02d} &nbsp; {f}</div>',
        unsafe_allow_html=True
    )

# ================================================================
# SINGLE PREDICTION
# ================================================================
if page == "Single Prediction":
    st.markdown('<div class="page-eyebrow">Incident Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Single Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Enter incident parameters to forecast the probability of a major event.</div>',
        unsafe_allow_html=True
    )

    with st.form("pred_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            severity = st.slider("Severity (1–5)", 1, 5, 3)
            downtime = st.number_input("Downtime (hours)", 0.0, 100.0, 5.0)
            financial_impact = st.number_input("Financial Impact ($)", 0, 500000, 50000, step=5000)
            regulatory_flag = st.selectbox("Regulatory Flag", [0, 1])
        with c2:
            data_sensitivity = st.slider("Data Sensitivity (0–1)", 0.0, 1.0, 0.5)
            criticality = st.slider("Criticality (1–5)", 1, 5, 3)
            asset_incident_prev_count = st.number_input("Prior Incidents on Asset", 0, 20, 0)
            days_since_audit = st.number_input("Days Since Last Audit", 0, 365, 30)
        with c3:
            sxd = severity * data_sensitivity
            st.markdown('<span class="section-label">Computed</span>', unsafe_allow_html=True)
            st.metric("Severity x Sensitivity", f"{sxd:.2f}")
            st.markdown('<span class="section-label" style="margin-top:1rem;display:block;">Input Vector</span>', unsafe_allow_html=True)
            st.json({
                "severity": severity, "downtime": downtime,
                "financial_impact": financial_impact, "regulatory_flag": regulatory_flag,
                "data_sensitivity": data_sensitivity, "criticality": criticality,
                "sev_x_sens": round(sxd, 3),
                "prior_incidents": asset_incident_prev_count,
                "days_since_audit": days_since_audit
            })
        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    if submitted:
        d = {
            'severity': severity, 'downtime': downtime, 'financial_impact': financial_impact,
            'regulatory_flag': regulatory_flag, 'data_sensitivity': data_sensitivity,
            'criticality': criticality, 'severity_x_data_sensitivity': sxd,
            'asset_incident_prev_count': asset_incident_prev_count,
            'days_since_audit': days_since_audit
        }
        pred, proba = predict_single(d)
        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        v_col, g_col, _ = st.columns([1, 1.4, 0.6])
        with v_col:
            st.markdown('<span class="section-label">Verdict</span>', unsafe_allow_html=True)
            if pred == 1:
                st.markdown('<div class="verdict-major">Major Event</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict-minor">Minor / Routine</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-block">
                <div class="stat-label">Probability</div>
                <div class="stat-value">{proba:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:10px;'>{risk_pill(proba)}</div>", unsafe_allow_html=True)
        with g_col:
            st.plotly_chart(gauge_chart(proba), use_container_width=True)

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown('<span class="section-label">Feature Importance</span>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            'Feature': feature_list,
            'Importance': model.feature_importances_
        }).sort_values('Importance')
        fig = go.Figure(go.Bar(
            x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
            marker=dict(color='#1A1714', opacity=[0.25 + 0.75 * v for v in
                        (imp_df['Importance'] / imp_df['Importance'].max()).tolist()])
        ))
        fig.update_layout(**PT, height=280, showlegend=False,
                          xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# BATCH UPLOAD
# ================================================================
elif page == "Batch Upload":
    st.markdown('<div class="page-eyebrow">Bulk Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Batch Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Upload a CSV to score multiple incidents in a single pass.</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "Download Template CSV",
        pd.DataFrame(columns=feature_list).to_csv(index=False),
        "aerie_template.csv", "text/csv"
    )
    uploaded = st.file_uploader("", type="csv", label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown('<span class="section-label">Preview</span>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("Score All Incidents", use_container_width=True):
            with st.spinner("Scoring…"):
                results, err = predict_batch(df)
            if err:
                st.error(err)
            else:
                out = pd.concat([df, results], axis=1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Major Events", int(results['predicted_major_event'].sum()))
                m2.metric("Avg Probability", f"{results['probability'].mean():.1%}")
                m3.metric("High Risk  >70%", int((results['probability'] > 0.7).sum()))
                st.markdown('<span class="section-label" style="margin-top:1.5rem;display:block;">Results</span>', unsafe_allow_html=True)
                st.dataframe(out, use_container_width=True)

                fig = go.Figure(go.Histogram(
                    x=results['probability'], nbinsx=20,
                    marker=dict(color='#1A1714', opacity=0.7, line=dict(color='#F9F8F6', width=1))
                ))
                fig.add_vline(x=0.5, line_dash="dot", line_color="#C0392B", line_width=1.5,
                              annotation_text="Decision boundary",
                              annotation_font=dict(color="#C0392B", size=9,
                                                   family="IBM Plex Mono, monospace"))
                fig.update_layout(**PT, title="Risk Probability Distribution",
                                  xaxis_title="Probability", yaxis_title="Count")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("Download Results", out.to_csv(index=False),
                                   "aerie_predictions.csv", "text/csv")

# ================================================================
# SCENARIO SIMULATOR
# ================================================================
elif page == "Scenario Simulator":
    st.markdown('<div class="page-eyebrow">What-If Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Adjust parameters and sweep any variable to see how it shifts predicted risk.</div>',
        unsafe_allow_html=True
    )

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

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    g_col, v_col = st.columns([1.4, 1])
    with g_col:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)
    with v_col:
        st.markdown('<span class="section-label">Assessment</span>', unsafe_allow_html=True)
        if pred == 1:
            st.markdown('<div class="verdict-major">Major Event</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-minor">Minor / Routine</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-block" style="margin-top:14px;">
            <div class="stat-label">Probability</div>
            <div class="stat-value">{proba:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:10px;margin-bottom:16px;'>{risk_pill(proba)}</div>",
                    unsafe_allow_html=True)
        st.markdown('<span class="section-label">Parameters</span>', unsafe_allow_html=True)
        for k, v in [("Severity", severity), ("Downtime", f"{downtime}h"),
                     ("Financial", f"${financial:,}"), ("Criticality", crit),
                     ("Prior incidents", prev_count), ("Days since audit", audit_days)]:
            st.markdown(
                f'<div class="data-row"><span>{k}</span><span>{v}</span></div>',
                unsafe_allow_html=True
            )

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Sensitivity Analysis</span>', unsafe_allow_html=True)
    sweep_feat = st.selectbox(
        "",
        [f for f in feature_list if f != 'severity_x_data_sensitivity'],
        format_func=lambda x: FEATURE_META[x]["label"],
        label_visibility="collapsed"
    )
    with st.spinner("Computing…"):
        st.plotly_chart(sweep_chart(base, sweep_feat), use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Worst-Case Delta</span>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-family:IBM Plex Sans,sans-serif;font-size:0.82rem;font-weight:300;'
        'color:#5C5750;margin-bottom:14px;">Risk change if each feature is pushed to its maximum independently.</p>',
        unsafe_allow_html=True
    )
    deltas = {}
    for feat in feature_list:
        if feat == 'severity_x_data_sensitivity':
            continue
        w = base.copy()
        w[feat] = FEATURE_META[feat]["max"]
        w['severity_x_data_sensitivity'] = w['severity'] * w['data_sensitivity']
        _, wp = predict_single(w)
        deltas[FEATURE_META[feat]["label"]] = round((wp - proba) * 100, 2)

    ddf = pd.DataFrame(list(deltas.items()), columns=["Feature", "Delta"]).sort_values("Delta")
    fig2 = go.Figure(go.Bar(
        x=ddf["Delta"], y=ddf["Feature"], orientation='h',
        marker=dict(
            color=['#C0392B' if v > 0 else '#2D8C6A' for v in ddf["Delta"]],
            opacity=0.75
        )
    ))
    fig2.update_layout(
        **PT,
        title="Percentage-point change in risk (feature at maximum)",
        xaxis_title="pp change",
        height=300,
        xaxis=dict(zeroline=True, zerolinecolor='#C4BFB8', zerolinewidth=1,
                   gridcolor='#E5E2DC'),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# MODEL INFO
# ================================================================
elif page == "Model Info":
    st.markdown('<div class="page-eyebrow">Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model Information</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Random Forest classifier — architecture details, feature weights, and usage notes.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "Random Forest")
    c2.metric("Trees", model.n_estimators)
    c3.metric("Features", len(feature_list))

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Feature Importances</span>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': model.feature_importances_
    }).sort_values('Importance')

    fig = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker=dict(
            color='#1A1714',
            opacity=[0.2 + 0.8 * v for v in
                     (imp_df['Importance'] / imp_df['Importance'].max()).tolist()]
        )
    ))
    fig.update_layout(**PT, height=300, xaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Pages</span>', unsafe_allow_html=True)
    for name, desc in [
        ("Single Prediction",    "Score one incident by entering its parameters manually."),
        ("Batch Upload",         "Score multiple incidents from a CSV file in one pass."),
        ("Scenario Simulator",   "Sweep any variable to watch the risk curve update live."),
        ("AI Scenario Generator","Use Gemini to write structured scenarios, auto-scored by AERIE."),
    ]:
        st.markdown(
            f'<div style="padding:12px 0;border-bottom:1px solid #E5E2DC;">'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;'
            f'color:#1A1714;">{name}</span>'
            f'<div style="font-family:IBM Plex Sans,sans-serif;font-size:0.8rem;'
            f'font-weight:300;color:#9C9590;margin-top:3px;">{desc}</div></div>',
            unsafe_allow_html=True
        )

# ================================================================
# AI SCENARIO GENERATOR
# ================================================================
elif page == "AI Scenario Generator":
    st.markdown('<div class="page-eyebrow">Generative Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">AI Scenario Generator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Describe the incident context. Gemini generates structured scenarios — AERIE scores each one automatically.</div>',
        unsafe_allow_html=True
    )

    try:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
        st.sidebar.markdown(
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
            'letter-spacing:0.1em;text-transform:uppercase;color:#2D8C6A;">Gemini Connected</span>',
            unsafe_allow_html=True
        )
    except:
        GEMINI_KEY = st.text_input(
            "Google Gemini API Key", type="password",
            help="Free key at aistudio.google.com/app/apikey"
        )

    if not GEMINI_KEY:
        st.markdown(
            '<div style="background:#F9F8F6;border-left:3px solid #C4BFB8;padding:14px 18px;">'
            '<span style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#5C5750;">'
            'Add your Gemini API key above, or store it in Streamlit secrets as '
            '<code>GEMINI_API_KEY</code>. Get a free key at aistudio.google.com/app/apikey</span></div>',
            unsafe_allow_html=True
        )
        st.stop()

    a1, a2 = st.columns(2)
    with a1:
        industry = st.selectbox("Industry / Context", [
            "Financial Services", "Healthcare", "Manufacturing",
            "Government", "Retail", "Energy & Utilities"
        ])
        n_scenarios = st.slider("Number of Scenarios", 3, 10, 5)
    with a2:
        threat = st.selectbox("Threat Focus", [
            "Mixed / Varied", "Cybersecurity", "Operational Failures",
            "Data Breaches", "Regulatory Incidents", "Third-party / Supply Chain"
        ])
        sev_bias = st.selectbox("Severity Bias", [
            "Realistic mix", "Mostly high-severity", "Mostly low-severity"
        ])
    extra = st.text_area(
        "Additional Context",
        placeholder="E.g. 'Focus on cloud infrastructure' or 'servers affected by flood'",
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
        with st.spinner("Generating scenarios…"):
            try:
                text, used_model = call_gemini_api(prompt, GEMINI_KEY)
                st.caption(f"model · {used_model}")

                csv_text = parse_csv_block(text)
                try:
                    scenarios_df = pd.read_csv(io.StringIO(csv_text))
                except Exception:
                    st.error("Could not parse structured CSV from the AI output.")
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

                    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
                    st.markdown('<span class="section-label">Scored Scenarios</span>', unsafe_allow_html=True)

                    def highlight(row):
                        p = row['probability']
                        c = ('rgba(192,57,43,0.06)' if p >= 0.7
                             else 'rgba(201,134,10,0.06)' if p >= 0.3
                             else 'rgba(45,140,106,0.06)')
                        return [f'background-color: {c}'] * len(row)

                    disp = [c for c in [
                        'description', 'severity', 'downtime', 'financial_impact',
                        'regulatory_flag', 'data_sensitivity', 'criticality',
                        'probability', 'predicted_major_event'
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

                    colors = ['#C0392B' if p >= 0.7 else ('#C9860A' if p >= 0.3 else '#2D8C6A')
                              for p in out['probability']]
                    fig = go.Figure(go.Bar(
                        x=list(range(len(out))),
                        y=out['probability'],
                        marker=dict(color=colors, opacity=0.7,
                                    line=dict(color='#F9F8F6', width=1)),
                        text=[f"{p:.0%}" for p in out['probability']],
                        textposition='outside',
                        textfont=dict(family='IBM Plex Mono, monospace',
                                      size=9, color='#9C9590')
                    ))
                    fig.update_layout(
                        **PT,
                        title="Risk Probability by Scenario",
                        xaxis_title="Scenario",
                        yaxis=dict(range=[0, 1.2], tickformat='.0%',
                                   gridcolor='#E5E2DC'),
                        showlegend=False,
                        height=300
                    )
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
