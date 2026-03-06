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

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="AERIE Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# GLOBAL CSS — Dark Command Center Theme
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

/* ---- ROOT VARIABLES ---- */
:root {
    --bg-primary:   #080C14;
    --bg-card:      #0E1623;
    --bg-elevated:  #141E2E;
    --border:       #1E2D42;
    --border-glow:  #1A4A7A;
    --accent-blue:  #0EA5E9;
    --accent-cyan:  #06B6D4;
    --accent-amber: #F59E0B;
    --accent-red:   #EF4444;
    --accent-green: #10B981;
    --text-primary: #E8F0FE;
    --text-muted:   #6B8CAE;
    --text-dim:     #3D5470;
}

/* ---- GLOBAL RESETS ---- */
html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Subtle grid background on main area */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(14,165,233,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(14,165,233,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1220 0%, #060D18 100%) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em;
    color: var(--text-muted) !important;
    padding: 6px 10px !important;
    border-radius: 6px;
    transition: all 0.2s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(14,165,233,0.08) !important;
    color: var(--accent-blue) !important;
}

/* ---- HEADERS & TITLES ---- */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.1rem !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 1.2rem !important;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em !important;
}
h4, h5, h6 { font-family: 'Inter', sans-serif !important; }

p, li, label, .stMarkdown {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-muted) !important;
    font-size: 0.88rem !important;
}

/* ---- METRIC CARDS ---- */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover {
    border-color: var(--border-glow) !important;
}
div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
div[data-testid="metric-container"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    color: var(--text-primary) !important;
}

/* ---- BUTTONS ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 0.55rem 1.4rem !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(14,165,233,0.4) !important;
}

/* ---- INPUTS / SLIDERS / SELECTS ---- */
.stSlider > div > div > div {
    background: var(--accent-blue) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-cyan) !important;
    border: 2px solid var(--bg-primary) !important;
    box-shadow: 0 0 8px rgba(6,182,212,0.6) !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
textarea {
    background: var(--bg-elevated) !important;
    border-color: var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(14,165,233,0.15) !important;
}
input[type="number"] {
    background: var(--bg-elevated) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Mono', monospace !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* ---- DATAFRAMES ---- */
.stDataFrame, iframe[title="st_aggrid"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
.stDataFrame thead th {
    background: var(--bg-elevated) !important;
    color: var(--accent-blue) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid var(--border-glow) !important;
}
.stDataFrame tbody tr:hover td { background: rgba(14,165,233,0.05) !important; }

/* ---- ALERTS / BANNERS ---- */
.stAlert {
    border-radius: 10px !important;
    border-left-width: 3px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
}
div[data-baseweb="notification"][kind="negative"],
.stAlert[data-baseweb="notification"] {
    background: rgba(239,68,68,0.08) !important;
    border-left-color: var(--accent-red) !important;
}

/* ---- SUCCESS / ERROR / WARNING / INFO ---- */
.stSuccess { background: rgba(16,185,129,0.08) !important; border-left: 3px solid var(--accent-green) !important; border-radius: 10px !important; }
.stError   { background: rgba(239,68,68,0.08) !important;  border-left: 3px solid var(--accent-red)   !important; border-radius: 10px !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border-left: 3px solid var(--accent-amber) !important; border-radius: 10px !important; }
.stInfo    { background: rgba(14,165,233,0.08) !important; border-left: 3px solid var(--accent-blue)  !important; border-radius: 10px !important; }

/* ---- FORM CONTAINER ---- */
div[data-testid="stForm"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 24px !important;
}

/* ---- EXPANDER ---- */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ---- SPINNER ---- */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* ---- HORIZONTAL DIVIDER ---- */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ---- CAPTION / SMALL TEXT ---- */
.stCaption, small { 
    font-family: 'DM Mono', monospace !important; 
    color: var(--text-dim) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.05em !important;
}

/* ---- JSON DISPLAY ---- */
.stJson {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ---- CODE BLOCKS ---- */
code, pre {
    background: var(--bg-elevated) !important;
    color: var(--accent-cyan) !important;
    font-family: 'DM Mono', monospace !important;
    border-radius: 6px !important;
    border: 1px solid var(--border) !important;
}

/* ---- FILE UPLOADER ---- */
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-glow) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent-blue) !important; }

/* ---- PLOTLY CHARTS — force dark bg ---- */
.js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* ---- SIDEBAR LOGO AREA ---- */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #0EA5E9, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.sidebar-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #3D5470 !important;
    text-transform: uppercase;
    -webkit-text-fill-color: #3D5470;
}

/* ---- PAGE TITLE ACCENT ---- */
.page-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 1.5rem;
}
.page-icon {
    font-size: 1.8rem;
    line-height: 1;
}
.page-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #E8F0FE;
    letter-spacing: -0.02em;
    line-height: 1;
}
.page-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    color: #6B8CAE;
    margin-top: 4px;
}

/* ---- SECTION LABEL ---- */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 0.6rem;
    display: block;
}

/* ---- STAT BADGE ---- */
.stat-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
}
.badge-blue  { background: rgba(14,165,233,0.12); color: #0EA5E9; border: 1px solid rgba(14,165,233,0.25); }
.badge-green { background: rgba(16,185,129,0.12); color: #10B981; border: 1px solid rgba(16,185,129,0.25); }
.badge-amber { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.25); }
.badge-red   { background: rgba(239,68,68,0.12);  color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }

/* ---- INFO CARD ---- */
.info-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
    position: relative;
}
.info-card-accent {
    border-left: 3px solid var(--accent-blue);
}

/* ---- NUMBER INPUT LABEL ---- */
.stNumberInput label, .stSlider label, .stSelectbox label, .stTextInput label, .stTextArea label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-muted) !important;
}

/* ---- SIDEBAR TEXT ---- */
.stSidebarContent .stText {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
}

/* ---- TABS (if used) ---- */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    gap: 4px !important;
    padding: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-elevated) !important;
    color: var(--accent-blue) !important;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# PLOTLY DARK THEME — applied globally
# ================================================================
PLOTLY_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(14,22,38,0.6)',
    font=dict(family='DM Mono, monospace', color='#6B8CAE', size=11),
    xaxis=dict(gridcolor='#1E2D42', linecolor='#1E2D42', tickfont=dict(color='#6B8CAE')),
    yaxis=dict(gridcolor='#1E2D42', linecolor='#1E2D42', tickfont=dict(color='#6B8CAE')),
    title=dict(font=dict(family='Syne, sans-serif', color='#E8F0FE', size=14)),
    margin=dict(t=50, b=40, l=60, r=30),
)

# ================================================================
# LOAD MODEL
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
    st.sidebar.markdown('<span class="stat-badge badge-green">● MODEL ONLINE</span>', unsafe_allow_html=True)
except Exception as e:
    st.sidebar.error(f"❌ {e}")
    st.stop()

# ================================================================
# FEATURE META
# ================================================================
FEATURE_META = {
    'severity':                   {"label": "Severity",               "min": 1,   "max": 5,      "default": 3,     "step": 1,    "fmt": ".0f"},
    'downtime':                   {"label": "Downtime (hrs)",         "min": 0.0, "max": 100.0,  "default": 5.0,   "step": 1.0,  "fmt": ".1f"},
    'financial_impact':           {"label": "Financial Impact ($)",   "min": 0,   "max": 500000, "default": 50000, "step": 5000, "fmt": ",.0f"},
    'regulatory_flag':            {"label": "Regulatory Flag",        "min": 0,   "max": 1,      "default": 0,     "step": 1,    "fmt": ".0f"},
    'data_sensitivity':           {"label": "Data Sensitivity",       "min": 0.0, "max": 1.0,    "default": 0.5,   "step": 0.05, "fmt": ".2f"},
    'criticality':                {"label": "Criticality",            "min": 1,   "max": 5,      "default": 3,     "step": 1,    "fmt": ".0f"},
    'severity_x_data_sensitivity':{"label": "Severity × Sensitivity", "min": 0.0, "max": 5.0,    "default": 1.5,   "step": 0.1,  "fmt": ".2f"},
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

def risk_label(p):
    if p < 0.3:
        return '<span class="stat-badge badge-green">LOW RISK</span>'
    elif p < 0.7:
        return '<span class="stat-badge badge-amber">MEDIUM RISK</span>'
    return '<span class="stat-badge badge-red">HIGH RISK</span>'

def gauge_chart(proba):
    color = "#10B981" if proba < 0.3 else ("#F59E0B" if proba < 0.7 else "#EF4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 42, "family": "Syne, sans-serif", "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3D5470", "tickfont": {"color": "#3D5470", "size": 10}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,30],  "color": "rgba(16,185,129,0.12)"},
                {"range": [30,70], "color": "rgba(245,158,11,0.10)"},
                {"range": [70,100],"color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": proba*100},
        },
        title={"text": "MAJOR EVENT PROBABILITY", "font": {"size": 10, "family": "DM Mono, monospace", "color": "#3D5470"}},
        domain={"x": [0,1], "y": [0,1]}
    ))
    fig.update_layout(height=260, **PLOTLY_THEME)
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

    fig = go.Figure()
    # Area fill
    fig.add_trace(go.Scatter(
        x=vals, y=probas, mode='lines',
        line=dict(color='#0EA5E9', width=2.5),
        fill='tozeroy',
        fillgradient=dict(type="vertical", colorscale=[[0,"rgba(14,165,233,0.25)"],[1,"rgba(14,165,233,0)"]]),
        name="Risk %"
    ))
    # Current position marker
    current_val = base_dict[sweep_feature]
    _, current_p = predict_single(base_dict)
    fig.add_trace(go.Scatter(
        x=[current_val], y=[current_p * 100],
        mode='markers',
        marker=dict(size=12, color='#F59E0B', symbol='circle',
                    line=dict(color='#080C14', width=2)),
        name="Current"
    ))
    fig.add_vline(x=current_val, line_dash="dash", line_color="#F59E0B", line_width=1.5,
        annotation_text=f"{current_val:{meta['fmt']}}", annotation_font_color="#F59E0B",
        annotation_font_size=10)
    fig.add_hline(y=50, line_dash="dot", line_color="#3D5470", line_width=1,
        annotation_text="50%", annotation_font_color="#3D5470", annotation_font_size=9)

    fig.update_layout(
        title=f"Sensitivity — {meta['label']}",
        xaxis_title=meta['label'],
        yaxis_title="Risk Probability (%)",
        yaxis=dict(range=[0, 108], gridcolor='#1E2D42'),
        showlegend=False,
        height=340,
        **PLOTLY_THEME
    )
    return fig

def get_available_gemini_models(api_key):
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        all_models = r.json().get("models", [])
        usable = [
            m["name"].replace("models/", "")
            for m in all_models
            if "generateContent" in m.get("supportedGenerationMethods", [])
            and "gemini" in m.get("name", "").lower()
            and "vision" not in m.get("name", "").lower()
            and "embedding" not in m.get("name", "").lower()
        ]
        usable.sort(key=lambda x: (0 if "flash" in x else 1, x))
        return usable if usable else ["gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash"]
    except Exception:
        return ["gemini-2.0-flash", "gemini-1.5-flash-8b", "gemini-1.5-flash"]

def call_gemini_api(prompt, api_key):
    models = get_available_gemini_models(api_key)
    last_error = None
    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1024}
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            if r.status_code in (429, 404, 403):
                last_error = f"{r.status_code} on {model_name}"
                continue
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"], model_name
        except Exception as e:
            last_error = str(e)
            continue
    raise Exception(f"No working Gemini model found.\nLast error: {last_error}\n\nIf you just rotated your key, wait 60 seconds and try again.")

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
<div style="padding: 8px 0 20px 0;">
    <div class="sidebar-logo">🛡️ AERIE</div>
    <div class="sidebar-sub">Adaptive Enterprise Risk Intelligence</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

page = st.sidebar.radio("", [
    "🔍 Single Prediction",
    "📤 Batch Upload",
    "🎮 Scenario Simulator",
    "📊 Model Info",
    "🤖 AI Scenario Generator"
])

st.sidebar.markdown("---")
st.sidebar.markdown('<span class="section-label">Feature Index</span>', unsafe_allow_html=True)
for i, f in enumerate(feature_list):
    st.sidebar.markdown(
        f'<div style="font-family:DM Mono,monospace;font-size:0.68rem;color:#3D5470;padding:2px 0;">'
        f'<span style="color:#1A4A7A;">{i+1:02d}</span> {f}</div>',
        unsafe_allow_html=True
    )

# ================================================================
# SINGLE PREDICTION
# ================================================================
if page == "🔍 Single Prediction":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🔍</div>
        <div>
            <div class="page-title">Single Incident Predictor</div>
            <div class="page-subtitle">Enter incident parameters to forecast major event probability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown('<span class="section-label">Auto-Calculated</span>', unsafe_allow_html=True)
            st.metric("Severity × Sensitivity", f"{sxd:.2f}")
            st.markdown('<span class="section-label" style="margin-top:12px;">Input Vector</span>', unsafe_allow_html=True)
            st.json({
                "severity": severity, "downtime": downtime,
                "financial_impact": financial_impact, "regulatory_flag": regulatory_flag,
                "data_sensitivity": data_sensitivity, "criticality": criticality,
                "sev_x_sens": round(sxd, 3), "prior_incidents": asset_incident_prev_count,
                "days_since_audit": days_since_audit
            })
        submitted = st.form_submit_button("⚡ RUN PREDICTION", use_container_width=True)

    if submitted:
        d = {
            'severity': severity, 'downtime': downtime, 'financial_impact': financial_impact,
            'regulatory_flag': regulatory_flag, 'data_sensitivity': data_sensitivity,
            'criticality': criticality, 'severity_x_data_sensitivity': sxd,
            'asset_incident_prev_count': asset_incident_prev_count, 'days_since_audit': days_since_audit
        }
        pred, proba = predict_single(d)
        st.markdown("---")

        r1, r2, r3 = st.columns([1, 1, 2])
        with r1:
            if pred == 1:
                st.error("🚨 MAJOR EVENT")
            else:
                st.success("✅ MINOR / ROUTINE")
        with r2:
            st.metric("Probability", f"{proba:.1%}")
            st.markdown(risk_label(proba), unsafe_allow_html=True)
        with r3:
            st.plotly_chart(gauge_chart(proba), use_container_width=True)

        st.markdown('<span class="section-label">Feature Contribution</span>', unsafe_allow_html=True)
        imp_df = pd.DataFrame({
            'Feature': feature_list,
            'Importance': model.feature_importances_
        }).sort_values('Importance')
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale=['#1E2D42','#0EA5E9'])
        fig.update_layout(**PLOTLY_THEME, showlegend=False,
                          coloraxis_showscale=False, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# BATCH UPLOAD
# ================================================================
elif page == "📤 Batch Upload":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">📤</div>
        <div>
            <div class="page-title">Batch Risk Scoring</div>
            <div class="page-subtitle">Upload a CSV to score multiple incidents simultaneously</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        "📥 DOWNLOAD TEMPLATE CSV",
        pd.DataFrame(columns=feature_list).to_csv(index=False),
        "aerie_template.csv", "text/csv"
    )
    uploaded = st.file_uploader("Drop your CSV here", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown('<span class="section-label">Data Preview</span>', unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("⚡ SCORE ALL INCIDENTS", use_container_width=True):
            with st.spinner("Running batch inference…"):
                results, err = predict_batch(df)
            if err:
                st.error(err)
            else:
                out = pd.concat([df, results], axis=1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Major Events", int(results['predicted_major_event'].sum()))
                m2.metric("Avg Probability", f"{results['probability'].mean():.1%}")
                m3.metric("High Risk  >70%", int((results['probability'] > 0.7).sum()))
                st.markdown('<span class="section-label">Scored Results</span>', unsafe_allow_html=True)
                st.dataframe(out, use_container_width=True)
                fig = px.histogram(results, x='probability', nbins=20,
                    title="Risk Probability Distribution",
                    color_discrete_sequence=['#0EA5E9'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="#F59E0B",
                    annotation_text="Decision boundary", annotation_font_color="#F59E0B")
                fig.update_layout(**PLOTLY_THEME)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("📥 DOWNLOAD RESULTS", out.to_csv(index=False),
                                   "aerie_predictions.csv", "text/csv")

# ================================================================
# SCENARIO SIMULATOR
# ================================================================
elif page == "🎮 Scenario Simulator":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🎮</div>
        <div>
            <div class="page-title">What-If Scenario Simulator</div>
            <div class="page-subtitle">Sweep any variable to see its effect on predicted risk in real time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

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
    gc, vc = st.columns([1.5, 1])
    with gc:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)
    with vc:
        st.markdown('<span class="section-label">Current Assessment</span>', unsafe_allow_html=True)
        if pred == 1:
            st.error("🚨 MAJOR EVENT")
        else:
            st.success("✅ MINOR / ROUTINE")
        st.metric("Probability", f"{proba:.1%}")
        st.markdown(risk_label(proba), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<span class="section-label">Scenario Snapshot</span>', unsafe_allow_html=True)
        for k, v in [("Severity", severity), ("Downtime", f"{downtime}h"),
                     ("Financial", f"${financial:,}"), ("Criticality", crit),
                     ("Prior Incidents", prev_count), ("Days Since Audit", audit_days)]:
            st.markdown(
                f'<div style="font-family:DM Mono,monospace;font-size:0.72rem;color:#3D5470;'
                f'padding:3px 0;border-bottom:1px solid #0E1623;">'
                f'<span style="color:#1A4A7A;">{k}</span> · {v}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown('<span class="section-label">Sensitivity Analysis</span>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#6B8CAE;margin-bottom:12px;">'
        'Select a variable to sweep. All others stay fixed at current slider values.</p>',
        unsafe_allow_html=True
    )
    sweep_feat = st.selectbox("Variable to sweep",
        [f for f in feature_list if f != 'severity_x_data_sensitivity'],
        format_func=lambda x: FEATURE_META[x]["label"])
    with st.spinner("Computing sensitivity curve…"):
        st.plotly_chart(sweep_chart(base, sweep_feat), use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="section-label">Worst-Case Risk Delta</span>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#6B8CAE;margin-bottom:12px;">'
        'How much does risk increase if each factor hits its maximum independently?</p>',
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

    ddf = pd.DataFrame(list(deltas.items()), columns=["Feature", "Delta (pp)"]).sort_values("Delta (pp)")
    fig2 = go.Figure(go.Bar(
        x=ddf["Delta (pp)"], y=ddf["Feature"], orientation='h',
        marker=dict(
            color=['#EF4444' if v > 0 else '#10B981' for v in ddf["Delta (pp)"]],
            opacity=0.85
        )
    ))
    fig2.update_layout(
        **PLOTLY_THEME,
        title="Risk delta (pp) if feature is maximised",
        xaxis_title="Percentage-point change",
        height=320,
        xaxis=dict(zeroline=True, zerolinecolor='#3D5470', zerolinewidth=1.5, gridcolor='#1E2D42'),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# MODEL INFO
# ================================================================
elif page == "📊 Model Info":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">📊</div>
        <div>
            <div class="page-title">Model Information</div>
            <div class="page-subtitle">Random Forest architecture, feature weights, and usage guide</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Type", "Random Forest")
    c2.metric("Decision Trees", model.n_estimators)
    c3.metric("Input Features", len(feature_list))

    st.markdown("---")
    st.markdown('<span class="section-label">Feature Importances</span>', unsafe_allow_html=True)
    imp_df = pd.DataFrame({
        'Feature': feature_list,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=imp_df['Importance'], y=imp_df['Feature'], orientation='h',
        marker=dict(
            color=imp_df['Importance'],
            colorscale=[[0,'#1E2D42'],[0.5,'#0284C7'],[1,'#06B6D4']],
            showscale=False,
        )
    ))
    fig.update_layout(**PLOTLY_THEME, height=320,
                      xaxis_title="Importance Score")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="section-label">Usage Guide</span>', unsafe_allow_html=True)
    for icon, mode, desc in [
        ("🔍","Single Prediction","Enter one incident manually and get an instant risk score"),
        ("📤","Batch Upload","Score dozens of incidents from a CSV in one pass"),
        ("🎮","Scenario Simulator","Sweep any variable and watch the risk curve update live"),
        ("🤖","AI Generator","Generate AI-written incident scenarios and auto-score them"),
    ]:
        st.markdown(
            f'<div class="info-card info-card-accent" style="display:flex;gap:14px;align-items:flex-start;">'
            f'<span style="font-size:1.3rem;">{icon}</span>'
            f'<div><strong style="font-family:DM Mono,monospace;font-size:0.75rem;color:#0EA5E9;">{mode}</strong>'
            f'<div style="font-size:0.8rem;color:#6B8CAE;margin-top:3px;">{desc}</div></div></div>',
            unsafe_allow_html=True
        )

# ================================================================
# AI SCENARIO GENERATOR
# ================================================================
elif page == "🤖 AI Scenario Generator":
    st.markdown("""
    <div class="page-header">
        <div class="page-icon">🤖</div>
        <div>
            <div class="page-title">AI Scenario Generator</div>
            <div class="page-subtitle">Gemini generates structured incident scenarios — AERIE scores them automatically</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        GEMINI_KEY = st.secrets["GEMINI_API_KEY"]
        st.sidebar.markdown('<span class="stat-badge badge-green">● GEMINI CONNECTED</span>', unsafe_allow_html=True)
    except:
        GEMINI_KEY = st.text_input("Google Gemini API Key", type="password",
            help="Free key at aistudio.google.com/app/apikey")

    if not GEMINI_KEY:
        st.markdown("""
        <div class="info-card info-card-accent">
            <strong style="font-family:DM Mono,monospace;font-size:0.75rem;color:#F59E0B;">API KEY REQUIRED</strong>
            <div style="font-size:0.8rem;color:#6B8CAE;margin-top:6px;">
            Get a free key at <code>aistudio.google.com/app/apikey</code> and paste it above — 
            or add it to Streamlit secrets as <code>GEMINI_API_KEY</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    a1, a2 = st.columns(2)
    with a1:
        industry = st.selectbox("Industry / Context", [
            "Financial Services","Healthcare","Manufacturing",
            "Government","Retail","Energy & Utilities"
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

Output ONLY a valid CSV block (no extra text, no markdown outside the CSV). Use these exact headers:
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

    if st.button("⚡ GENERATE & SCORE SCENARIOS", use_container_width=True):
        with st.spinner("Gemini is synthesising scenarios…"):
            try:
                text, used_model = call_gemini_api(prompt, GEMINI_KEY)
                st.caption(f"model: {used_model}")

                csv_text = parse_csv_block(text)
                try:
                    scenarios_df = pd.read_csv(io.StringIO(csv_text))
                except Exception:
                    st.error("Could not parse CSV from AI output.")
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
                    m3.metric("High Risk >70%", int((scored['probability'] > 0.7).sum()))

                    st.markdown("---")
                    st.markdown('<span class="section-label">Scored Scenarios</span>', unsafe_allow_html=True)

                    def highlight(row):
                        p = row['probability']
                        c = 'rgba(239,68,68,0.1)' if p >= 0.7 else ('rgba(245,158,11,0.1)' if p >= 0.3 else 'rgba(16,185,129,0.1)')
                        return [f'background-color: {c}'] * len(row)

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

                    # Chart
                    fig = go.Figure()
                    colors = ['#EF4444' if p >= 0.7 else ('#F59E0B' if p >= 0.3 else '#10B981')
                              for p in out['probability']]
                    fig.add_trace(go.Bar(
                        x=list(range(len(out))),
                        y=out['probability'],
                        marker_color=colors,
                        marker_opacity=0.85,
                        text=[f"{p:.0%}" for p in out['probability']],
                        textposition='outside',
                        textfont=dict(family='DM Mono, monospace', size=10, color='#6B8CAE')
                    ))
                    fig.update_layout(
                        **PLOTLY_THEME,
                        title="Risk Probability by Scenario (sorted highest → lowest)",
                        xaxis_title="Scenario #",
                        yaxis_title="Risk Probability",
                        yaxis=dict(range=[0, 1.2], tickformat='.0%', gridcolor='#1E2D42'),
                        height=320,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button(
                        "📥 DOWNLOAD SCORED SCENARIOS",
                        out.to_csv(index=False),
                        "aerie_ai_scenarios.csv", "text/csv"
                    )

            except requests.exceptions.Timeout:
                st.error("Request timed out. Try again in 30s.")
            except Exception as e:
                st.error(f"Error: {e}")
