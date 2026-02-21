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

# Page config
st.set_page_config(page_title="AERIE Risk Intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load('aerie_model.pkl')
    scaler = joblib.load('aerie_scaler.pkl')
    with open('feature_list.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

try:
    model, scaler, feature_list = load_model()
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error(f"❌ {e}")
    st.stop()

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
    return pd.DataFrame({'predicted_major_event': model.predict(scaled), 'probability': model.predict_proba(scaled)[:,1]}), None

def risk_label(p):
    return "🟢 Low Risk" if p < 0.3 else ("🟡 Medium Risk" if p < 0.7 else "🔴 High Risk")

def gauge_chart(proba):
    color = "green" if proba < 0.3 else ("orange" if proba < 0.7 else "red")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.3},
            "steps": [{"range": [0,30], "color": "#d4edda"}, {"range": [30,70], "color": "#fff3cd"}, {"range": [70,100], "color": "#f8d7da"}],
        },
        title={"text": "Major Event Probability", "font": {"size": 16}},
    ))
    fig.update_layout(height=280, margin=dict(t=50, b=10, l=30, r=30))
    return fig

def sweep_chart(base_dict, sweep_feature):
    meta = FEATURE_META[sweep_feature]
    vals = np.linspace(meta["min"], meta["max"], 40)
    probas = []
    for v in vals:
        d = base_dict.copy()
        d[sweep_feature] = v
        d['severity_x_data_sensitivity'] = d['severity'] * d['data_sensitivity']
        _, p = predict_single(d)
        probas.append(p * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vals, y=probas, mode='lines+markers',
        line=dict(color='royalblue', width=2.5), fill='tozeroy', fillcolor='rgba(65,105,225,0.1)'))
    fig.add_vline(x=base_dict[sweep_feature], line_dash="dash", line_color="red", line_width=2,
        annotation_text=f"Current: {base_dict[sweep_feature]:{meta['fmt']}}", annotation_position="top right")
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
        annotation_text="50% threshold", annotation_position="right")
    fig.update_layout(title=f"Risk Sensitivity to: {meta['label']}", xaxis_title=meta['label'],
        yaxis_title="Risk Probability (%)", yaxis=dict(range=[0, 105]), height=360,
        margin=dict(t=50, b=40, l=60, r=30), showlegend=False)
    return fig

def call_hf_api(prompt, token):
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    r = requests.post(url, headers={"Authorization": f"Bearer {token}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": 700, "temperature": 0.7, "return_full_text": False}}, timeout=60)
    return r.json()

def parse_csv_block(text):
    m = re.search(r'```(?:csv)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    for i, line in enumerate(text.splitlines()):
        if 'severity' in line.lower() and ',' in line:
            return "\n".join(text.splitlines()[i:])
    return text.strip()

# Sidebar
st.sidebar.title("🛡️ AERIE")
st.sidebar.markdown("**A**daptive **E**nterprise **R**isk **I**ntelligence **E**ngine")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["🔍 Single Prediction","📤 Batch Upload","🎮 Scenario Simulator","📊 Model Info","🤖 AI Scenario Generator"])
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
for i, f in enumerate(feature_list):
    st.sidebar.text(f"{i+1}. {f}")

# ================================================================
# SINGLE PREDICTION
# ================================================================
if page == "🔍 Single Prediction":
    st.title("🔍 Single Incident Risk Predictor")
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
            st.markdown("### Auto-calculated")
            sxd = severity * data_sensitivity
            st.metric("Severity × Sensitivity", f"{sxd:.2f}")
            st.json({"severity": severity, "downtime": downtime, "financial_impact": financial_impact,
                     "regulatory_flag": regulatory_flag, "data_sensitivity": data_sensitivity,
                     "criticality": criticality, "sev_x_sens": sxd,
                     "prior_incidents": asset_incident_prev_count, "days_since_audit": days_since_audit})
        submitted = st.form_submit_button("🚀 Predict Risk", use_container_width=True)

    if submitted:
        d = {'severity': severity, 'downtime': downtime, 'financial_impact': financial_impact,
             'regulatory_flag': regulatory_flag, 'data_sensitivity': data_sensitivity,
             'criticality': criticality, 'severity_x_data_sensitivity': sxd,
             'asset_incident_prev_count': asset_incident_prev_count, 'days_since_audit': days_since_audit}
        pred, proba = predict_single(d)
        st.markdown("---")
        r1, r2, r3 = st.columns([1,1,2])
        with r1:
            st.error("🚨 **MAJOR EVENT**") if pred == 1 else st.success("✅ **Minor / Routine**")
        with r2:
            st.metric("Probability", f"{proba:.1%}")
            st.markdown(risk_label(proba))
        with r3:
            st.plotly_chart(gauge_chart(proba), use_container_width=True)
        st.subheader("📊 Feature Importance")
        imp_df = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_}).sort_values('Importance')
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# BATCH UPLOAD
# ================================================================
elif page == "📤 Batch Upload":
    st.title("📤 Batch Risk Scoring")
    st.download_button("📥 Download Template CSV", pd.DataFrame(columns=feature_list).to_csv(index=False), "aerie_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("🔮 Score All"):
            results, err = predict_batch(df)
            if err:
                st.error(err)
            else:
                out = pd.concat([df, results], axis=1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Major Events", results['predicted_major_event'].sum())
                m2.metric("Avg Probability", f"{results['probability'].mean():.1%}")
                m3.metric("High Risk >70%", (results['probability'] > 0.7).sum())
                st.dataframe(out)
                fig = px.histogram(results, x='probability', nbins=20, title="Risk Distribution", color_discrete_sequence=['royalblue'])
                fig.add_vline(x=0.5, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button("📥 Download Results", out.to_csv(index=False), "aerie_predictions.csv", "text/csv")

# ================================================================
# SCENARIO SIMULATOR (rebuilt)
# ================================================================
elif page == "🎮 Scenario Simulator":
    st.title("🎮 What-If Scenario Simulator")
    st.markdown("Adjust sliders to model a scenario. The **sensitivity chart** shows how changing any single factor shifts predicted risk while holding everything else constant.")

    c1, c2 = st.columns(2)
    with c1:
        severity    = st.slider("Severity", 1, 5, 3)
        downtime    = st.slider("Downtime (hrs)", 0.0, 100.0, 5.0)
        financial   = st.slider("Financial Impact ($)", 0, 500000, 50000, step=5000)
        reg_flag    = st.selectbox("Regulatory Flag", [0, 1])
    with c2:
        data_sens   = st.slider("Data Sensitivity", 0.0, 1.0, 0.5)
        crit        = st.slider("Criticality", 1, 5, 3)
        prev_count  = st.slider("Prior Incidents", 0, 20, 0)
        audit_days  = st.slider("Days Since Audit", 0, 365, 30)

    base = {
        'severity': severity, 'downtime': downtime, 'financial_impact': financial,
        'regulatory_flag': reg_flag, 'data_sensitivity': data_sens, 'criticality': crit,
        'severity_x_data_sensitivity': severity * data_sens,
        'asset_incident_prev_count': prev_count, 'days_since_audit': audit_days,
    }
    pred, proba = predict_single(base)

    st.markdown("---")
    gc, vc = st.columns([1.4, 1])
    with gc:
        st.plotly_chart(gauge_chart(proba), use_container_width=True)
    with vc:
        st.markdown("### Prediction")
        st.error("🚨 **MAJOR EVENT**") if pred == 1 else st.success("✅ **Minor / Routine**")
        st.markdown(f"**Probability:** {proba:.1%}")
        st.markdown(risk_label(proba))

    st.markdown("---")
    st.subheader("📈 Sensitivity Analysis")
    st.markdown("Pick a variable to sweep across its full range — all other factors stay fixed at your current slider values.")
    sweep_feat = st.selectbox("Variable to sweep",
        [f for f in feature_list if f != 'severity_x_data_sensitivity'],
        format_func=lambda x: FEATURE_META[x]["label"])
    with st.spinner("Calculating..."):
        st.plotly_chart(sweep_chart(base, sweep_feat), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Worst-Case Risk Delta — What Happens If Each Factor Hits Its Maximum?")
    deltas = {}
    for feat in feature_list:
        if feat == 'severity_x_data_sensitivity':
            continue
        w = base.copy()
        w[feat] = FEATURE_META[feat]["max"]
        w['severity_x_data_sensitivity'] = w['severity'] * w['data_sensitivity']
        _, wp = predict_single(w)
        deltas[FEATURE_META[feat]["label"]] = round((wp - proba) * 100, 2)
    ddf = pd.DataFrame(list(deltas.items()), columns=["Feature","Delta (pp)"]).sort_values("Delta (pp)")
    fig2 = go.Figure(go.Bar(x=ddf["Delta (pp)"], y=ddf["Feature"], orientation='h',
        marker_color=['red' if v > 0 else 'green' for v in ddf["Delta (pp)"]]))
    fig2.update_layout(title="Risk change (pp) if each feature is pushed to maximum",
        xaxis_title="Percentage-point change", height=340,
        margin=dict(t=50, b=40, l=160, r=30),
        xaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1.5))
    st.plotly_chart(fig2, use_container_width=True)

# ================================================================
# MODEL INFO
# ================================================================
elif page == "📊 Model Info":
    st.title("📊 Model Information")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Model")
        st.info("**Random Forest Classifier**")
        st.write(f"Trees: {model.n_estimators}  |  Features: {len(feature_list)}")
    with c2:
        st.subheader("Class Weights")
        st.write(model.class_weight_)
    imp_df = pd.DataFrame({'Feature': feature_list, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# AI SCENARIO GENERATOR (rebuilt — closed loop)
# ================================================================
elif page == "🤖 AI Scenario Generator":
    st.title("🤖 AI Scenario Generator")
    st.markdown("Describe the incidents you want to stress-test. The AI writes structured scenarios — AERIE **automatically scores them**.")

    try:
        HF_TOKEN = st.secrets["HF_TOKEN"]
        st.sidebar.success("🔑 Token loaded from secrets")
    except:
        HF_TOKEN = st.text_input("Hugging Face API Token", type="password", help="huggingface.co/settings/tokens")

    if not HF_TOKEN:
        st.warning("Enter your Hugging Face API token to continue.")
        st.stop()

    a1, a2 = st.columns(2)
    with a1:
        industry = st.selectbox("Industry / Context", ["Financial Services","Healthcare","Manufacturing","Government","Retail","Energy & Utilities"])
        n_scenarios = st.slider("Number of scenarios", 3, 10, 5)
    with a2:
        threat = st.selectbox("Threat Focus", ["Mixed / Varied","Cybersecurity","Operational Failures","Data Breaches","Regulatory Incidents","Third-party / Supply Chain"])
        sev_bias = st.selectbox("Severity Bias", ["Realistic mix","Mostly high-severity","Mostly low-severity"])

    extra = st.text_area("Additional context (optional)", placeholder="E.g. 'Focus on cloud infrastructure'", height=70)

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

    if st.button("🚀 Generate & Score", use_container_width=True):
        with st.spinner("AI is writing scenarios… (15–30 seconds)"):
            try:
                raw = call_hf_api(prompt, HF_TOKEN)
                if isinstance(raw, dict) and 'error' in raw:
                    st.error(f"API error: {raw['error']}")
                    st.stop()
                text = raw[0].get('generated_text', str(raw)) if isinstance(raw, list) else str(raw)

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
                    out = pd.concat([scenarios_df, scored], axis=1).sort_values('probability', ascending=False).reset_index(drop=True)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Major Events", scored['predicted_major_event'].sum())
                    m2.metric("Avg Risk", f"{scored['probability'].mean():.1%}")
                    m3.metric("High Risk >70%", (scored['probability'] > 0.7).sum())

                    st.markdown("---")
                    st.subheader("📋 Scored Scenarios")

                    def highlight(row):
                        p = row['probability']
                        c = '#f8d7da' if p >= 0.7 else ('#fff3cd' if p >= 0.3 else '#d4edda')
                        return [f'background-color: {c}'] * len(row)

                    disp = [c for c in ['description','severity','downtime','financial_impact','regulatory_flag','data_sensitivity','criticality','probability','predicted_major_event'] if c in out.columns]
                    st.dataframe(
                        out[disp].style.apply(highlight, axis=1).format({
                            'probability': '{:.1%}', 'financial_impact': '${:,.0f}',
                            'data_sensitivity': '{:.2f}', 'downtime': '{:.1f}h'
                        }),
                        use_container_width=True
                    )

                    fig = px.bar(out.reset_index(), x='index', y='probability',
                        color='probability', color_continuous_scale=['green','yellow','red'],
                        range_color=[0,1], labels={'index':'Scenario #','probability':'Risk'},
                        title="Risk Probability by Scenario (highest → lowest)",
                        text=out['probability'].apply(lambda p: f"{p:.0%}").values)
                    fig.update_traces(textposition='outside')
                    fig.update_layout(yaxis=dict(range=[0,1.15]), coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)

                    st.download_button("📥 Download Scored Scenarios", out.to_csv(index=False), "aerie_ai_scenarios.csv", "text/csv")

            except requests.exceptions.Timeout:
                st.error("Timed out — the model may be cold-starting. Try again in 30s.")
            except Exception as e:
                st.error(f"Error: {e}")
