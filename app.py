import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# EXACT COLUMN ORDER FROM YOUR NOTEBOOK (including duplicates)
# ─────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MonthlyCharges', 'TotalCharges',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
    # Duplicates present in your training data — kept to match model exactly
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
]

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Manrope:wght@300;400;500;600&display=swap');

:root {
    --bg:      #080c14;
    --surface: #0e1422;
    --card:    #131929;
    --border:  #1e2d45;
    --accent:  #00e5ff;
    --danger:  #ff3d6b;
    --success: #00e096;
    --warn:    #ffb300;
    --text:    #e8edf5;
    --muted:   #5a6a82;
}
html, body, [class*="css"] { font-family: 'Manrope', sans-serif; background: var(--bg) !important; color: var(--text); }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); min-width: 320px !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
button[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; background: var(--accent) !important; color: #080c14 !important; border-radius: 50% !important; }
.main .block-container { padding: 1.5rem 2rem; max-width: 100%; }

.topbar { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.5rem; padding-bottom:1rem; border-bottom:1px solid var(--border); }
.logo { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; letter-spacing:-0.03em; }
.logo span { color:var(--accent); }
.badge { background:rgba(0,229,255,0.1); border:1px solid rgba(0,229,255,0.3); color:var(--accent); font-size:0.7rem; font-weight:600; padding:3px 10px; border-radius:20px; text-transform:uppercase; letter-spacing:0.08em; }

.metric-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-bottom:1.5rem; }
.metric-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.3rem; position:relative; overflow:hidden; transition:border-color 0.2s; }
.metric-card:hover { border-color:var(--accent); }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.metric-card.blue::before  { background:var(--accent); }
.metric-card.red::before   { background:var(--danger); }
.metric-card.green::before { background:var(--success); }
.metric-card.warn::before  { background:var(--warn); }
.metric-label { font-size:0.72rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.metric-value { font-family:'Syne',sans-serif; font-size:2rem; font-weight:700; line-height:1; }
.metric-value.blue  { color:var(--accent); }
.metric-value.red   { color:var(--danger); }
.metric-value.green { color:var(--success); }
.metric-value.warn  { color:var(--warn); }
.metric-sub { font-size:0.75rem; color:var(--muted); margin-top:4px; }

.panel { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:1.2rem; }
.panel-title { font-family:'Syne',sans-serif; font-size:0.85rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:0.12em; margin-bottom:1rem; }

.result-wrap { border-radius:14px; padding:1.5rem; text-align:center; margin-top:0.5rem; }
.result-churn { background:rgba(255,61,107,0.1); border:1px solid rgba(255,61,107,0.4); }
.result-safe  { background:rgba(0,224,150,0.08); border:1px solid rgba(0,224,150,0.35); }
.result-big { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; }
.result-sub { font-size:0.85rem; color:var(--muted); margin-top:4px; }

.riskbar-wrap { background:rgba(255,255,255,0.06); border-radius:8px; height:10px; overflow:hidden; margin:8px 0; }
.riskbar-fill { height:100%; border-radius:8px; }

.sidebar-section { font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700; color:var(--accent); text-transform:uppercase; letter-spacing:0.12em; margin:1.2rem 0 0.5rem; padding-bottom:4px; border-bottom:1px solid var(--border); }

.hist-row { display:flex; justify-content:space-between; align-items:center; padding:8px 0; border-bottom:1px solid var(--border); font-size:0.82rem; }
.hist-row:last-child { border-bottom:none; }
.pill { font-size:0.7rem; font-weight:600; padding:2px 10px; border-radius:20px; }
.pill-churn { background:rgba(255,61,107,0.15); color:var(--danger); }
.pill-safe  { background:rgba(0,224,150,0.12);  color:var(--success); }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div { background:#0d1520 !important; border-color:var(--border) !important; color:var(--text) !important; }

div.stButton > button { background:linear-gradient(135deg,#00e5ff 0%,#0099cc 100%); color:#080c14; font-family:'Syne',sans-serif; font-weight:700; font-size:0.9rem; border:none; border-radius:10px; padding:0.65rem 1.5rem; width:100%; letter-spacing:0.04em; }
div.stButton > button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "history"        not in st.session_state: st.session_state.history        = []
if "model"          not in st.session_state: st.session_state.model          = None
if "total_analyzed" not in st.session_state: st.session_state.total_analyzed = 0
if "total_churn"    not in st.session_state: st.session_state.total_churn    = 0

# ─────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 0.5rem;'>
        <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:800;letter-spacing:-0.02em;'>
            Churn<span style='color:#00e5ff;'>IQ</span>
        </div>
        <div style='font-size:0.72rem;color:#5a6a82;margin-top:2px;'>Telecom Churn Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**Load Model**")
    model_file = st.file_uploader("Upload .pkl file", type=["pkl"], label_visibility="collapsed")
    if model_file:
        try:
            loaded = pickle.load(model_file)
        except Exception:
            try:
                model_file.seek(0)
                loaded = joblib.load(model_file)
            except Exception as e:
                st.error(f"Error loading file: {e}")
                loaded = None

        if loaded is not None:
            # Handle both formats: plain model OR dict {"model": ..., "columns": ...}
            if isinstance(loaded, dict):
                st.session_state.model = loaded.get("model", None)
                if st.session_state.model is None:
                    st.error("❌ Dict has no 'model' key. Check how you saved the model.")
            else:
                st.session_state.model = loaded

    if st.session_state.model:
        st.success(f"✓ {type(st.session_state.model).__name__}")
    else:
        st.info("Upload churn_pipeline.pkl to enable predictions")

    st.divider()

    st.markdown('<div class="sidebar-section">Demographics</div>', unsafe_allow_html=True)
    gender     = st.selectbox("Gender",          ["Male", "Female"])
    senior     = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner    = st.selectbox("Partner",          ["Yes", "No"])
    dependents = st.selectbox("Dependents",       ["No", "Yes"])
    tenure     = st.slider("Tenure (months)", 0, 72, 12)

    st.markdown('<div class="sidebar-section">Services</div>', unsafe_allow_html=True)
    phone_service    = st.selectbox("Phone Service",     ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines",    ["No", "Yes", "No phone service"])
    internet         = st.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
    online_security  = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup",     ["Yes", "No", "No internet service"])
    device_protect   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    st.markdown('<div class="sidebar-section">Billing</div>', unsafe_allow_html=True)
    contract        = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment         = st.selectbox("Payment Method",    [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0,  65.0, 0.5)
    total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0, 780.0, 10.0)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡ Run Prediction")

# ─────────────────────────────────────────────────────────────────────
# PREPROCESSING — matches your notebook column order EXACTLY
# ─────────────────────────────────────────────────────────────────────
def preprocess(inp):
    # Step 1: build all unique dummy values
    all_dummies = {
        'MultipleLines_No': 0,                      'MultipleLines_No phone service': 0,      'MultipleLines_Yes': 0,
        'InternetService_DSL': 0,                   'InternetService_Fiber optic': 0,         'InternetService_No': 0,
        'OnlineSecurity_No': 0,                     'OnlineSecurity_No internet service': 0,  'OnlineSecurity_Yes': 0,
        'OnlineBackup_No': 0,                       'OnlineBackup_No internet service': 0,    'OnlineBackup_Yes': 0,
        'DeviceProtection_No': 0,                   'DeviceProtection_No internet service': 0,'DeviceProtection_Yes': 0,
        'TechSupport_No': 0,                        'TechSupport_No internet service': 0,     'TechSupport_Yes': 0,
        'StreamingTV_No': 0,                        'StreamingTV_No internet service': 0,     'StreamingTV_Yes': 0,
        'StreamingMovies_No': 0,                    'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 0,
        'Contract_Month-to-month': 0,               'Contract_One year': 0,                   'Contract_Two year': 0,
        'PaperlessBilling_No': 0,                   'PaperlessBilling_Yes': 0,
        'PaymentMethod_Bank transfer (automatic)': 0,'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,        'PaymentMethod_Mailed check': 0,
    }

    # Step 2: set selected category to 1
    for col, val in [
        ('MultipleLines',    inp['multiple_lines']),
        ('InternetService',  inp['internet']),
        ('OnlineSecurity',   inp['online_security']),
        ('OnlineBackup',     inp['online_backup']),
        ('DeviceProtection', inp['device_protect']),
        ('TechSupport',      inp['tech_support']),
        ('StreamingTV',      inp['streaming_tv']),
        ('StreamingMovies',  inp['streaming_movies']),
        ('Contract',         inp['contract']),
        ('PaperlessBilling', inp['paperless']),
        ('PaymentMethod',    inp['payment']),
    ]:
        key = f"{col}_{val}"
        if key in all_dummies:
            all_dummies[key] = 1

    # Step 3: base numeric row
    base = {
        'gender':         1 if inp['gender'] == 'Male' else 0,
        'SeniorCitizen':  1 if inp['senior']  == 'Yes' else 0,
        'Partner':        1 if inp['partner']  == 'Yes' else 0,
        'Dependents':     1 if inp['dependents'] == 'Yes' else 0,
        'tenure':         inp['tenure'],
        'PhoneService':   1 if inp['phone_service'] == 'Yes' else 0,
        'MonthlyCharges': inp['monthly_charges'],
        'TotalCharges':   inp['total_charges'],
        **all_dummies,
    }

    # Step 4: build DataFrame and reindex to EXACT column order
    # (including the 3 duplicate MultipleLines columns at the end)
    df = pd.DataFrame([base])
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return df

# ─────────────────────────────────────────────────────────────────────
# RUN PREDICTION
# ─────────────────────────────────────────────────────────────────────
prediction_result = None
churn_prob        = None

if predict_btn:
    if not st.session_state.model:
        st.sidebar.error("Upload a model first!")
    else:
        inp = dict(
            gender=gender, senior=senior, partner=partner,
            dependents=dependents, tenure=tenure,
            phone_service=phone_service, multiple_lines=multiple_lines,
            internet=internet, online_security=online_security,
            online_backup=online_backup, device_protect=device_protect,
            tech_support=tech_support, streaming_tv=streaming_tv,
            streaming_movies=streaming_movies, contract=contract,
            paperless=paperless, payment=payment,
            monthly_charges=monthly_charges, total_charges=total_charges,
        )
        try:
            X    = preprocess(inp)
            pred = int(st.session_state.model.predict(X)[0])
            prob = 0.5
            if hasattr(st.session_state.model, "predict_proba"):
                prob = float(st.session_state.model.predict_proba(X)[0][1])

            prediction_result = pred
            churn_prob        = prob

            st.session_state.total_analyzed += 1
            if pred == 1:
                st.session_state.total_churn += 1

            st.session_state.history.insert(0, {
                "time":     datetime.now().strftime("%H:%M:%S"),
                "contract": contract,
                "tenure":   tenure,
                "monthly":  monthly_charges,
                "churn":    pred,
                "prob":     prob,
            })
            if len(st.session_state.history) > 10:
                st.session_state.history = st.session_state.history[:10]

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Tip: Make sure your model was saved AFTER the get_dummies step in your notebook.")

# ─────────────────────────────────────────────────────────────────────
# DASHBOARD UI
# ─────────────────────────────────────────────────────────────────────
churn_rate = (st.session_state.total_churn / st.session_state.total_analyzed * 100) if st.session_state.total_analyzed > 0 else 0
safe_count = st.session_state.total_analyzed - st.session_state.total_churn

# Top bar
st.markdown(f"""
<div class="topbar">
    <div class="logo">Churn<span>IQ</span> <span style="font-size:0.9rem;font-weight:400;color:#5a6a82;">· Dashboard</span></div>
    <div style="display:flex;gap:10px;align-items:center;">
        <span class="badge">Live</span>
        <span style="font-size:0.78rem;color:#5a6a82;">{datetime.now().strftime("%d %b %Y · %H:%M")}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric cards
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card blue">
        <div class="metric-label">Total Analyzed</div>
        <div class="metric-value blue">{st.session_state.total_analyzed}</div>
        <div class="metric-sub">customers this session</div>
    </div>
    <div class="metric-card red">
        <div class="metric-label">Churn Risk</div>
        <div class="metric-value red">{st.session_state.total_churn}</div>
        <div class="metric-sub">flagged customers</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Safe Customers</div>
        <div class="metric-value green">{safe_count}</div>
        <div class="metric-sub">low risk</div>
    </div>
    <div class="metric-card warn">
        <div class="metric-label">Churn Rate</div>
        <div class="metric-value warn">{churn_rate:.1f}%</div>
        <div class="metric-sub">session average</div>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1], gap="medium")

with col_left:
    # Prediction result
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Prediction Result</div>', unsafe_allow_html=True)
    if prediction_result is None:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0;color:#5a6a82;'>
            <div style='font-size:2.5rem;margin-bottom:8px;'>⚡</div>
            <div style='font-family:Syne,sans-serif;font-size:1rem;'>
                Configure customer in sidebar<br>and click <b style='color:#00e5ff;'>Run Prediction</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        label   = "CHURN RISK" if prediction_result == 1 else "SAFE"
        css_res = "result-churn" if prediction_result == 1 else "result-safe"
        color   = "#ff3d6b"     if prediction_result == 1 else "#00e096"
        icon    = "🔴"           if prediction_result == 1 else "🟢"
        pct     = churn_prob * 100
        action  = "Consider a retention offer or contract upgrade." if prediction_result == 1 else "No immediate action required."
        st.markdown(f"""
        <div class="result-wrap {css_res}">
            <div style='font-size:2.5rem;'>{icon}</div>
            <div class="result-big" style='color:{color};'>{label}</div>
            <div class="result-sub">{action}</div>
            <div style='margin:14px 0 4px;font-size:0.75rem;color:#5a6a82;'>Churn probability</div>
            <div class="riskbar-wrap">
                <div class="riskbar-fill" style='width:{pct:.1f}%;background:{"linear-gradient(90deg,#ff3d6b,#ff6b35)" if prediction_result==1 else "linear-gradient(90deg,#00e096,#00b8d4)"}'></div>
            </div>
            <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;color:{color};'>{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Gauge
    if churn_prob is not None:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Risk Gauge</div>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(churn_prob * 100, 1),
            number={"suffix": "%", "font": {"size": 36, "color": "#e8edf5", "family": "Syne"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5a6a82", "tickfont": {"color": "#5a6a82"}},
                "bar": {"color": "#ff3d6b" if churn_prob > 0.5 else "#00e096", "thickness": 0.25},
                "bgcolor": "#131929", "bordercolor": "#1e2d45",
                "steps": [
                    {"range": [0,  40], "color": "rgba(0,224,150,0.1)"},
                    {"range": [40, 70], "color": "rgba(255,179,0,0.1)"},
                    {"range": [70,100], "color": "rgba(255,61,107,0.1)"},
                ],
                "threshold": {"line": {"color": "#00e5ff", "width": 2}, "thickness": 0.75, "value": 50},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=20, b=10, l=20, r=20), height=200,
            font={"family": "Manrope"}
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Key risk factors
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Key Risk Factors</div>', unsafe_allow_html=True)
    factors = {
        "Contract type":    1.0 - (["Month-to-month","One year","Two year"].index(contract) * 0.45),
        "Tenure":           max(0, 1.0 - tenure / 72),
        "Monthly charges":  min(monthly_charges / 120, 1.0),
        "Tech support":     0.7 if tech_support == "No" else 0.2,
        "Online security":  0.65 if online_security == "No" else 0.2,
        "Internet (Fiber)": 0.8 if internet == "Fiber optic" else 0.3,
        "Paperless billing":0.55 if paperless == "Yes" else 0.25,
    }
    factor_df = pd.DataFrame({"Factor": list(factors.keys()), "Score": list(factors.values())}).sort_values("Score", ascending=True)
    colors_f  = ["#ff3d6b" if v > 0.6 else "#ffb300" if v > 0.4 else "#00e096" for v in factor_df["Score"]]
    fig_f = go.Figure(go.Bar(
        x=factor_df["Score"], y=factor_df["Factor"], orientation="h",
        marker_color=colors_f,
        text=[f"{v:.0%}" for v in factor_df["Score"]],
        textposition="outside", textfont={"color": "#e8edf5", "size": 11},
    ))
    fig_f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=0, b=0, l=0, r=50), height=240,
        xaxis={"showgrid": False, "zeroline": False, "visible": False},
        yaxis={"tickfont": {"color": "#a0b0c8", "size": 11}, "gridcolor": "#1e2d45"},
        font={"family": "Manrope"},
    )
    st.plotly_chart(fig_f, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    # Customer profile
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Customer Profile</div>', unsafe_allow_html=True)
    profile_items = [
        ("Gender", gender), ("Senior Citizen", senior), ("Partner", partner),
        ("Dependents", dependents), ("Tenure", f"{tenure} months"),
        ("Contract", contract), ("Internet", internet),
        ("Monthly Charges", f"${monthly_charges:.2f}"),
        ("Total Charges", f"${total_charges:,.2f}"), ("Payment", payment),
    ]
    st.markdown("".join([
        f"<div class='hist-row'>"
        f"<span style='color:#5a6a82;font-size:0.78rem;'>{k}</span>"
        f"<span style='font-weight:500;font-size:0.82rem;'>{v}</span></div>"
        for k, v in profile_items
    ]), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Donut
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Session Overview</div>', unsafe_allow_html=True)
    if st.session_state.total_analyzed > 0:
        fig_donut = go.Figure(go.Pie(
            labels=["Churn Risk", "Safe"], values=[st.session_state.total_churn, safe_count],
            hole=0.65, marker_colors=["#ff3d6b", "#00e096"], textinfo="none",
        ))
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=0, b=0, l=0, r=0), height=180,
            showlegend=True,
            legend={"font": {"color": "#a0b0c8", "size": 11}, "orientation": "h", "y": -0.1},
            annotations=[{"text": f"{churn_rate:.0f}%<br>churn", "x": 0.5, "y": 0.5,
                           "showarrow": False, "font": {"size": 20, "color": "#ff3d6b", "family": "Syne"}}],
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("<p style='color:#5a6a82;text-align:center;padding:1.5rem 0;font-size:0.85rem;'>No predictions yet this session</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # History
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Recent Predictions</div>', unsafe_allow_html=True)
    if st.session_state.history:
        for h in st.session_state.history[:6]:
            pill_cls  = "pill-churn" if h["churn"] else "pill-safe"
            pill_text = "Churn" if h["churn"] else "Safe"
            st.markdown(f"""
            <div class="hist-row">
                <div>
                    <div style='font-weight:500;'>{h['contract']}</div>
                    <div style='color:#5a6a82;font-size:0.72rem;'>{h['tenure']}mo · ${h['monthly']:.0f}/mo · {h['time']}</div>
                </div>
                <div style='display:flex;align-items:center;gap:8px;'>
                    <span style='font-size:0.78rem;color:#5a6a82;'>{h['prob']*100:.0f}%</span>
                    <span class="pill {pill_cls}">{pill_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color:#5a6a82;text-align:center;padding:1rem 0;font-size:0.85rem;'>No predictions yet</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Trend line
if len(st.session_state.history) >= 2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Churn Probability Trend · Session</div>', unsafe_allow_html=True)
    hist_rev = list(reversed(st.session_state.history))
    probs    = [h["prob"] * 100 for h in hist_rev]
    labels   = [f"#{i+1}" for i in range(len(hist_rev))]
    colors_t = ["#ff3d6b" if p > 50 else "#00e096" for p in probs]
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(
        x=labels, y=probs, mode="lines+markers",
        line={"color": "#00e5ff", "width": 2},
        marker={"color": colors_t, "size": 10, "line": {"width": 2, "color": "#080c14"}},
        fill="tozeroy", fillcolor="rgba(0,229,255,0.05)",
    ))
    fig_t.add_hline(y=50, line_dash="dash", line_color="#ffb300", line_width=1,
                    annotation_text="50% threshold", annotation_font_color="#ffb300")
    fig_t.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=10, r=10), height=180,
        xaxis={"gridcolor": "#1e2d45", "tickfont": {"color": "#5a6a82"}},
        yaxis={"gridcolor": "#1e2d45", "tickfont": {"color": "#5a6a82"}, "range": [0,100], "ticksuffix": "%"},
        font={"family": "Manrope"}, showlegend=False,
    )
    st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
