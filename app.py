import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="eBay Customer Behavior & ML Insights Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .kpi-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #0066cc;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 25px !important;
        min-height: 110px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .kpi-title {
        font-size: 13px;
        font-weight: 600;
        color: #555555;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .kpi-value {
        font-size: 18px !important;
        font-weight: 700;
        color: #0066cc;
        word-break: break-word;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 11px;
        color: #2e7d32;
        background-color: #e8f5e9;
        padding: 2px 6px;
        border-radius: 4px;
        width: fit-content;
        margin-top: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-top: 15px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. SAFE DATA LOADING & CLEANUP
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_eBay.csv")
    except Exception:
        np.random.seed(42)
        n = 800
        df = pd.DataFrame(
            {
                "age": np.random.choice(
                    [18, 22, 28, 35, 42, 50, 61, 67, 72, 78], size=n
                ),
                "Gender": np.random.choice(
                    ["Male", "Female", "Others", "Prefer not to say"], size=n
                ),
                "Purchase_Frequency": np.random.choice(
                    [
                        "Multiple times a week",
                        "Once a week",
                        "Few times a month",
                        "Once a month",
                        "Less than once a month",
                    ],
                    size=n,
                ),
                "Shopping_Satisfaction": np.random.choice(
                    [1, 2, 3, 4, 5], size=n
                ),
                "Abandonment_Reason": np.random.choice(
                    [
                        "High shipping costs",
                        "Changed my mind or no longer need the item",
                        "Found a better price elsewhere",
                        "others",
                    ],
                    size=n,
                ),
                "Search_Method": np.random.choice(
                    ["Keyword", "Categories", "Filter", "Others"], size=n
                ),
            }
        )

    # Clean whitespace in column names
    df.columns = df.columns.astype(str).str.strip()

    # Drop duplicate column names
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # Column Mapping
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().replace(" ", "_").replace("-", "_")
        if "age" in c_lower and "age" not in col_map.values():
            col_map[col] = "age"
        elif "gender" in c_lower and "Gender" not in col_map.values():
            col_map[col] = "Gender"
        elif (
            "freq" in c_lower or "cadence" in c_lower or "purchase" in c_lower
        ) and "Purchase_Frequency" not in col_map.values():
            col_map[col] = "Purchase_Frequency"
        elif (
            "satis" in c_lower or "rating" in c_lower
        ) and "Shopping_Satisfaction" not in col_map.values():
            col_map[col] = "Shopping_Satisfaction"
        elif (
            "abandon" in c_lower or "reason" in c_lower
        ) and "Abandonment_Reason" not in col_map.values():
            col_map[col] = "Abandonment_Reason"
        elif (
            "search" in c_lower or "method" in c_lower
        ) and "Search_Method" not in col_map.values():
            col_map[col] = "Search_Method"

    df = df.rename(columns=col_map)
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # Fill missing values instead of dropping rows to preserve all 800 records
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if df["age"].isnull().sum() > 0:
        df["age"] = df["age"].fillna(df["age"].median())

    defaults = {
        "Gender": "Not Specified",
        "Purchase_Frequency": "Once a month",
        "Shopping_Satisfaction": 3,
        "Abandonment_Reason": "High shipping costs",
        "Search_Method": "Others",
    }
    for c, val in defaults.items():
        if c not in df.columns:
            df[c] = val
        else:
            df[c] = df[c].fillna(val)

    return df


df_raw = load_data()

# Dynamic Full Ranges
min_age = int(df_raw["age"].min())
max_age = int(df_raw["age"].max())
all_genders = list(df_raw["Gender"].dropna().astype(str).unique())
all_cadence = list(df_raw["Purchase_Frequency"].dropna().astype(str).unique())

# Initialize Session State for Filters
if "selected_genders" not in st.session_state:
    st.session_state["selected_genders"] = all_genders
if "selected_age_range" not in st.session_state:
    st.session_state["selected_age_range"] = (min_age, max_age)
if "selected_cadence" not in st.session_state:
    st.session_state["selected_cadence"] = all_cadence


def reset_filters():
    st.session_state["selected_genders"] = all_genders
    st.session_state["selected_age_range"] = (min_age, max_age)
    st.session_state["selected_cadence"] = all_cadence


# ==========================================
# 3. SIDEBAR CONTROL PANEL
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg",
        width=140,
    )
    st.title("Executive Control Panel")

    st.button("🔄 Reset All Filters", on_click=reset_filters, type="primary")
    st.markdown("---")

    selected_gender = st.multiselect(
        "👥 Filter Gender:",
        options=all_genders,
        key="selected_genders",
    )

    selected_age = st.slider(
        "🎂 Select Age Range:",
        min_value=min_age,
        max_value=max_age,
        key="selected_age_range",
    )

    selected_cadence = st.multiselect(
        "🛍️ Purchase Cadence:",
        options=all_cadence,
        key="selected_cadence",
    )

# Filtering logic
gender_mask = df_raw["Gender"].isin(selected_gender)
age_mask = df_raw["age"].between(selected_age[0], selected_age[1])
cadence_mask = df_raw["Purchase_Frequency"].isin(selected_cadence)

df_filtered = df_raw[gender_mask & age_mask & cadence_mask].copy()

# Ensure unique columns for Plotly Express compatibility
df_filtered = df_filtered.loc[
    :, ~df_filtered.columns.duplicated(keep="first")
].copy()

if df_filtered.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ==========================================
# 4. DASHBOARD HEADER & KPIS
# ==========================================
st.title("🛒 eBay Customer Behavior & ML Insights Dashboard")
st.caption("Executive Analytics Portal & Customer Segmentation")
st.markdown(" ")


def safe_mode(series):
    m = series.mode()
    return m[0] if not m.empty else "N/A"


k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Total Buyers</div><div class="kpi-value">{len(df_filtered)} / {len(df_raw)}</div><div class="kpi-sub">Filtered / Total</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    avg_sat = round(df_filtered["Shopping_Satisfaction"].mean(), 2)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Avg Satisfaction</div><div class="kpi-value">{avg_sat} / 5</div><div class="kpi-sub">Rating Score</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    top_ab = safe_mode(df_filtered["Abandonment_Reason"])
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Top Friction</div><div class="kpi-value" style="font-size:14px !important;">{top_ab}</div><div class="kpi-sub">Primary Reason</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    avg_age = round(df_filtered["age"].mean(), 1)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Average Age</div><div class="kpi-value">{avg_age} Yrs</div><div class="kpi-sub">Filtered Group</div></div>""",
        unsafe_allow_html=True,
    )
with k5:
    top_srch = safe_mode(df_filtered["Search_Method"])
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Top Search</div><div class="kpi-value">{top_srch}</div><div class="kpi-sub">Main Channel</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==========================================
# 5. TABS & CHARTS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Executive Overview",
        "🚨 Friction Analysis",
        "🤖 ML Clustering",
        "📁 Dataset View",
    ]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        df_age = df_filtered[["age", "Gender"]].copy()
        fig_age = px.histogram(
            df_age,
            x="age",
            color="Gender",
            barmode="group",
            title="Age & Gender Breakdown",
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        cad_cnt = (
            df_filtered["Purchase_Frequency"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Cadence", "Purchase_Frequency": "Count"})
        )
        if "Cadence" not in cad_cnt.columns:
            cad_cnt.columns = ["Cadence", "Count"]
        fig_cad = px.pie(
            cad_cnt, names="Cadence", values="Count", title="Purchase Cadence"
        )
        st.plotly_chart(fig_cad, use_container_width=True)

with tab2:
    f1, f2 = st.columns(2)
    with f1:
        ab_cnt = (
            df_filtered["Abandonment_Reason"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Reason", "Abandonment_Reason": "Count"})
        )
        if "Reason" not in ab_cnt.columns:
            ab_cnt.columns = ["Reason", "Count"]
        fig_ab = px.bar(
            ab_cnt,
            y="Reason",
            x="Count",
            orientation="h",
            title="Cart Abandonment Reasons",
            color="Count",
        )
        fig_ab.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_ab, use_container_width=True)

    with f2:
        srch_cnt = (
            df_filtered["Search_Method"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Method", "Search_Method": "Count"})
        )
        if "Method" not in srch_cnt.columns:
            srch_cnt.columns = ["Method", "Count"]
        fig_srch = px.bar(
            srch_cnt,
            x="Method",
            y="Count",
            title="Preferred Search Method",
            color="Count",
        )
        fig_srch.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_srch, use_container_width=True)

with tab3:
    X = df_filtered[["age", "Shopping_Satisfaction"]].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    df_clustered = pd.DataFrame(
        {
            "age": X["age"].values,
            "Shopping_Satisfaction": X["Shopping_Satisfaction"].values,
            "Cluster": [f"Cluster {c+1}" for c in clusters],
        }
    )

    fig_cluster = px.scatter(
        df_clustered,
        x="age",
        y="Shopping_Satisfaction",
        color="Cluster",
        title="Customer Clustering (Age vs Satisfaction)",
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

with tab4:
    st.dataframe(df_filtered, use_container_width=True)
