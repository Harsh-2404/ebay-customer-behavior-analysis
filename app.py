import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
        margin-bottom: 10px;
        height: 120px;
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
        font-size: 20px !important;
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
# 2. SAFE DATA LOADING & DYNAMIC COLUMN MAPPING
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
                    [3, 6, 10, 18, 22, 28, 35, 42, 50, 61, 67], size=n
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
                "Exploration_Depth": np.random.choice(
                    ["First page", "Multiple pages"], size=n
                ),
                "Appreciated_Feature": np.random.choice(
                    [
                        "Customer service",
                        "User-friendly website/app interface",
                        "Wide product selection",
                        "Product recommendations",
                    ],
                    size=n,
                ),
                "Improvement_Area": np.random.choice(
                    [
                        "Scrolling option would be much better than going to next page",
                        "Shipping speed and reliability",
                        "Quality of product is very poor according to the big offers",
                    ],
                    size=n,
                ),
                "Customer_Reviews_Importance": np.random.uniform(1, 5, size=n),
                "Personalized_Recommendation_Rating": np.random.uniform(
                    1, 5, size=n
                ),
            }
        )

    # Column name cleaning: Spaces ko remove/replace aur matching fix
    df.columns = df.columns.str.strip()

    # Dynamic Column Name Mapper (Flexible matching for different CSV column headers)
    column_mapper = {}
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        if "age" in col_lower:
            column_mapper[col] = "age"
        elif "gender" in col_lower:
            column_mapper[col] = "Gender"
        elif "frequency" in col_lower or "cadence" in col_lower:
            column_mapper[col] = "Purchase_Frequency"
        elif "satisfaction" in col_lower or "rating" in col_lower:
            column_mapper[col] = "Shopping_Satisfaction"
        elif "abandon" in col_lower or "reason" in col_lower:
            column_mapper[col] = "Abandonment_Reason"
        elif "search" in col_lower or "method" in col_lower:
            column_mapper[col] = "Search_Method"
        elif "depth" in col_lower or "exploration" in col_lower:
            column_mapper[col] = "Exploration_Depth"
        elif "feature" in col_lower or "appreciat" in col_lower:
            column_mapper[col] = "Appreciated_Feature"
        elif "improve" in col_lower or "area" in col_lower:
            column_mapper[col] = "Improvement_Area"
        elif "review" in col_lower:
            column_mapper[col] = "Customer_Reviews_Importance"
        elif "recommend" in col_lower:
            column_mapper[col] = "Personalized_Recommendation_Rating"

    df = df.rename(columns=column_mapper)

    # Required columns check with fallback values
    required_defaults = {
        "age": 25,
        "Gender": "Not Specified",
        "Purchase_Frequency": "Once a month",
        "Shopping_Satisfaction": 3,
        "Abandonment_Reason": "High shipping costs",
        "Search_Method": "Keyword",
        "Exploration_Depth": "First page",
        "Appreciated_Feature": "User-friendly website/app interface",
        "Improvement_Area": "User interface",
        "Customer_Reviews_Importance": 3.0,
        "Personalized_Recommendation_Rating": 3.0,
    }

    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Data Cleanups
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 13) & (df["age"] <= 80)].copy()

    df["Appreciated_Feature"] = df["Appreciated_Feature"].replace(
        {"Unknown": "Not Specified", np.nan: "Not Specified"}
    )
    df["Improvement_Area"] = df["Improvement_Area"].replace(
        {"Unknown": "Not Specified", np.nan: "Not Specified"}
    )

    return df


df_raw = load_data()

# ==========================================
# 3. SIDEBAR FILTERS
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg",
        width=140,
    )
    st.title("Executive Control Panel")
    st.markdown("---")

    all_genders = list(df_raw["Gender"].dropna().unique())
    selected_gender = st.multiselect(
        "👥 Filter Gender:", options=all_genders, default=all_genders
    )

    min_age_val = int(df_raw["age"].min()) if not df_raw.empty else 13
    max_age_val = int(df_raw["age"].max()) if not df_raw.empty else 70
    selected_age = st.slider(
        "🎂 Select Age Range:",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val),
    )

    all_cadence = list(df_raw["Purchase_Frequency"].dropna().unique())
    selected_cadence = st.multiselect(
        "🛍️ Purchase Cadence:", options=all_cadence, default=all_cadence
    )

df_filtered = df_raw[
    (df_raw["Gender"].isin(selected_gender))
    & (df_raw["age"].between(selected_age[0], selected_age[1]))
    & (df_raw["Purchase_Frequency"].isin(selected_cadence))
]

if df_filtered.empty:
    st.warning(
        "⚠️ Selected filters ka koi data nahi mila. Sidebar filters reset karein."
    )
    st.stop()

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("🛒 eBay Customer Behavior & ML Insights Dashboard")
st.caption("Executive Analytics Portal & Machine Learning Customer Segmentation")
st.markdown(" ")

# KPI Bar
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Total Buyers</div><div class="kpi-value">{len(df_filtered)}</div><div class="kpi-sub">Cohort Count</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    avg_sat = round(df_filtered["Shopping_Satisfaction"].mean(), 2)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Avg Satisfaction</div><div class="kpi-value">{avg_sat} / 5</div><div class="kpi-sub">Rating Score</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    top_ab = df_filtered["Abandonment_Reason"].mode()[0]
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Top Friction</div><div class="kpi-value" style="font-size:15px !important;">{top_ab}</div><div class="kpi-sub">Primary Reason</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    avg_age = round(df_filtered["age"].mean(), 1)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Average Age</div><div class="kpi-value">{avg_age} Yrs</div><div class="kpi-sub">Filtered Group</div></div>""",
        unsafe_allow_html=True,
    )
with k5:
    top_srch = df_filtered["Search_Method"].mode()[0]
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Top Search</div><div class="kpi-value">{top_srch}</div><div class="kpi-sub">Main Channel</div></div>""",
        unsafe_allow_html=True,
    )

st.markdown("---")

# Tabs
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
        fig_age = px.histogram(
            df_filtered,
            x="age",
            color="Gender",
            barmode="group",
            title="Age & Gender Breakdown",
        )
        st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        cad_cnt = df_filtered["Purchase_Frequency"].value_counts().reset_index()
        cad_cnt.columns = ["Cadence", "Count"]
        fig_cad = px.pie(
            cad_cnt, names="Cadence", values="Count", title="Purchase Cadence"
        )
        st.plotly_chart(fig_cad, use_container_width=True)

with tab2:
    f1, f2 = st.columns(2)
    with f1:
        ab_cnt = (
            df_filtered["Abandonment_Reason"].value_counts().reset_index()
        )
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
        srch_cnt = df_filtered["Search_Method"].value_counts().reset_index()
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
    num_cols = ["age", "Shopping_Satisfaction"]
    X = df_filtered[num_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_clustered = df_filtered.loc[X.index].copy()
    df_clustered["Cluster"] = [f"Cluster {c+1}" for c in clusters]

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
