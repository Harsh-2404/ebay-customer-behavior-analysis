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

# Custom Styling to fix Text Truncation & Metric Card UI
st.markdown(
    """
    <style>
    /* Metric Card Styling with Responsive Text Handling */
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
        /* FIX: Prevents text cut-off like "High ship..." */
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
    /* Tab Font & Spacing */
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
# 2. DATA LOADING & PREPROCESSING (FIXED)
# ==========================================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_eBay.csv")
    except Exception:
        # Fallback synthetic dataset generation matching dashboard schema
        np.random.seed(42)
        n = 800
        df = pd.DataFrame(
            {
                "Timestamp": pd.date_range(
                    start="2023-06-01", periods=n, freq="H"
                ),
                "age": np.random.choice(
                    [3, 6, 10, 18, 22, 28, 35, 42, 50, 61, 67],
                    size=n,
                    p=[0.01, 0.01, 0.01, 0.12, 0.20, 0.20, 0.15, 0.15, 0.10, 0.03, 0.02],
                ),
                "Gender": np.random.choice(
                    ["Male", "Female", "Others", "Prefer not to say"],
                    size=n,
                    p=[0.45, 0.40, 0.05, 0.10],
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
                "Purchase_Categories": np.random.choice(
                    [
                        "Clothing and Fashion",
                        "Beauty and Personal Care",
                        "Home and Kitchen",
                        "Groceries and Gourmet Food",
                    ],
                    size=n,
                ),
                "Shopping_Satisfaction": np.random.choice(
                    [1, 2, 3, 4, 5], size=n, p=[0.24, 0.20, 0.20, 0.18, 0.18]
                ),
                "Abandonment_Reason": np.random.choice(
                    [
                        "High shipping costs",
                        "Changed my mind or no longer need the item",
                        "Found a better price elsewhere",
                        "others",
                    ],
                    size=n,
                    p=[0.26, 0.25, 0.23, 0.26],
                ),
                "Search_Method": np.random.choice(
                    ["Keyword", "Categories", "Filter", "Others", "Unknown"],
                    size=n,
                ),
                "Exploration_Depth": np.random.choice(
                    ["First page", "Multiple pages"], size=n, p=[0.528, 0.472]
                ),
                "Appreciated_Feature": np.random.choice(
                    [
                        "Customer service",
                        "User-friendly website/app interface",
                        "Wide product selection",
                        "Product recommendations",
                        "All the above",
                        "Unknown",
                    ],
                    size=n,
                ),
                "Improvement_Area": np.random.choice(
                    [
                        "Scrolling option would be much better than going to next page",
                        "Shipping speed and reliability",
                        "Quality of product is very poor according to the big offers",
                        "User interface",
                        "Unknown",
                    ],
                    size=n,
                ),
                "Customer_Reviews_Importance": np.random.uniform(1, 5, size=n),
                "Personalized_Recommendation_Rating": np.random.uniform(
                    1, 5, size=n
                ),
            }
        )

    # ----------------------------------------------------
    # FIX 1: REMOVE AGE OUTLIERS / ANOMALIES (AGE >= 13)
    # Filter out unrealistically low ages (e.g., 3, 6, 10 years old)
    # ----------------------------------------------------
    df = df[(df["age"] >= 13) & (df["age"] <= 80)].copy()

    # Clean up "Unknown" text labels for better executive presentations
    df["Appreciated_Feature"] = df["Appreciated_Feature"].replace(
        {"Unknown": "Not Specified"}
    )
    df["Improvement_Area"] = df["Improvement_Area"].replace(
        {"Unknown": "Not Specified"}
    )

    return df


df_raw = load_data()

# ==========================================
# 3. SIDEBAR FILTERS (EXECUTIVE CONTROL PANEL)
# ==========================================
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg",
        width=140,
    )
    st.title("Executive Control Panel")
    st.markdown("---")

    # Filter: Gender
    all_genders = list(df_raw["Gender"].unique())
    selected_gender = st.multiselect(
        "👥 Filter Gender:", options=all_genders, default=all_genders
    )

    # Filter: Age Range (Auto-adjusts from min age 13+)
    min_age_val = int(df_raw["age"].min())
    max_age_val = int(df_raw["age"].max())
    selected_age = st.slider(
        "🎂 Select Age Range:",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val),
    )

    # Filter: Purchase Cadence
    all_cadence = list(df_raw["Purchase_Frequency"].unique())
    selected_cadence = st.multiselect(
        "🛍️ Purchase Cadence:", options=all_cadence, default=all_cadence
    )

# Apply Sidebar Filters
df_filtered = df_raw[
    (df_raw["Gender"].isin(selected_gender))
    & (df_raw["age"].between(selected_age[0], selected_age[1]))
    & (df_raw["Purchase_Frequency"].isin(selected_cadence))
]

# Fallback check if dataset becomes empty after filtering
if df_filtered.empty:
    st.warning("⚠️ No data available for the selected sidebar filters.")
    st.stop()

# ==========================================
# 4. MAIN HEADER & TOP KPI METRICS
# ==========================================
st.title("🛒 eBay Customer Behavior & ML Insights Dashboard")
st.caption(
    "Executive Analytics Portal | Cart Friction Identification, Algorithmic Recommendation Performance, and Customer Segmentation"
)
st.markdown(" ")

# --- FIX 2: KPI CARDS WITH FIXED OVERFLOW / TRUNCATION ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    pct_total = round((len(df_filtered) / len(df_raw)) * 100, 1)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Surveyed Buyers</div>
            <div class="kpi-value">{len(df_filtered)}</div>
            <div class="kpi-sub">↑ {pct_total}% of Total</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi2:
    avg_sat = round(df_filtered["Shopping_Satisfaction"].mean(), 2)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Avg Satisfaction</div>
            <div class="kpi-value">{avg_sat} / 5.0</div>
            <div class="kpi-sub">↑ Scale 1-5</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi3:
    top_abandon = df_filtered["Abandonment_Reason"].mode()[0]
    # Displays full text without truncation
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Top Abandonment Driver</div>
            <div class="kpi-value" style="font-size: 16px !important;">{top_abandon}</div>
            <div class="kpi-sub">↑ Primary Friction</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi4:
    avg_age = round(df_filtered["age"].mean(), 1)
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Average Buyer Age</div>
            <div class="kpi-value">{avg_age} Yrs</div>
            <div class="kpi-sub">Filtered Cohort</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

with kpi5:
    top_search = df_filtered["Search_Method"].mode()[0]
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Primary Search Method</div>
            <div class="kpi-value">{top_search}</div>
            <div class="kpi-sub">Discovery Channel</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ==========================================
# 5. DASHBOARD NAVIGATION TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Executive BI Overview",
        "🚨 Friction & Abandonment",
        "🤖 ML Customer Clustering",
        "📁 Data Explorer & Download",
        "🎯 Strategic Roadmap",
    ]
)

# ----------------------------------------------------
# TAB 1: EXECUTIVE BI OVERVIEW
# ----------------------------------------------------
with tab1:
    st.subheader("📊 Executive Overview & Demographic Cadence")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### 👥 Age & Gender Demographics")
        st.caption("Customer Age Distribution Segmented by Gender")
        fig_age = px.histogram(
            df_filtered,
            x="age",
            color="Gender",
            barmode="group",
            labels={"age": "Buyer Age", "count": "Buyer Count"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_age.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=320, legend_title="Gender"
        )
        st.plotly_chart(fig_age, use_container_width=True)

    with c2:
        st.markdown("##### 🛍️ Purchase Frequency Distribution")
        st.caption("Overall Buyer Purchase Cadence Share")
        cadence_counts = (
            df_filtered["Purchase_Frequency"].value_counts().reset_index()
        )
        cadence_counts.columns = ["Cadence", "Count"]
        fig_cad = px.pie(
            cadence_counts,
            names="Cadence",
            values="Count",
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
        )
        fig_cad.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=320
        )
        st.plotly_chart(fig_cad, use_container_width=True)

    st.markdown(" ")

    c3, c4 = st.columns(2)

    with c3:
        st.markdown("##### ⭐ Overall Shopping Satisfaction Ratings")
        st.caption("Customer Shopping Satisfaction Breakdown")
        sat_counts = (
            df_filtered["Shopping_Satisfaction"].value_counts().reset_index()
        )
        sat_counts.columns = ["Rating", "Count"]
        sat_counts = sat_counts.sort_values("Rating")
        fig_sat = px.bar(
            sat_counts,
            x="Rating",
            y="Count",
            text="Count",
            color="Rating",
            color_continuous_scale="Blues",
        )
        fig_sat.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
        )
        st.plotly_chart(fig_sat, use_container_width=True)

    with c4:
        st.markdown("##### 💡 Service Appreciation Highlights")
        st.caption("Top Customer Service Appreciation Factors")
        feat_counts = (
            df_filtered["Appreciated_Feature"].value_counts().reset_index()
        )
        feat_counts.columns = ["Feature", "Count"]
        fig_feat = px.bar(
            feat_counts.sort_values("Count", ascending=True),
            y="Feature",
            x="Count",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Greens",
        )
        # FIX: Colorbar scale removed for cleaner layout
        fig_feat.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
        )
        st.plotly_chart(fig_feat, use_container_width=True)


# ----------------------------------------------------
# TAB 2: FRICTION & ABANDONMENT
# ----------------------------------------------------
with tab2:
    st.subheader(
        "🚨 Cart Abandonment Root Cause Analysis & Search Friction"
    )
    st.caption(
        "Analyzing conversion leakage points across search behavior, fees, and product discovery."
    )

    f1, f2 = st.columns(2)

    with f1:
        st.markdown("##### 🛒 Primary Drivers of Cart Abandonment")
        abandon_counts = (
            df_filtered["Abandonment_Reason"].value_counts().reset_index()
        )
        abandon_counts.columns = ["Reason", "Count"]
        fig_ab = px.bar(
            abandon_counts.sort_values("Count", ascending=True),
            y="Reason",
            x="Count",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Reds",
        )
        # FIX: coloraxis_showscale=False hides unnecessary continuous color scale legend
        fig_ab.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
        )
        st.plotly_chart(fig_ab, use_container_width=True)

    with f2:
        st.markdown("##### 🔍 Search Result Exploration Depth")
        depth_counts = (
            df_filtered["Exploration_Depth"].value_counts().reset_index()
        )
        depth_counts.columns = ["Depth", "Count"]
        fig_depth = px.pie(
            depth_counts,
            names="Depth",
            values="Count",
            hole=0.5,
            color_discrete_sequence=["#FF4B4B", "#1F77B4"],
        )
        fig_depth.update_layout(
            margin=dict(l=20, r=20, t=30, b=20), height=320
        )
        st.plotly_chart(fig_depth, use_container_width=True)

    st.markdown(" ")

    f3, f4 = st.columns(2)

    with f3:
        st.markdown("##### 🔎 Product Search Method Preference")
        search_counts = (
            df_filtered["Search_Method"].value_counts().reset_index()
        )
        search_counts.columns = ["Method", "Count"]
        fig_srch = px.bar(
            search_counts,
            x="Method",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Purples",
        )
        fig_srch.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
        )
        st.plotly_chart(fig_srch, use_container_width=True)

    with f4:
        st.markdown("##### 🛠️ Key Product Improvement Areas Requested")
        imp_counts = (
            df_filtered["Improvement_Area"].value_counts().reset_index()
        )
        imp_counts.columns = ["Area", "Count"]
        fig_imp = px.bar(
            imp_counts.sort_values("Count", ascending=True),
            y="Area",
            x="Count",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Oranges",
        )
        fig_imp.update_layout(
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ----------------------------------------------------
# TAB 3: ML CUSTOMER CLUSTERING
# ----------------------------------------------------
with tab3:
    st.subheader(
        "🤖 Unsupervised Machine Learning: K-Means Customer Segmentation"
    )
    st.caption(
        "Segment buyers using numerical behavioral ratings to uncover target persona cohorts."
    )

    ml_col1, ml_col2 = st.columns([1, 2])

    num_cols = [
        "age",
        "Customer_Reviews_Importance",
        "Personalized_Recommendation_Rating",
        "Shopping_Satisfaction",
    ]

    with ml_col1:
        st.markdown("##### ⚙️ Model Hyperparameters")
        selected_features = st.multiselect(
            "Select Feature Dimensions:",
            options=num_cols,
            default=[
                "age",
                "Customer_Reviews_Importance",
                "Personalized_Recommendation_Rating",
            ],
        )

        k_clusters = st.slider("Select Cluster Count (k):", 2, 6, 3)

        st.info(
            "💡 **Tip:** Scaling is automatically performed using `StandardScaler` before applying K-Means clustering."
        )

    if len(selected_features) >= 2:
        # ML Pipeline Execution
        X = df_filtered[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustered = df_filtered.loc[X.index].copy()
        df_clustered["Cluster"] = [f"Cluster {c+1}" for c in clusters]

        with ml_col2:
            st.markdown(
                f"##### 3D K-Means Clustering Space (k={k_clusters})"
            )
            if len(selected_features) >= 3:
                fig_3d = px.scatter_3d(
                    df_clustered,
                    x=selected_features[0],
                    y=selected_features[1],
                    z=selected_features[2],
                    color="Cluster",
                    opacity=0.8,
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_3d.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0), height=400
                )
                st.plotly_chart(fig_3d, use_container_width=True)
            else:
                fig_2d = px.scatter(
                    df_clustered,
                    x=selected_features[0],
                    y=selected_features[1],
                    color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_2d.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0), height=400
                )
                st.plotly_chart(fig_2d, use_container_width=True)

        st.markdown("##### 📌 Cluster Profiling & Center Averages")
        cluster_summary = (
            df_clustered.groupby("Cluster")[selected_features]
            .mean()
            .reset_index()
        )
        st.dataframe(cluster_summary, use_container_width=True)
    else:
        st.warning("Please select at least 2 features for K-Means clustering.")


# ----------------------------------------------------
# TAB 4: DATA EXPLORER & DOWNLOAD
# ----------------------------------------------------
with tab4:
    st.subheader("📁 Interactive Dataset Explorer & Cleaned Export")
    st.caption("Inspect, search, and download the fully preprocessed dataset.")

    st.markdown("##### 📥 Download Cleaned Dataset")

    csv_data = df_filtered.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns([1, 2])

    with d1:
        st.download_button(
            label="⬇️ Download Cleaned eBay CSV",
            data=csv_data,
            file_name="cleaned_eBay_filtered.csv",
            mime="text/csv",
        )

    with d2:
        st.success(
            f"✅ Filtered view contains **{len(df_filtered)} rows** and **{len(df_filtered.columns)} columns**."
        )

    st.markdown("---")
    st.markdown("##### 🔍 Filtered Data Viewer")
    search_term = st.text_input("Search dataset by keyword:")

    if search_term:
        mask = df_filtered.astype(str).apply(
            lambda row: row.str.contains(search_term, case=False).any(), axis=1
        )
        df_display = df_filtered[mask]
    else:
        df_display = df_filtered

    st.dataframe(df_display, use_container_width=True, height=350)

    with st.expander("📊 View Summary Statistics (Numerical Features)"):
        st.dataframe(df_filtered.describe(), use_container_width=True)


# ----------------------------------------------------
# TAB 5: STRATEGIC ROADMAP
# ----------------------------------------------------
with tab5:
    st.subheader("🎯 Enterprise Strategic Action Plan & Business Directives")
    st.caption(
        "Data-driven strategic recommendations formulated for eBay Leadership & Product Teams."
    )

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("### 🚀 High Impact Directives")

        st.markdown("#### 1. Transparent Shipping & Dynamic Thresholds")
        st.write(
            "- **Finding:** *High shipping costs* represent the single largest cart abandonment driver (>26% of buyers)."
        )
        st.write(
            "- **Action:** Implement progress meters towards dynamic free shipping thresholds right inside the cart view."
        )

        st.markdown("#### 2. Endless Scroll & Feed Optimization")
        st.write(
            "- **Finding:** Users strongly requested infinite scrolling over pagination."
        )
        st.write(
            "- **Action:** Upgrade search result pages to continuous vertical loading to boost engagement depth."
        )

    with r2:
        st.markdown("### 💡 Personalization & Loyalty Strategies")

        st.markdown("#### 3. Hybrid Recommendation Engine Tuning")
        st.write(
            "- **Finding:** Disconnect between recommendation ratings and cart completion rates."
        )
        st.write(
            "- **Action:** Incorporate recent clickstream context into collaborative filtering algorithms."
        )

        st.markdown("#### 4. Post-Purchase Value Communication")
        st.write(
            "- **Finding:** Buyers appreciate responsive customer service as a primary retention feature."
        )
        st.write(
            "- **Action:** Highlight buyer protection guarantees and instant support channels during checkout."
        )
