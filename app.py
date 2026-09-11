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
# 2. SAFE DATA LOADING & COMPLETE DEDUPLICATION
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
                    [18, 22, 28, 35, 42, 50, 61, 67], size=n
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
            }
        )

    # 1. Clean space from columns
    df.columns = df.columns.str.strip()

    # 2. STRICT DUPLICATE REMOVAL (Position based)
    df = df.loc[:, ~df.columns.duplicated(keep="first")].copy()

    # 3. Rename columns using mapping safely
    col_map = {}
    seen_mapped = set()

    for col in df.columns:
        c_lower = col.lower().replace(" ", "_").replace("-", "_")

        target = None
        if "age" in c_lower:
            target = "age"
        elif "gender" in c_lower:
            target = "Gender"
        elif "freq" in c_lower or "cadence" in c_lower or "purchase" in c_lower:
            target = "Purchase_Frequency"
        elif "satis" in c_lower or "rating" in c_lower:
            target = "Shopping_Satisfaction"
        elif "abandon" in c_lower or "reason" in c_lower:
            target = "Abandonment_Reason"
        elif "search" in c_lower or "method" in c_lower:
            target = "Search_Method"
        elif "depth" in c_lower or "explore" in c_lower:
            target = "Exploration_Depth"
        elif "feature" in c_lower or "apprec" in c_lower:
            target = "Appreciated_Feature"
        elif "improve" in c_lower or "area" in c_lower:
            target = "Improvement_Area"

        if target and target not in seen_mapped:
            col_map[col] = target
            seen_mapped.add(target)

    df = df.rename(columns=col_map)

    # 4. Filter to keep only target clean columns
    keep_cols = [
        "age",
        "Gender",
        "Purchase_Frequency",
        "Shopping_Satisfaction",
        "Abandonment_Reason",
        "Search_Method",
        "Exploration_Depth",
        "Appreciated_Feature",
        "Improvement_Area",
    ]

    # Fill missing expected columns with default dummy series
    defaults = {
        "age": 25,
        "Gender": "Not Specified",
        "Purchase_Frequency": "Once a month",
        "Shopping_Satisfaction": 3,
        "Abandonment_Reason": "High shipping costs",
        "Search_Method": "Keyword",
        "Exploration_Depth": "First page",
        "Appreciated_Feature": "User-friendly interface",
        "Improvement_Area": "User interface",
    }

    for c in keep_cols:
        if c not in df.columns:
            df[c] = defaults[c]

    # Final explicit column selection & copy to eliminate Narwhals duplicate issues
    df = df[keep_cols].copy()

    # Data types cleanup
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df = df[(df["age"] >= 13) & (df["age"] <= 80)].copy()

    df["Shopping_Satisfaction"] = pd.to_numeric(
        df["Shopping_Satisfaction"], errors="coerce"
    ).fillna(3)

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

    all_genders = list(df_raw["Gender"].dropna().astype(str).unique())
    selected_gender = st.multiselect(
        "👥 Filter Gender:", options=all_genders, default=all_genders
    )

    min_age_val = (
        int(df_raw["age"].min()) if not df_raw["age"].empty else 13
    )
    max_age_val = (
        int(df_raw["age"].max()) if not df_raw["age"].empty else 70
    )

    selected_age = st.slider(
        "🎂 Select Age Range:",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val),
    )

    all_cadence = list(
        df_raw["Purchase_Frequency"].dropna().astype(str).unique()
    )
    selected_cadence = st.multiselect(
        "🛍️ Purchase Cadence:", options=all_cadence, default=all_cadence
    )

# Filtering logic
df_filtered = df_raw[
    (df_raw["Gender"].isin(selected_gender))
    & (df_raw["age"].between(selected_age[0], selected_age[1]))
    & (df_raw["Purchase_Frequency"].isin(selected_cadence))
].copy()

if df_filtered.empty:
    st.warning("⚠️ Selected filters ka koi data nahi mila. Reset filters.")
    st.stop()

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("🛒 eBay Customer Behavior & ML Insights Dashboard")
st.caption("Executive Analytics Portal & Customer Segmentation")
st.markdown(" ")

# KPI Row
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
    mode_ab = df_filtered["Abandonment_Reason"].mode()
    top_ab = mode_ab[0] if not mode_ab.empty else "N/A"
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
    mode_srch = df_filtered["Search_Method"].mode()
    top_srch = mode_srch[0] if not mode_srch.empty else "N/A"
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
        # Pass explicit copy with clean columns to Plotly Express
        df_hist = df_filtered[["age", "Gender"]].copy()
        fig_age = px.histogram(
            df_hist,
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
            .reset_index(name="Count")
        )
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
            .reset_index(name="Count")
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
        srch_cnt = (
            df_filtered["Search_Method"]
            .value_counts()
            .reset_index(name="Count")
        )
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
    X = df_filtered[num_cols].dropna().copy()

    if len(X) >= 3:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        df_clustered = X.copy()
        df_clustered["Cluster"] = [f"Cluster {c+1}" for c in clusters]

        fig_cluster = px.scatter(
            df_clustered,
            x="age",
            y="Shopping_Satisfaction",
            color="Cluster",
            title="Customer Clustering (Age vs Satisfaction)",
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        st.info("Insufficient data points for ML Clustering.")

with tab4:
    st.dataframe(df_filtered, use_container_width=True)        font-weight: 600;
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
# 2. SAFE DATA LOADING & DUPLICATE COLUMN CLEANUP
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

    # Clean whitespace in column names
    df.columns = df.columns.str.strip()

    # FIX: Remove Duplicate Columns if CSV has duplicate names
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Smart Matching Dictionary (Matches original CSV column names safely)
    col_map = {}
    for col in df.columns:
        c_lower = col.lower().replace(" ", "_").replace("-", "_")
        if "age" in c_lower and "age" not in col_map.values():
            col_map[col] = "age"
        elif "gender" in c_lower and "Gender" not in col_map.values():
            col_map[col] = "Gender"
        elif (
            ("freq" in c_lower or "cadence" in c_lower or "purchase" in c_lower)
            and "Purchase_Frequency" not in col_map.values()
        ):
            col_map[col] = "Purchase_Frequency"
        elif (
            ("satis" in c_lower or "rating" in c_lower)
            and "Shopping_Satisfaction" not in col_map.values()
        ):
            col_map[col] = "Shopping_Satisfaction"
        elif (
            ("abandon" in c_lower or "reason" in c_lower)
            and "Abandonment_Reason" not in col_map.values()
        ):
            col_map[col] = "Abandonment_Reason"
        elif (
            ("search" in c_lower or "method" in c_lower)
            and "Search_Method" not in col_map.values()
        ):
            col_map[col] = "Search_Method"
        elif (
            ("depth" in c_lower or "explore" in c_lower)
            and "Exploration_Depth" not in col_map.values()
        ):
            col_map[col] = "Exploration_Depth"
        elif (
            ("feature" in c_lower or "apprec" in c_lower)
            and "Appreciated_Feature" not in col_map.values()
        ):
            col_map[col] = "Appreciated_Feature"
        elif (
            ("improve" in c_lower or "area" in c_lower)
            and "Improvement_Area" not in col_map.values()
        ):
            col_map[col] = "Improvement_Area"

    df = df.rename(columns=col_map)

    # Ensure required columns exist
    defaults = {
        "age": 25,
        "Gender": "Not Specified",
        "Purchase_Frequency": "Once a month",
        "Shopping_Satisfaction": 3,
        "Abandonment_Reason": "High shipping costs",
        "Search_Method": "Keyword",
        "Exploration_Depth": "First page",
        "Appreciated_Feature": "User-friendly interface",
        "Improvement_Area": "User interface",
    }

    for c, val in defaults.items():
        if c not in df.columns:
            df[c] = val

    # Helper function to extract 1D Series safely
    def get_series(dataframe, col_name):
        res = dataframe[col_name]
        if isinstance(res, pd.DataFrame):
            res = res.iloc[:, 0]
        return res

    # Preprocessing
    age_series = pd.to_numeric(get_series(df, "age"), errors="coerce")
    df["age"] = age_series
    df = df[(df["age"] >= 13) & (df["age"] <= 80)].copy()

    app_feat = get_series(df, "Appreciated_Feature").replace(
        {"Unknown": "Not Specified", np.nan: "Not Specified"}
    )
    df["Appreciated_Feature"] = app_feat

    imp_area = get_series(df, "Improvement_Area").replace(
        {"Unknown": "Not Specified", np.nan: "Not Specified"}
    )
    df["Improvement_Area"] = imp_area

    return df


df_raw = load_data()


# Helper function to get safe unique list from Series
def get_unique_list(df, col_name):
    val = df[col_name]
    if isinstance(val, pd.DataFrame):
        val = val.iloc[:, 0]
    return list(val.dropna().astype(str).unique())


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

    all_genders = get_unique_list(df_raw, "Gender")
    selected_gender = st.multiselect(
        "👥 Filter Gender:", options=all_genders, default=all_genders
    )

    age_col = (
        df_raw["age"].iloc[:, 0]
        if isinstance(df_raw["age"], pd.DataFrame)
        else df_raw["age"]
    )
    min_age_val = int(age_col.min()) if not age_col.empty else 13
    max_age_val = int(age_col.max()) if not age_col.empty else 70

    selected_age = st.slider(
        "🎂 Select Age Range:",
        min_value=min_age_val,
        max_value=max_age_val,
        value=(min_age_val, max_age_val),
    )

    all_cadence = get_unique_list(df_raw, "Purchase_Frequency")
    selected_cadence = st.multiselect(
        "🛍️ Purchase Cadence:", options=all_cadence, default=all_cadence
    )

# Filtering logic
gender_mask = (
    df_raw["Gender"].iloc[:, 0]
    if isinstance(df_raw["Gender"], pd.DataFrame)
    else df_raw["Gender"]
).isin(selected_gender)
age_mask = (
    df_raw["age"].iloc[:, 0]
    if isinstance(df_raw["age"], pd.DataFrame)
    else df_raw["age"]
).between(selected_age[0], selected_age[1])
cadence_mask = (
    df_raw["Purchase_Frequency"].iloc[:, 0]
    if isinstance(df_raw["Purchase_Frequency"], pd.DataFrame)
    else df_raw["Purchase_Frequency"]
).isin(selected_cadence)

df_filtered = df_raw[gender_mask & age_mask & cadence_mask].copy()

if df_filtered.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ==========================================
# 4. MAIN DASHBOARD UI
# ==========================================
st.title("🛒 eBay Customer Behavior & ML Insights Dashboard")
st.caption("Executive Analytics Portal & Customer Segmentation")
st.markdown(" ")


def safe_mode(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    m = s.mode()
    return m[0] if not m.empty else "N/A"


def safe_mean(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.mean()


# KPI Row
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Total Buyers</div><div class="kpi-value">{len(df_filtered)}</div><div class="kpi-sub">Cohort Count</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    avg_sat = round(safe_mean(df_filtered, "Shopping_Satisfaction"), 2)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Avg Satisfaction</div><div class="kpi-value">{avg_sat} / 5</div><div class="kpi-sub">Rating Score</div></div>""",
        unsafe_allow_html=True,
    )
with k3:
    top_ab = safe_mode(df_filtered, "Abandonment_Reason")
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Top Friction</div><div class="kpi-value" style="font-size:15px !important;">{top_ab}</div><div class="kpi-sub">Primary Reason</div></div>""",
        unsafe_allow_html=True,
    )
with k4:
    avg_age = round(safe_mean(df_filtered, "age"), 1)
    st.markdown(
        f"""<div class="kpi-card"><div class="kpi-title">Average Age</div><div class="kpi-value">{avg_age} Yrs</div><div class="kpi-sub">Filtered Group</div></div>""",
        unsafe_allow_html=True,
    )
with k5:
    top_srch = safe_mode(df_filtered, "Search_Method")
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
        pf = (
            df_filtered["Purchase_Frequency"].iloc[:, 0]
            if isinstance(df_filtered["Purchase_Frequency"], pd.DataFrame)
            else df_filtered["Purchase_Frequency"]
        )
        cad_cnt = pf.value_counts().reset_index()
        cad_cnt.columns = ["Cadence", "Count"]
        fig_cad = px.pie(
            cad_cnt, names="Cadence", values="Count", title="Purchase Cadence"
        )
        st.plotly_chart(fig_cad, use_container_width=True)

with tab2:
    f1, f2 = st.columns(2)
    with f1:
        ab = (
            df_filtered["Abandonment_Reason"].iloc[:, 0]
            if isinstance(df_filtered["Abandonment_Reason"], pd.DataFrame)
            else df_filtered["Abandonment_Reason"]
        )
        ab_cnt = ab.value_counts().reset_index()
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
        sm = (
            df_filtered["Search_Method"].iloc[:, 0]
            if isinstance(df_filtered["Search_Method"], pd.DataFrame)
            else df_filtered["Search_Method"]
        )
        srch_cnt = sm.value_counts().reset_index()
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
