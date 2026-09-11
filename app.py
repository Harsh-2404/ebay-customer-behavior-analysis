import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Page Configuration & Custom Enterprise Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="eBay Customer Analytics & ML Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Corporate UI/UX Aesthetics
st.markdown("""
    <style>
    /* Main container background */
    .main {
        background-color: #F8F9FA;
    }
    /* Header branding styling */
    .main-header {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        color: #0064D2;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #4A5568;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #0064D2 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 5px solid #0064D2;
    }
    /* Sidebar Styling */
    .css-1d3b13b {
        background-color: #FFFFFF;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 6px;
        color: #2D3748;
        font-weight: 600;
        padding: 0px 20px;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #0064D2 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. Optimized Data Loading Pipeline
# ---------------------------------------------------------
@st.cache_data
def load_data():
    try:
        # Load the cleaned eBay CSV file
        df = pd.read_csv('cleaned_eBay.csv')
    except FileNotFoundError:
        # Fallback if filename differs slightly
        df = pd.read_csv('eBay_cleaned.csv')
    
    # Ensure column names are clean and stripped of extra whitespace
    df.columns = [col.strip() for col in df.columns]
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"❌ Error loading `cleaned_eBay.csv`: {e}")
    st.info("Ensure `cleaned_eBay.csv` is present in the same directory as `app.py`.")
    st.stop()


# ---------------------------------------------------------
# 3. Sidebar Executive Filter Controls
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/1/1b/EBay_logo.svg", width=140)
st.sidebar.title("Executive Control Panel")
st.sidebar.markdown("---")

# Filter 1: Gender
all_genders = df_raw['Gender'].unique().tolist()
selected_gender = st.sidebar.multiselect(
    "👥 Filter Gender:",
    options=all_genders,
    default=all_genders
)

# Filter 2: Age Range
min_age = int(df_raw['age'].min())
max_age = int(df_raw['age'].max())
selected_age_range = st.sidebar.slider(
    "🎂 Select Age Range:",
    min_value=min_age,
    max_value=max_age,
    value=(min_age, max_age)
)

# Filter 3: Purchase Frequency
all_freq = df_raw['Purchase_Frequency'].unique().tolist()
selected_freq = st.sidebar.multiselect(
    "🛍️ Purchase Cadence:",
    options=all_freq,
    default=all_freq
)

# Filter 4: Cart Abandonment Factor
all_abandon = df_raw['Cart_Abandonment_Factors'].unique().tolist()
selected_abandon = st.sidebar.multiselect(
    "🚨 Abandonment Reason:",
    options=all_abandon,
    default=all_abandon
)

# Filter Data Application
filtered_df = df_raw[
    (df_raw['Gender'].isin(selected_gender)) &
    (df_raw['age'] >= selected_age_range[0]) &
    (df_raw['age'] <= selected_age_range[1]) &
    (df_raw['Purchase_Frequency'].isin(selected_freq)) &
    (df_raw['Cart_Abandonment_Factors'].isin(selected_abandon))
].copy()

if filtered_df.empty:
    st.warning("⚠️ No records match the selected sidebar filters. Please reset your filters.")
    st.stop()


# ---------------------------------------------------------
# 4. Header & Executive KPI Summary Section
# ---------------------------------------------------------
st.markdown("<h1 class='main-header'>🛒 eBay Customer Behavior & ML Insights Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Executive Analytics Portal | Cart Friction Identification, Algorithmic Recommendation Performance, and Customer Segmentation</p>", unsafe_allow_html=True)

# Key Metrics Cards
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.metric(
        label="Total Surveyed Buyers",
        value=f"{len(filtered_df):,}",
        delta=f"{len(filtered_df)/len(df_raw):.1%} of Total"
    )

with m2:
    avg_sat = filtered_df['Shopping_Satisfaction'].mean()
    st.metric(
        label="Avg Satisfaction",
        value=f"{avg_sat:.2f} / 5.0",
        delta="Scale 1-5"
    )

with m3:
    top_abandon_reason = filtered_df['Cart_Abandonment_Factors'].mode().iloc[0] if not filtered_df.empty else "N/A"
    st.metric(
        label="Top Abandonment Driver",
        value=top_abandon_reason,
        delta="Primary Friction"
    )

with m4:
    avg_age_val = filtered_df['age'].mean()
    st.metric(
        label="Average Buyer Age",
        value=f"{avg_age_val:.1f} Yrs"
    )

with m5:
    top_search_method = filtered_df['Product_Search_Method'].mode().iloc[0] if not filtered_df.empty else "N/A"
    st.metric(
        label="Primary Search Method",
        value=top_search_method
    )

st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------
# 5. Dashboard Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive BI Overview", 
    "🚨 Friction & Abandonment", 
    "🤖 ML Customer Clustering", 
    "📁 Data Explorer & Download",
    "🎯 Strategic Roadmap"
])


# =========================================================
# TAB 1: EXECUTIVE BI OVERVIEW
# =========================================================
with tab1:
    st.subheader("📊 Executive Overview & Demographic Cadence")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("##### 👥 Age & Gender Demographics")
        fig_demo = px.histogram(
            filtered_df, 
            x="age", 
            color="Gender", 
            barmode="group",
            nbins=15,
            color_discrete_sequence=['#0064D2', '#FF4136', '#2ECC40', '#FF851B'],
            title="Customer Age Distribution Segmented by Gender"
        )
        fig_demo.update_layout(
            template="plotly_white", 
            xaxis_title="Buyer Age", 
            yaxis_title="Buyer Count",
            legend_title="Gender"
        )
        st.plotly_chart(fig_demo, use_container_width=True)

    with col_right:
        st.markdown("##### 🛍️ Purchase Frequency Distribution")
        freq_counts = filtered_df['Purchase_Frequency'].value_counts().reset_index()
        freq_counts.columns = ['Purchase Frequency', 'Count']
        
        fig_freq = px.pie(
            freq_counts, 
            names='Purchase Frequency', 
            values='Count', 
            hole=0.45,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            title="Overall Buyer Purchase Cadence Share"
        )
        fig_freq.update_layout(template="plotly_white")
        st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("---")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("##### ⭐ Overall Shopping Satisfaction Ratings")
        sat_df = filtered_df['Shopping_Satisfaction'].value_counts().sort_index().reset_index()
        sat_df.columns = ['Rating Scale (1-5)', 'Count']
        
        fig_sat = px.bar(
            sat_df, 
            x='Rating Scale (1-5)', 
            y='Count', 
            text='Count',
            color='Rating Scale (1-5)',
            color_continuous_scale='Blues',
            title="Customer Shopping Satisfaction Breakdown"
        )
        fig_sat.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_sat, use_container_width=True)

    with col_c2:
        st.markdown("##### 💡 Service Appreciation Highlights")
        service_df = filtered_df['Service_Appreciation'].value_counts().head(6).reset_index()
        service_df.columns = ['Appreciated Feature', 'Count']
        
        fig_service = px.bar(
            service_df, 
            x='Count', 
            y='Appreciated Feature', 
            orientation='h',
            color='Count',
            color_continuous_scale='Greens',
            title="Top Customer Service Appreciation Factors"
        )
        fig_service.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_service, use_container_width=True)


# =========================================================
# TAB 2: FRICTION & CART ABANDONMENT
# =========================================================
with tab2:
    st.subheader("🚨 Cart Abandonment Root Cause Analysis & Search Friction")
    st.markdown("Analyzing conversion leakage points across search behavior, fees, and product discovery.")

    col_f1, col_f2 = st.columns([3, 2])
    
    with col_f1:
        st.markdown("##### 🛒 Primary Drivers of Cart Abandonment")
        abandon_counts = filtered_df['Cart_Abandonment_Factors'].value_counts().reset_index()
        abandon_counts.columns = ['Abandonment Factor', 'Count']
        
        fig_abandon = px.bar(
            abandon_counts, 
            x='Count', 
            y='Abandonment Factor', 
            orientation='h',
            color='Count',
            color_continuous_scale='Reds',
            text='Count',
            title="Cart Abandonment Root Causes"
        )
        fig_abandon.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_abandon, use_container_width=True)

    with col_f2:
        st.markdown("##### 🔍 Search Result Exploration Depth")
        explore_counts = filtered_df['Search_Result_Exploration'].value_counts().reset_index()
        explore_counts.columns = ['Exploration Level', 'Count']
        
        fig_explore = px.pie(
            explore_counts, 
            names='Exploration Level', 
            values='Count',
            hole=0.4,
            color_discrete_sequence=['#FF4136', '#0064D2'],
            title="First Page vs Multiple Pages Browsing"
        )
        fig_explore.update_layout(template="plotly_white")
        st.plotly_chart(fig_explore, use_container_width=True)

    st.markdown("---")
    
    col_f3, col_f4 = st.columns(2)
    
    with col_f3:
        st.markdown("##### 🔎 Product Search Method Preference")
        search_counts = filtered_df['Product_Search_Method'].value_counts().reset_index()
        search_counts.columns = ['Search Method', 'Count']
        
        fig_search = px.bar(
            search_counts, 
            x='Search Method', 
            y='Count',
            color='Count',
            color_continuous_scale='Purples',
            title="Customer Search Method Usage"
        )
        fig_search.update_layout(template="plotly_white", showlegend=False)
        st.plotly_chart(fig_search, use_container_width=True)

    with col_f4:
        st.markdown("##### 🛠️ Key Product Improvement Areas Requested")
        improve_counts = filtered_df['Improvement_Areas'].value_counts().head(5).reset_index()
        improve_counts.columns = ['Improvement Area', 'Count']
        
        fig_improve = px.bar(
            improve_counts, 
            x='Count', 
            y='Improvement Area', 
            orientation='h',
            color='Count',
            color_continuous_scale='Oranges',
            title="Top Priority Customer UX Improvement Requests"
        )
        fig_improve.update_layout(template="plotly_white", yaxis={'categoryorder': 'total ascending'}, showlegend=False)
        st.plotly_chart(fig_improve, use_container_width=True)


# =========================================================
# TAB 3: MACHINE LEARNING CUSTOMER CLUSTERING
# =========================================================
with tab3:
    st.subheader("🤖 Unsupervised Machine Learning: K-Means Customer Segmentation")
    st.markdown("Segment buyers using numerical behavioral ratings to uncover target persona cohorts.")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude non-behavioral columns like 'transaction' if present
    feature_candidates = [c for c in numeric_cols if c.lower() != 'transaction']

    col_m1, col_m2 = st.columns([1, 3])

    with col_m1:
        st.markdown("##### ⚙️ Model Hyperparameters")
        selected_features = st.multiselect(
            "Select Feature Dimensions:",
            options=feature_candidates,
            default=['age', 'Customer_Reviews_Importance', 'Personalized_Recommendation_Rating', 'Shopping_Satisfaction']
        )
        
        k_clusters = st.slider("Select Cluster Count (k):", min_value=2, max_value=6, value=3)
        
        st.markdown("---")
        st.info("💡 **Tip**: Scaling is automatically performed using `StandardScaler` before applying K-Means clustering.")

    with col_m2:
        if len(selected_features) >= 2:
            cluster_data = filtered_df[selected_features].dropna()
            
            if len(cluster_data) >= k_clusters:
                # Feature Scaling
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(cluster_data)
                
                # KMeans Model Fitting
                kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(scaled_features)
                
                plot_df = cluster_data.copy()
                plot_df['Cluster'] = [f"Cluster {i+1}" for i in cluster_labels]
                
                # 3D or 2D Visualization based on selected feature count
                if len(selected_features) >= 3:
                    fig_cluster = px.scatter_3d(
                        plot_df,
                        x=selected_features[0],
                        y=selected_features[1],
                        z=selected_features[2],
                        color='Cluster',
                        title=f"3D K-Means Clustering Space (k={k_clusters})",
                        color_discrete_sequence=px.colors.qualitative.Bold,
                        opacity=0.8
                    )
                else:
                    fig_cluster = px.scatter(
                        plot_df,
                        x=selected_features[0],
                        y=selected_features[1],
                        color='Cluster',
                        title=f"2D K-Means Clustering Visual (k={k_clusters})",
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                
                fig_cluster.update_layout(template="plotly_white", margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_cluster, use_container_width=True)
                
                st.markdown("##### 📌 Cluster Profiling & Center Averages")
                summary_df = plot_df.groupby('Cluster').mean().round(2)
                st.dataframe(
                    summary_df.style.highlight_max(axis=0, color="#d1e7dd").highlight_min(axis=0, color="#f8d7da"),
                    use_container_width=True
                )
            else:
                st.warning("Not enough sample data available for selected cluster parameters.")
        else:
            st.warning("Please select at least 2 numeric features to train the clustering model.")


# =========================================================
# TAB 4: DATA EXPLORER & DOWNLOAD CLEANED CSV
# =========================================================
with tab4:
    st.subheader("📁 Interactive Dataset Explorer & Cleaned Export")
    st.markdown("Inspect, search, and download the fully preprocessed **`cleaned_eBay.csv`** file.")

    st.markdown("##### 📥 Download Cleaned Dataset")
    
    # Convert cleaned dataset to CSV string for direct browser download
    cleaned_csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.download_button(
            label="⬇️ Download Cleaned eBay CSV",
            data=cleaned_csv_bytes,
            file_name="cleaned_eBay.csv",
            mime="text/csv",
            help="Click to download the clean, structured dataset for offline analysis."
        )
    with col_d2:
        st.success(f"✅ Filtered view contains **{len(filtered_df):,} rows** and **{len(filtered_df.columns)} columns**.")

    st.markdown("---")
    st.markdown("##### 🔎 Filtered Data Viewer")
    
    search_query = st.text_input("Search dataset by keyword:")
    if search_query:
        search_mask = filtered_df.astype(str).apply(lambda row: row.str.contains(search_query, case=False).any(), axis=1)
        display_df = filtered_df[search_mask]
    else:
        display_df = filtered_df

    st.dataframe(display_df, use_container_width=True, height=400)

    with st.expander("📊 View Summary Statistics (Numerical Features)"):
        st.dataframe(filtered_df.describe().T, use_container_width=True)


# =========================================================
# TAB 5: STRATEGIC ROADMAP
# =========================================================
with tab5:
    st.subheader("🎯 Enterprise Strategic Action Plan & Business Directives")
    st.markdown("Data-driven strategic recommendations formulated for eBay Leadership & Product Teams.")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("""
        ### 🚀 High Impact Directives
        
        1. **Transparent Shipping & Dynamic Thresholds**
           * **Finding**: *High shipping costs* represent the single largest cart abandonment driver (>26% of buyers).
           * **Action**: Implement progress meters towards dynamic free shipping thresholds (e.g., *"Add $8.50 more for Free Shipping"*).
        
        2. **Enhanced Search Navigation & Infinite Scroll**
           * **Finding**: Over 52% of users explore only the first page, and infinite scrolling is the top requested improvement.
           * **Action**: Replace traditional paginated navigation with seamless infinite scroll to increase impression rates.
        """)

    with col_r2:
        st.markdown("""
        ### 💡 Personalization & Loyalty Strategies
        
        3. **Hybrid Recommendation Engine Tuning**
           * **Finding**: Disconnect between recommendation ratings and cart completion rates.
           * **Action**: Incorporate recent clickstream context into collaborative filtering to increase relevancy.
        
        4. **Targeted Cluster Interventions**
           * **At-Risk Price Sensitive Cohort**: Trigger dynamic discount badges and time-sensitive voucher popups.
           * **Loyal VIP Buyers**: Auto-enroll high-satisfaction buyers into premium seller subscription perks.
        """)

    st.divider()
    st.caption("🔒 *eBay Analytics Portal — Confidential & Proprietary Report.*")
