import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Online Retail Sales Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Set visual style for Seaborn plots
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------
# Data Loading & Cleaning Function
# ---------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv('data/OnlineRetail.csv', encoding='ISO-8859-1')
    
    # Data cleaning steps
    df.drop_duplicates(inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Remove null CustomerIDs and keep positive quantities/prices
    df_clean = df.dropna(subset=['CustomerID']).copy()
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    # Feature engineering
    df_clean['TotalRevenue'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M').astype(str)
    
    return df_clean

# Try loading the data
try:
    df = load_and_clean_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}. Please ensure 'OnlineRetail.csv' is inside the 'data/' directory.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.title("📌 Dashboard Controls")

# Country Filter
countries = ["All"] + list(df['Country'].unique())
selected_country = st.sidebar.selectbox("Select Country:", countries)

# Date Filter
min_date = df['InvoiceDate'].min().date()
max_date = df['InvoiceDate'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Handle partial user clicks cleanly
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

# Filter Data Based on User Input
filtered_df = df[(df['InvoiceDate'].dt.date >= start_date) & (df['InvoiceDate'].dt.date <= end_date)]
if selected_country != "All":
    filtered_df = filtered_df[filtered_df['Country'] == selected_country]

# ---------------------------------------------------------
# Navigation
# ---------------------------------------------------------
page = st.sidebar.radio("Navigate", [
    "📊 Overview & Metrics", 
    "📈 Exploratory Visualizations",
    "🎯 Customer Segmentation (RFM)",
    "🔢 Basic Statistics", 
    "💡 Key Findings & Recommendations"
])

# ---------------------------------------------------------
# Page 1: Overview & Metrics
# ---------------------------------------------------------
if page == "📊 Overview & Metrics":
    st.title("🛍️ Online Retail Data Analysis Dashboard")
    st.markdown("An interactive analysis dashboard for the Kaggle/UCI Online Retail Dataset.")
    st.write("---")

    # Key Performance Indicators (KPIs)
    total_revenue = filtered_df['TotalRevenue'].sum()
    total_orders = filtered_df['InvoiceNo'].nunique()
    total_customers = filtered_df['CustomerID'].nunique()
    avg_order_val = total_revenue / total_orders if total_orders > 0 else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Revenue", f"${total_revenue:,.2f}")
    kpi2.metric("Total Orders", f"{total_orders:,}")
    kpi3.metric("Unique Customers", f"{total_customers:,}")
    kpi4.metric("Avg Order Value", f"${avg_order_val:,.2f}")

    st.write("---")
    st.subheader("📋 Filtered Dataset Preview")
    st.dataframe(filtered_df.head(100), use_container_width=True)

# ---------------------------------------------------------
# Page 2: Visualizations
# ---------------------------------------------------------
elif page == "📈 Exploratory Visualizations":
    st.title("📈 Visual Data Analysis")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Sales Revenue Trend")
        monthly_sales = filtered_df.groupby('YearMonth')['TotalRevenue'].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=monthly_sales, x='YearMonth', y='TotalRevenue', marker='o', ax=ax1, color='b')
        plt.xticks(rotation=45)
        ax1.set_ylabel("Revenue ($)")
        st.pyplot(fig1)

    with col2:
        st.subheader("Top 5 Products by Revenue")
        top_products = filtered_df.groupby('Description')['TotalRevenue'].sum().nlargest(5).reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_products, y='Description', x='TotalRevenue', palette='Greens_r', ax=ax2)
        ax2.set_xlabel("Revenue ($)")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 5 Countries by Revenue")
        top_countries = filtered_df.groupby('Country')['TotalRevenue'].sum().nlargest(5).reset_index()
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        sns.barplot(data=top_countries, x='TotalRevenue', y='Country', palette='Blues_r', ax=ax3)
        ax3.set_xlabel("Revenue ($)")
        st.pyplot(fig3)

    with col4:
        st.subheader("Revenue Distribution Per Transaction")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.histplot(data=filtered_df, x='TotalRevenue', bins=50, kde=True, color='purple', ax=ax4)
        ax4.set_xlim(0, 100)  # Cropped for visibility
        ax4.set_xlabel("Revenue per Line Item ($)")
        st.pyplot(fig4)

# ---------------------------------------------------------
# Page 3: Customer Segmentation (RFM)
# ---------------------------------------------------------
elif page == "🎯 Customer Segmentation (RFM)":
    st.title("🎯 Customer Segmentation (RFM Analysis)")
    st.markdown("Group customers into behavioral segments based on Recency, Frequency, and Monetary value.")

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Calculate RFM Metrics
        snapshot_date = filtered_df['InvoiceDate'].max() + dt.timedelta(days=1)
        
        rfm = filtered_df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalRevenue': 'sum'
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

        # Quantile scoring (1-4 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=[4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=[1, 2, 3, 4])

        # Assign segments
        def assign_segment(x):
            if x['R_Score'] == 4 and x['F_Score'] == 4 and x['M_Score'] == 4:
                return 'Champions'
            elif x['F_Score'] >= 3 and x['M_Score'] >= 3:
                return 'Loyal Customers'
            elif x['R_Score'] >= 3 and x['F_Score'] <= 2:
                return 'Recent / New Customers'
            elif x['R_Score'] <= 2 and x['F_Score'] >= 3:
                return 'At Risk / Need Attention'
            else:
                return 'Lost / Inactive'

        rfm['Segment'] = rfm.apply(assign_segment, axis=1)

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Customer Count by Segment")
            segment_counts = rfm['Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            
            fig_rfm1, ax_rfm1 = plt.subplots(figsize=(8, 4))
            sns.barplot(data=segment_counts, y='Segment', x='Count', palette='Blues_r', ax=ax_rfm1)
            ax_rfm1.set_xlabel("Number of Customers")
            st.pyplot(fig_rfm1)

        with col2:
            st.subheader("Total Revenue by Segment")
            segment_rev = rfm.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).reset_index()
            
            fig_rfm2, ax_rfm2 = plt.subplots(figsize=(8, 4))
            sns.barplot(data=segment_rev, y='Segment', x='Monetary', palette='Greens_r', ax=ax_rfm2)
            ax_rfm2.set_xlabel("Revenue ($)")
            st.pyplot(fig_rfm2)

        st.write("---")
        st.subheader("📋 Segment Sample View")
        st.dataframe(rfm.head(100), use_container_width=True)

# ---------------------------------------------------------
# Page 4: Basic Statistics
# ---------------------------------------------------------
elif page == "🔢 Basic Statistics":
    st.title("🔢 Statistical Summary")
    st.markdown("Descriptive statistics for numerical variables in the retail dataset.")

    num_cols = filtered_df[['Quantity', 'UnitPrice', 'TotalRevenue']]

    st.subheader("Summary Metrics")
    st.dataframe(num_cols.describe(percentiles=[0.25, 0.50, 0.75, 0.90]), use_container_width=True)

    st.subheader("Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 3))
    sns.heatmap(num_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

# ---------------------------------------------------------
# Page 5: Findings & Recommendations
# ---------------------------------------------------------
elif page == "💡 Key Findings & Recommendations":
    st.title("💡 Insights & Recommendations")

    st.subheader("📌 Key Findings")
    st.markdown("""
    * **Seasonal Peak:** Revenue increases sharply in Q4, peaking in November driven by holiday purchases.
    * **Geographic Dominance:** The majority of transactions and revenue originate within the United Kingdom.
    * **Top Revenue Performers:** A small concentration of high-volume giftware products drives a significant share of revenue.
    * **Customer Base Concentration:** RFM segmentation highlights that **Champions** and **Loyal Customers** generate the highest revenue share despite representing a smaller fraction of overall users.
    * **At Risk Revenue:** A notable portion of historical revenue belongs to the **At Risk** segment, representing accounts with zero recent orders.
    """)

    st.subheader("🚀 Business Recommendations")
    st.markdown("""
    1. **Q4 Inventory Preparation:** Increase stock levels and logistical capacity by late September to capture holiday surge demand.
    2. **VIP Retention Program:** Create exclusive loyalty benefits for **Champions** to maintain long-term retention.
    3. **Win-Back Campaigns:** Deploy targeted email automation with localized promotional discounts aimed at the **At Risk** segment.
    4. **International Marketing:** Explore targeted promotions in secondary high-volume countries (e.g., Germany, France) to diversify revenue.
    """)