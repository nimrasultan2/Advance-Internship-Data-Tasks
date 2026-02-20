import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Page config 
st.set_page_config(page_title="Global Superstore Dashboard", layout="wide")
st.title(" Global Superstore — Sales Dashboard")

# Load data 
def load_data():
    df = pd.read_csv("Superstore.csv")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df
df = load_data()

# Sidebar Filters 
st.sidebar.header("Filters")
regions = st.sidebar.multiselect("Region", options=df['Region'].unique(), default=df['Region'].unique())
categories = st.sidebar.multiselect("Category", options=df['Category'].unique(), default=df['Category'].unique())
subcategories = st.sidebar.multiselect("Sub-Category", options=df['Sub-Category'].unique(), default=df['Sub-Category'].unique())

# Apply filters
filtered = df[
    df['Region'].isin(regions) &
    df['Category'].isin(categories) &
    df['Sub-Category'].isin(subcategories)
]

# KPI Cards 
st.subheader("Key Performance Indicators")
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${filtered['Sales'].sum():,.2f}")
col2.metric("Total Profit", f"${filtered['Profit'].sum():,.2f}")
col3.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
st.divider()

# Charts Row 
col_left, col_right = st.columns(2)

# Sales by Region
with col_left:
    st.subheader("Sales by Region")
    region_sales = filtered.groupby('Region')['Sales'].sum().sort_values(ascending=False)
    fig1, ax1 = plt.subplots()
    ax1.bar(region_sales.index, region_sales.values, color='steelblue')
    ax1.set_ylabel("Sales ($)")
    ax1.set_xlabel("Region")
    st.pyplot(fig1)

# Sales by Category
with col_right:
    st.subheader("Sales by Category")
    cat_sales = filtered.groupby('Category')['Sales'].sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(cat_sales.values, labels=cat_sales.index, autopct='%1.1f%%', startangle=90)
    st.pyplot(fig2)

st.divider()

# Top 5 Customers 
st.subheader("Top 5 Customers by Sales")
top_customers = (
    filtered.groupby('Customer Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(5)
    .reset_index()
)
top_customers.columns = ['Customer Name', 'Total Sales']
top_customers['Total Sales'] = top_customers['Total Sales'].map("${:,.2f}".format)
st.table(top_customers)

st.divider()

# Profit by Sub-Category 
st.subheader("Profit by Sub-Category")
sub_profit = filtered.groupby('Sub-Category')['Profit'].sum().sort_values()
fig3, ax3 = plt.subplots(figsize=(10, 5))
colors = ['red' if v < 0 else 'green' for v in sub_profit.values]
ax3.barh(sub_profit.index, sub_profit.values, color=colors)
ax3.axvline(0, color='black', linewidth=0.8)
ax3.set_xlabel("Profit ($)")
st.pyplot(fig3)
st.divider()

# Monthly Sales Trend 
st.subheader("Monthly Sales Trend")
monthly = filtered.set_index('Order Date')['Sales'].resample('M').sum()
fig4, ax4 = plt.subplots(figsize=(12, 4))
ax4.plot(monthly.index, monthly.values, color='steelblue', linewidth=2)
ax4.set_ylabel("Sales ($)")
ax4.set_xlabel("Month")
st.pyplot(fig4)