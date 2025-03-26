import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =============================================================================
# 1. Helper Functions
# =============================================================================

def compute_time_decay(last_activity_date, current_date, decay_constant=30):
    """
    Compute an exponential time-decay factor based on days since last CRM activity.
    """
    days_since = (current_date - last_activity_date).days
    decay = np.exp(-days_since / decay_constant)
    return decay

def risk_band(score):
    """
    Assign a risk band based on the score:
      - Hot: score >= 70
      - Warm: 50 <= score < 70
      - Cold: score < 50
    """
    if score >= 70:
        return "Hot"
    elif score >= 50:
        return "Warm"
    else:
        return "Cold"

# =============================================================================
# 2. Data Integrity & Missing Column Handling
# =============================================================================

def ensure_columns(df):
    """
    Ensure required columns are present and add missing optional columns with default values.
    
    Required: 'AUM', 'GDC'
    Optional: 'competitor_site_visits', 'event_attendance'
    """
    required_columns = ['AUM', 'GDC']
    optional_columns = {'competitor_site_visits': 0, 'event_attendance': 0}
    
    # Check required columns
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()  # Stop app execution if a required column is missing.
    
    # Add optional columns if missing
    for col, default in optional_columns.items():
        if col not in df.columns:
            df[col] = default
    return df

# =============================================================================
# 3. Score Calculation Functions
# =============================================================================

def calculate_blix_score(df):
    """
    Calculate BLIX Score using a simple weighted model.
    Normalize each field (GDC, AUM, competitor_site_visits, event_attendance)
    and combine them with sample weights.
    """
    # Ensure missing values are handled
    df['competitor_site_visits'] = df['competitor_site_visits'].fillna(0)
    df['event_attendance'] = df['event_attendance'].fillna(0)
    
    # Normalize values (add a small constant to avoid division by zero)
    max_gdc = df['GDC'].max() or 1
    max_aum = df['AUM'].max() or 1
    max_comp = df['competitor_site_visits'].max() or 1
    
    df['BLIX Score'] = (
        0.3 * (df['GDC'] / max_gdc) +
        0.2 * (df['AUM'] / max_aum) +
        0.25 * (df['competitor_site_visits'] / (max_comp + 1)) +
        0.25 * df['event_attendance']
    ) * 100  # Scale to 0-100
    return df

def calculate_fit_score(df):
    """
    Calculate Fit Score as a function of AUM and GDC.
    For demonstration, this is a simple average of normalized AUM and GDC.
    """
    max_aum = df['AUM'].max() or 1
    max_gdc = df['GDC'].max() or 1
    df['Fit Score'] = ((df['AUM'] / max_aum) + (df['GDC'] / max_gdc)) * 50
    return df

def calculate_priority_score(df):
    """
    Combine BLIX and Fit scores to derive a Priority Score.
    """
    df['Priority Score'] = (df['BLIX Score'] * 0.6 + df['Fit Score'] * 0.4)
    return df

# =============================================================================
# 4. Clustering Function
# =============================================================================

def perform_clustering(df, n_clusters=3):
    """
    Perform KMeans clustering on the BLIX and Fit Scores.
    The cluster label is added as a new column 'Cluster'.
    """
    # Recompute the dynamic scores (if needed)
    df = calculate_blix_score(df)
    df = calculate_fit_score(df)
    
    features = df[['BLIX Score', 'Fit Score']].fillna(0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    return df

# =============================================================================
# 5. Retention Priority Calculation
# =============================================================================

def compute_retention_priority(df):
    """
    Compute a Retention Priority Score for advisors based on:
      - BLIX Score (likelihood to leave, 0-100)
      - Fit Score (alignment with firm, 0-100)
      - GDC and AUM as additional factors

    Steps:
      1. Normalize GDC and AUM to a 0-100 scale.
      2. Compute the Retention Priority Score:
         Retention Priority = 0.4 * BLIX Score 
                            + 0.4 * Fit Score 
                            + 0.1 * GDC_norm 
                            + 0.1 * AUM_norm

    Returns the DataFrame with added columns 'GDC_norm', 'AUM_norm', and 'Retention Priority'.
    """
    df = df.copy()
    max_gdc = df['GDC'].max() or 1
    max_aum = df['AUM'].max() or 1
    df['GDC_norm'] = (df['GDC'] / max_gdc) * 100
    df['AUM_norm'] = (df['AUM'] / max_aum) * 100
    
    df['Retention Priority'] = (
        0.4 * df['BLIX Score'] +
        0.4 * df['Fit Score'] +
        0.1 * df['GDC_norm'] +
        0.1 * df['AUM_norm']
    )
    return df

# =============================================================================
# 6. Data Processing Pipeline (Cached)
# =============================================================================

@st.cache_data
def process_data(file):
    """
    Process the uploaded CSV data:
      - Read CSV
      - Ensure required and optional columns
      - Calculate dynamic score columns (BLIX, Fit, Priority)
      - Apply clustering
      - Compute Retention Priority for retention triage
    """
    df = pd.read_csv(file)
    df = ensure_columns(df)
    st.info("Calculating dynamic columns: BLIX Score, Fit Score, and Priority Score.")
    df = calculate_blix_score(df)
    df = calculate_fit_score(df)
    df = calculate_priority_score(df)
    df = perform_clustering(df)
    df = compute_retention_priority(df)
    return df

# =============================================================================
# 7. Streamlit App Layout
# =============================================================================

st.title("Financial Advisor Scoring App")
st.write("""
This app calculates various scores for financial advisors based on data such as AUM, GDC, competitor site visits, and event attendance.
It generates:
- **BLIX Report:** Sorted by BLIX Score (highest likelihood to leave).
- **Fit Report:** Sorted by Fit Score (lowest fit first).
- **Retention Triage Report:** A combined report using a Retention Priority Score that accounts for high BLIX and high Fit, along with normalized GDC and AUM.
""")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    with st.spinner("Processing data..."):
        df_processed = process_data(uploaded_file)
    st.success("Data processed successfully!")
    
    # --- BLIX Report: Sorted by BLIX Score (descending) ---
    df_blix = df_processed.sort_values(by='BLIX Score', ascending=False)
    st.subheader("BLIX Report")
    st.dataframe(df_blix[['AUM', 'GDC', 'BLIX Score', 'Cluster']])
    
    # --- Fit Report: Sorted by Fit Score (ascending) ---
    df_fit = df_processed.sort_values(by='Fit Score', ascending=True)
    st.subheader("Fit Report")
    st.dataframe(df_fit[['AUM', 'GDC', 'Fit Score', 'Cluster']])
    
    # --- Retention Triage Report ---
    df_retention = df_processed.sort_values(by='Retention Priority', ascending=False)
    st.subheader("Retention Triage Report")
    st.dataframe(df_retention[['AUM', 'GDC', 'BLIX Score', 'Fit Score', 'GDC_norm', 'AUM_norm', 'Retention Priority']])
    
    # --- Additional Plot ---
    if st.checkbox("Show BLIX Score Distribution Plot"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df_processed['BLIX Score'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_title("BLIX Score Distribution")
        ax.set_xlabel("BLIX Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    
    # --- Download Processed Data ---
    csv_data = df_processed.to_csv(index=False).encode('utf-8')
    st.download_button("Download Full Processed CSV", csv_data, "processed_advisors.csv", "text/csv")
else:
    st.info("Please upload a CSV file to begin processing.")
