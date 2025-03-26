
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cache the CSV loading function for performance benefits
@st.cache_data
def load_csv_data(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Clean the column names: strip whitespace and convert to lower case
    df.columns = df.columns.str.strip().str.lower()
    return df

def validate_data(df, required_columns=["name", "aum", "tenure", "gdc", "fee_based", "commission"]):
    # Ensure that the DataFrame contains all required columns
    st.write("Loaded CSV Columns:", df.columns)  # Debugging the columns
    df.columns = df.columns.str.strip()  # Strip spaces from column names

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing required columns: {', '.join(missing_columns)}. Default values will be assigned.")
        for col in missing_columns:
            # Add missing columns with default values (0 for numeric columns)
            if col in ['blix_score', 'fit_score', 'priority_score', 'cluster']:
                df[col] = 0
            else:
                df[col] = 0  # Default value for missing columns (adjust logic as necessary)
    return True

def calculate_blix_score(df):
    st.write("Columns in the DataFrame: ", df.columns)  # Debugging the columns
    
    # Check if 'competitor_site_visits' exists in the DataFrame
    if 'competitor_site_visits' not in df.columns:
        st.warning("'competitor_site_visits' column is missing. BLIX score will be calculated without it.")
        df['competitor_site_visits'] = 0  # Default value for missing column

    if 'event_attendance' not in df.columns:
        st.warning("'event_attendance' column is missing. BLIX score will be calculated with default values.")
        df['event_attendance'] = 0  # Default value for missing column

    df['blix_score'] = (
        (df['tenure'] ** 0.5) * 0.2 + 
        np.log(df['aum'] + 1) * 0.3 + 
        (df['gdc'] / df['gdc'].max()) * 0.3 + 
        (df['competitor_site_visits'] * 0.1) + 
        (df['event_attendance'] * 0.1)
    )
    df['blix_score'] = df['blix_score'].clip(0, 100)  # Ensure the score is between 0 and 100
    return df

def calculate_fit_score(df):
    df['fit_score'] = (df['fee_based'] * 0.7) + (df['commission'] * 0.3)
    df['fit_score'] = df['fit_score'].clip(0, 100)  # Ensure the score is between 0 and 100
    return df

def calculate_priority_score(df, blix_weight=0.4, fit_weight=0.3, gdc_weight=0.2, aum_weight=0.1):
    df['priority_score'] = (
        (df['blix_score'] * blix_weight) + 
        (df['fit_score'] * fit_weight) + 
        (df['gdc'] * gdc_weight) + 
        (df['aum'] * aum_weight)
    )
    return df

def clustering(df):
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['blix_score', 'fit_score']])
    return df, kmeans

def display_reports(df):
    st.write("### Processed Advisor Data")
    st.dataframe(df)

    # Ensure 'name', 'blix_score', 'fit_score', 'priority_score', and 'cluster' exist
    required_columns = ['name', 'blix_score', 'fit_score', 'priority_score', 'cluster']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing columns for display: {', '.join(missing_columns)}")
        return

    # BLIX vs Fit scatter plot
    st.write("### BLIX Score vs. Fit Score")
    fig, ax = plt.subplots()
    ax.scatter(df['blix_score'], df['fit_score'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('BLIX Score')
    ax.set_ylabel('Fit Score')
    st.pyplot(fig)

    st.write("### Advisor Intervention Priority")
    df_sorted = df.sort_values(by='priority_score', ascending=False)
    st.dataframe(df_sorted[['name', 'blix_score', 'fit_score', 'priority_score', 'cluster']])

def main():
    st.title("Advisor Intelligence App")
    st.markdown("Upload your CSV file containing advisor data to view BLIX, Fit Score, and intervention priority reports.")
    
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = load_csv_data(file)
        if df is not None:
            if validate_data(df):
                df = calculate_blix_score(df)
                df = calculate_fit_score(df)
                df = calculate_priority_score(df)
                df, kmeans = clustering(df)
                st.success("Data loaded, processed, and clustered successfully!")
                display_reports(df)
            else:
                st.error("Data validation failed. Please correct the CSV file.")
        else:
            st.error("Failed to load CSV data. Please try again with a valid CSV file.")

if __name__ == '__main__':
    main()
