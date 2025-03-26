
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cache the CSV loading function for performance benefits
@st.cache_data
def load_csv_data(file):
    # Load CSV file and return a cleaned DataFrame
    try:
        df = pd.read_csv(file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

    # Clean the column names: strip whitespace and convert to lower case
    df.columns = df.columns.str.strip().str.lower()
    return df

def validate_data(df, required_columns=["name", "aum", "tenure", "gdc"]):
    # Ensure that the DataFrame contains all required columns
    st.write("Loaded CSV Columns:", df.columns)  # Debugging the columns
    df.columns = df.columns.str.strip()  # Strip spaces from column names

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}. Please verify your CSV file.")
        return False
    return True

def calculate_blix_score(df):
    # Calculate the BLIX Score based on provided advisor data
    st.write("Columns in the DataFrame: ", df.columns)  # Debugging the columns
    
    # Check if 'competitor_site_visits' exists in the DataFrame
    if 'competitor_site_visits' not in df.columns:
        st.warning("'competitor_site_visits' column is missing. BLIX score will be calculated without it.")
        df['blix_score'] = (
            (df['tenure'] ** 0.5) * 0.2 + 
            np.log(df['aum'] + 1) * 0.3 + 
            (df['gdc'] / df['gdc'].max()) * 0.3
        )
    else:
        # Calculate BLIX score with competitor site visits
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
    # Calculate the Fit Score based on advisor attributes
    df['fit_score'] = (df['fee_based'] * 0.7) + (df['commission'] * 0.3)
    df['fit_score'] = df['fit_score'].clip(0, 100)  # Ensure the score is between 0 and 100
    return df

def clustering(df):
    # Apply KMeans clustering to group advisors by Fit and BLIX scores
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['blix_score', 'fit_score']])
    return df, kmeans

def display_reports(df):
    # Display processed advisor data and further analyses
    st.write("### Processed Advisor Data")
    st.dataframe(df)
    
    # BLIX vs Fit scatter plot
    st.write("### BLIX Score vs. Fit Score")
    fig, ax = plt.subplots()
    ax.scatter(df['blix_score'], df['fit_score'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('BLIX Score')
    ax.set_ylabel('Fit Score')
    st.pyplot(fig)

def main():
    st.title("Advisor Intelligence App")
    st.markdown("Upload your CSV file containing advisor data to view BLIX and Fit Score reports.")
    
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = load_csv_data(file)
        if df is not None:
            if validate_data(df):
                df = calculate_blix_score(df)
                df = calculate_fit_score(df)
                df, kmeans = clustering(df)
                st.success("Data loaded, processed, and clustered successfully!")
                display_reports(df)
            else:
                st.error("Data validation failed. Please correct the CSV file.")
        else:
            st.error("Failed to load CSV data. Please try again with a valid CSV file.")

if __name__ == '__main__':
    main()
