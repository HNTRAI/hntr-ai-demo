
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
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def validate_data(df, required_columns=['Name', 'BLIX Score', 'Fit Score']):
    # Ensure that the DataFrame contains all required columns.
    st.write("Loaded CSV Columns:", df.columns)  # Debugging the columns
    df.columns = df.columns.str.strip()  # Strip spaces from column names

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}. Please verify your CSV file.")
        return False
    return True

def process_data(df):
    # Process the data: handle missing values and compute any additional scores.
    df.fillna({'BLIX Score': 0, 'Fit Score': 0}, inplace=True)  # Fill missing values with 0
    scaler = MinMaxScaler()
    df[['BLIX Score', 'Fit Score']] = scaler.fit_transform(df[['BLIX Score', 'Fit Score']])
    return df

def clustering(df):
    # Apply KMeans clustering to group advisors by Fit and BLIX scores.
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df[['BLIX Score', 'Fit Score']])
    return df, kmeans

def display_reports(df):
    # Display various reports from the processed DataFrame.
    st.write("### Processed Advisor Data")
    st.dataframe(df)
    
    # BLIX vs Fit scatter plot
    st.write("### BLIX Score vs. Fit Score")
    fig, ax = plt.subplots()
    ax.scatter(df['BLIX Score'], df['Fit Score'], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel('BLIX Score')
    ax.set_ylabel('Fit Score')
    st.pyplot(fig)

def main():
    st.title("Advisor Intelligence App")
    st.markdown("Upload your advisor CSV data to see BLIX and Fit Score reports.")
    
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = load_csv_data(file)
        if df is not None:
            if validate_data(df):
                df = process_data(df)
                df, kmeans = clustering(df)
                st.success("Data loaded, processed, and clustered successfully!")
                display_reports(df)
            else:
                st.error("Data validation failed. Please correct the issues in your CSV file.")
        else:
            st.error("Failed to load data. Please try again with a valid CSV file.")

if __name__ == '__main__':
    main()
