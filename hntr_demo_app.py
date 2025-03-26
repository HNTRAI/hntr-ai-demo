# HNTR AI Demo App - Streamlit Integration for HNTR Fit + BLIX

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="HNTR AI Demo", layout="wide")
st.title("🔍 HNTR AI – Advisor Intelligence Demo")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/512px-React-icon.svg.png", width=100)  # Replace with HNTR AI logo
st.sidebar.title("HNTR AI")
st.sidebar.markdown("This app demonstrates the dual scoring engine behind HNTR AI: \n\n- **BLIX™**: Breakaway Likelihood \n- **Fit Score**: Cultural & performance alignment\n\nUpload advisor data to get real-time insights.")

# Upload data
uploaded_file = st.file_uploader("📂 Upload Advisor Dataset (CSV)", type="csv", help="Must include columns for both BLIX and Fit models.")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🗂 Uploaded Data Preview")
    st.dataframe(df.head())

    st.subheader("🔥 BLIX Score Calculation (Breakaway Risk)")
    current_date = datetime.now()

    def compute_time_decay(last_activity_date, current_date, decay_constant=30):
        days_since = (current_date - pd.to_datetime(last_activity_date)).days
        return np.exp(-days_since / decay_constant)

    def risk_band(score):
        if score >= 70:
            return "🔥 Hot"
        elif score >= 50:
            return "⚠️ Warm"
        else:
            return "❄️ Cold"

    df['CRM_decay'] = df['Last CRM activity date'].apply(lambda x: compute_time_decay(x, current_date))
    linkedin_map = {'low': 0, 'med': 0.5, 'high': 1}
    df['LinkedIn_activity'] = df['Recent LinkedIn activity'].map(linkedin_map)

    numeric_cols_blix = ['CRM_decay', 'Competitor site visits', 'Advisor tenure', 'AUM change %']
    scaler_blix = MinMaxScaler()
    scaled_blix = scaler_blix.fit_transform(df[numeric_cols_blix])
    scaled_df_blix = pd.DataFrame(scaled_blix, columns=[col + '_scaled' for col in numeric_cols_blix])

    features_blix = pd.concat([
        scaled_df_blix,
        df[['LinkedIn_activity', 'Digital engagement score',
            'Event attendance', 'Compensation plan change', 'Legal trigger']]
    ], axis=1)

    model_blix = LogisticRegression()
    dummy_y = np.random.randint(0, 2, features_blix.shape[0])
    model_blix.fit(features_blix, dummy_y)
    blix_scores = model_blix.predict_proba(features_blix)[:, 1] * 100
    df['BLIX Score'] = blix_scores
    df['Risk Band'] = df['BLIX Score'].apply(risk_band)

    st.markdown("#### 📊 Risk Bands")
    st.dataframe(df[['BLIX Score', 'Risk Band']])

    st.markdown("#### 📈 BLIX Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['BLIX Score'], bins=10, edgecolor='black')
    ax.set_title("BLIX Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🧬 HNTR Fit Score Calculation (Advisor Compatibility)")

    numeric_cols_fit = ['GDC', 'AUM', 'tenure', 'fee_based', 'commission']
    df[numeric_cols_fit] = df[numeric_cols_fit].apply(pd.to_numeric, errors='coerce')
    df[numeric_cols_fit] = df[numeric_cols_fit].fillna(df[numeric_cols_fit].median())

    scaler_fit = MinMaxScaler()
    numeric_scaled_fit = pd.DataFrame(scaler_fit.fit_transform(df[numeric_cols_fit]), columns=numeric_cols_fit)

    for col in ['cultural_tags', 'lifestyle_attributes']:
        df[col] = df[col].fillna('').apply(lambda x: [tag.strip() for tag in x.split(',') if tag.strip()])

    mlb_cultural = MultiLabelBinarizer()
    cultural_encoded = pd.DataFrame(mlb_cultural.fit_transform(df['cultural_tags']),
                                    columns=[f"cultural_{tag}" for tag in mlb_cultural.classes_])

    mlb_lifestyle = MultiLabelBinarizer()
    lifestyle_encoded = pd.DataFrame(mlb_lifestyle.fit_transform(df['lifestyle_attributes']),
                                     columns=[f"lifestyle_{tag}" for tag in mlb_lifestyle.classes_])

    features_fit = pd.concat([numeric_scaled_fit, cultural_encoded, lifestyle_encoded], axis=1)

    n_clusters = min(5, len(features_fit))  # Prevent requesting more clusters than rows
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_fit)
    df['Fit Cluster'] = cluster_labels
    centroids = kmeans.cluster_centers_
    similarities = cosine_similarity(features_fit, centroids)
    fit_scores = similarities.max(axis=1) * 100
    df['Fit Score'] = fit_scores

    st.dataframe(df[['Fit Score', 'Fit Cluster']])

    st.markdown("#### 📈 Fit Score Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df['Fit Score'], bins=10, edgecolor='black')
    ax2.set_title("Fit Score Distribution")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Frequency")
    st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🧠 Combined Scoring Overview")
    st.dataframe(df[['BLIX Score', 'Risk Band', 'Fit Score', 'Fit Cluster']])
