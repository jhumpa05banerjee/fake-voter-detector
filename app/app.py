import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import re
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Fake Voter Detector", layout="wide")


# ---------------------------------------------------------
# YOUR FUNCTIONS (unchanged)
# ---------------------------------------------------------

def get_id_type(id_str):
    if pd.isna(id_str):
        return "UNKNOWN"
    id_str = str(id_str).strip().upper()
    if re.match(r'^[A-Z]{3}\d{7}$', id_str):
        return "New_EPIC"
    elif re.match(r'^KL/\d+/\d+/\d+$', id_str):
        return "Old_EPIC"
    elif re.match(r'^SECID[A-Z0-9]+$', id_str):
        return "System_ID"
    else:
        return "UNKNOWN"


def clean_voter_id(id_str):
    if pd.isna(id_str):
        return ""
    id_str = str(id_str).strip().upper()
    if re.match(r'^[A-Z]{3}\d{7}$', id_str):
        return id_str
    elif re.match(r'^KL/\d+/\d+/\d+$', id_str):
        return id_str.replace("/", "")
    elif re.match(r'^SECID[A-Z0-9]+$', id_str):
        return "SYS_" + id_str[-6:]
    else:
        return ""


def clean_voter_data(filepath):
    df = pd.read_csv(filepath)

    df['ID_Type'] = df["ID Card No."].apply(get_id_type)
    df['Cleaned_ID'] = df["ID Card No."].apply(clean_voter_id)

    df['Age'] = df['Age'].fillna(0).astype(int)
    df['Name'] = df['Name'].fillna("UNKNOWN").str.upper()
    df["Guardian's Name"] = df["Guardian's Name"].fillna("UNKNOWN").str.upper()
    df["ID Card No."] = df["ID Card No."].fillna("UNKNOWN")
    df["OldWard No/ House No."] = df["OldWard No/ House No."].fillna("UNKNOWN")
    df["House Name"] = df["House Name"].fillna("UNKNOWN").str.upper()
    df["Gender"] = df["Gender"].fillna("U")

    df['Serial No'] = pd.to_numeric(df['Serial No'], errors='coerce').fillna(0).astype(int)

    df['Gender'] = df['Gender'].replace({
        'FEMALE': 'F', 'MALE': 'M', 'FEM': 'F', 'MAL': 'M'
    })
    df['Gender'] = df['Gender'].apply(lambda x: x if x in ['M', 'F'] else 'U')

    return df


def engineer_anomaly_features(df):
    df_features = df.copy()

    df_features['age_below_18'] = (df_features['Age'] < 18).astype(int)
    df_features['age_above_100'] = (df_features['Age'] > 100).astype(int)
    df_features['age_anomaly'] = (
        (df_features['Age'] < 18) | (df_features['Age'] > 120)
    ).astype(int)

    df_features['missing_name'] = (df_features['Name'] == 'UNKNOWN').astype(int)
    df_features['missing_guardian'] = (df_features["Guardian's Name"] == 'UNKNOWN').astype(int)
    df_features['missing_houseno'] = (df_features['OldWard No/ House No.'] == 'UNKNOWN').astype(int)
    df_features['missing_gender'] = (df_features['Gender'] == 'U').astype(int)

    df_features['id_anomaly'] = (
        (df_features['ID Card No.'] == 'UNKNOWN') | (df_features['Cleaned_ID'] == '')
    ).astype(int)

    df_features['invalid_id_format'] = df_features.apply(
        lambda row: 1 if (row['ID_Type'] == 'New_EPIC' and 
                           not re.match(r'^[A-Z]{3}\d{7}$', str(row['Cleaned_ID']))) else 0,
        axis=1
    )

    house_counts = df_features['House Name'].value_counts()
    df_features['crowded_house'] = df_features['House Name'].apply(lambda x: 1 if house_counts.get(x, 0) > 10 else 0)

    df_features['duplicate_id'] = df_features['Cleaned_ID'].duplicated(keep=False).astype(int)
    df_features['duplicate_combo'] = df_features.duplicated(
        subset=['Name', "Guardian's Name", "OldWard No/ House No."], keep=False
    ).astype(int)

    df_features['name_length'] = df_features['Name'].str.len()
    df_features['guardian_length'] = df_features["Guardian's Name"].str.len()
    df_features['house_name_length'] = df_features['House Name'].str.len()
    df_features['id_length'] = df_features['Cleaned_ID'].str.len()

    return df_features




# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.title("🗳️ Fake Voter Detection using ML (IsolationForest + LOF)")
st.write("Upload CSV → Clean → Feature Engineering → Anomaly Detection → Download Results")

uploaded_file = st.file_uploader("📤 Upload VOTERS.csv file", type=["csv"])

if uploaded_file:
    st.success("File uploaded successfully!")

    df = clean_voter_data(uploaded_file)
    st.subheader("📌 Cleaned Data Preview")
    st.dataframe(df.head())

    df_features = engineer_anomaly_features(df)

    feature_columns = [
        'age_below_18', 'age_above_100', 'age_anomaly',
        'missing_name', 'missing_guardian', 'missing_houseno',
        'missing_gender', 'id_anomaly', 'invalid_id_format',
        'house_name_anomaly', 'crowded_house', 'duplicate_id',
        'duplicate_combo', 'name_length', 'guardian_length',
        'house_name_length', 'id_length'
    ]

    X = df_features[feature_columns].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Models
    iso = IsolationForest(contamination=0.02, random_state=42)
    iso_scores = iso.fit_predict(X_scaled)
    iso_dist = iso.score_samples(X_scaled)

    lof = LocalOutlierFactor(contamination=0.02)
    lof_scores = lof.fit_predict(X_scaled)
    lof_dist = lof.negative_outlier_factor_

    # Ensemble
    iso_norm = (iso_dist - iso_dist.min()) / (iso_dist.max() - iso_dist.min())
    lof_norm = (lof_dist - lof_dist.min()) / (lof_dist.max() - lof_dist.min())
    ensemble_score = (iso_norm + lof_norm) / 2

    threshold = st.slider("🔍 Select anomaly threshold", 0.5, 0.95, 0.75)

    df["Anomaly_Score"] = ensemble_score
    df["Is_Fake"] = (ensemble_score >= threshold).astype(int)

    st.subheader("📊 Anomaly Score Distribution")
    fig, ax = plt.subplots()
    ax.hist(ensemble_score, bins=50)
    st.pyplot(fig)

    st.subheader("🚨 Detected Fake Voters")
    st.write(f"Total Fake: **{df['Is_Fake'].sum()}**")
    st.dataframe(df[df["Is_Fake"] == 1].head(50))

    csv_download = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Full Results CSV", csv_download, "Voter_Anomaly_Results.csv")

    st.success("Analysis completed!")


