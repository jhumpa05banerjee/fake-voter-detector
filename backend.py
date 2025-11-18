import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

def process_voter_csv(filedata):

    df = pd.read_csv(filedata)

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

    df['ID_Type'] = df["ID Card No."].apply(get_id_type)
    df['Cleaned_ID'] = df["ID Card No."].apply(clean_voter_id)

    df['Age'] = df['Age'].fillna(0).astype(int)
    df['Name'] = df['Name'].fillna("UNKNOWN").str.upper()
    df["Guardian's Name"] = df["Guardian's Name"].fillna("UNKNOWN").str.upper()
    df["House Name"] = df["House Name"].fillna("UNKNOWN").str.upper()
    df["Gender"] = df["Gender"].fillna("U")

    house_counts = df['House Name'].value_counts()
    df['Frequent_House'] = df['House Name'].apply(lambda x: house_counts[x] > 25)

    def detect_anomalies(row):
        anomalies = []
        if (row['Age'] < 18) or (row['Age'] > 120):
            anomalies.append('Age_Anomaly')
        if row['Name'] == 'UNKNOWN':
            anomalies.append('Missing_Name')
        if row["Guardian's Name"] == 'UNKNOWN':
            anomalies.append('Missing_Guardian')
        if row["Cleaned_ID"] == "":
            anomalies.append('Invalid_ID')
        if row['Frequent_House']:
            anomalies.append('Crowded_House')
        return anomalies

    df['Anomaly_Types'] = df.apply(detect_anomalies, axis=1)
    df['Anomaly_Count'] = df['Anomaly_Types'].apply(len)
    df['Rule_Based_Fake'] = (df['Anomaly_Count'] > 0).astype(int)

    df['Anomaly_Types'] = df['Anomaly_Types'].apply(lambda x: ', '.join(x) if x else "None")

    features = df[['Age', 'Gender', 'ID_Type', 'Anomaly_Count']].copy()

    le1 = LabelEncoder()
    le2 = LabelEncoder()

    features['Gender'] = le1.fit_transform(features['Gender'])
    features['ID_Type'] = le2.fit_transform(features['ID_Type'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = IsolationForest(contamination=0.02)
    preds = model.fit_predict(X_scaled)

    df['ML_Flag'] = np.where(preds == -1, 1, 0)

    df['Final_Fake'] = np.where(
        (df['Rule_Based_Fake'] == 1) | (df['ML_Flag'] == 1),
        1, 0
    )

    return df
