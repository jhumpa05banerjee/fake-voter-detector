import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
from datetime import datetime
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def normalize_columns(df):
    
    cols = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.replace(r'_{2,}', '_', regex=True)
        .str.strip('_')
    )
    df.columns = cols
    return df

def map_columns(df):
    
    
    df = df.copy()
    df = normalize_columns(df)

    keyword_map = {
        "age": ["age", "voter_age", "age_years"],
        "name": ["name", "voter_name", "full_name"],
        "guardian": ["guardian", "guardian_name", "father_name", "husband_name", "parent_name"],
        "gender": ["gender", "sex"],
        "id": ["id", "epic", "voter_id", "id_card_no", "epic_no", "id_number"],
        "serial_no": ["serial", "serial_no", "s_no", "sno", "sl_no"],
        "house_name": ["house_name", "building", "home", "house"],
        "house_no": ["old_ward_name", "ward", "house_no", "address", "house_no_"]
    }

    mapped = {}
    for std_col, keywords in keyword_map.items():
        found = None
        for col in df.columns:
            for key in keywords:
                if key in col:
                    found = col
                    break
            if found:
                break
        mapped[std_col] = found

    print("[+] Column Mapping:", mapped)
    return mapped

def anomaly_detection(df_results):
    st.subheader("Anomaly Detection Graph")

    
    score_col = "Anomaly_Score"

    if score_col not in df_results.columns:
        st.error(f"Score column '{score_col}' not found in df_results")
        return

    try:
        
        fig, ax = plt.subplots(figsize=(5, 3))

        
        ax.scatter(
            df_results[score_col],
            df_results["Is_Potential_Fake"].astype(int),
            s=12,     
            alpha=0.7
        )

        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Fake (1 = Fraud, 0 = Normal)")
        ax.set_title("Anomaly Detection Scatter Plot")

        plt.tight_layout()

        st.pyplot(fig)  

    except Exception as e:
        st.error(f"Graph Error: {e}")



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
    

def clean_voter_data(filepath):
    
    print("[*] Loading raw voter data...")
    df_raw = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df_raw)} voters")

    
    df_raw = normalize_columns(df_raw)

    
    mapping = map_columns(df_raw)

    
    df_std = pd.DataFrame()
    df_std["Age"] = df_raw[mapping["age"]] if mapping["age"] else 0
    df_std["Name"] = df_raw[mapping["name"]] if mapping["name"] else "UNKNOWN"
    df_std["Guardian"] = df_raw[mapping["guardian"]] if mapping["guardian"] else "UNKNOWN"
    df_std["Gender"] = df_raw[mapping["gender"]] if mapping["gender"] else "U"
    df_std["ID"] = df_raw[mapping["id"]] if mapping["id"] else "UNKNOWN"
    df_std["Serial_No"] = df_raw[mapping["serial_no"]] if mapping["serial_no"] else 0
    df_std["House_Name"] = df_raw[mapping["house_name"]] if mapping["house_name"] else "UNKNOWN"
    df_std["House_No"] = df_raw[mapping["house_no"]] if mapping["house_no"] else "UNKNOWN"

    
    df_std["Age"] = pd.to_numeric(df_std["Age"], errors="coerce").fillna(0).astype(int)
    df_std["Name"] = df_std["Name"].astype(str).fillna("UNKNOWN").str.strip().str.upper()
    df_std["Guardian"] = df_std["Guardian"].astype(str).fillna("UNKNOWN").str.strip().str.upper()
    df_std["Gender"] = df_std["Gender"].astype(str).fillna("U").str.strip().str.upper()
    df_std["ID"] = df_std["ID"].astype(str).fillna("UNKNOWN").str.strip().str.upper()
    df_std["Serial_No"] = pd.to_numeric(df_std["Serial_No"], errors="coerce").fillna(0).astype(int)
    df_std["House_Name"] = df_std["House_Name"].astype(str).fillna("UNKNOWN").str.strip().str.upper()
    df_std["House_No"] = df_std["House_No"].astype(str).fillna("UNKNOWN").str.strip().str.upper()

   
    df_std["Cleaned_ID"] = df_std["ID"].astype(str).str.replace(r'\s+', '', regex=True).str.upper().fillna("")

    print("[+] Cleaning completed")
    print("Columns now:", df_std.columns.tolist())
    print("Shape:", df_std.shape)
    return df_std

def engineer_anomaly_features(df):
    print("[*] Engineering features for anomaly detection...")

    df_feat = df.copy()

    
    for col, default in [("Age",0),("Name",""),("Guardian",""),("House_No",""),("Gender",""),("ID",""),("Cleaned_ID",""),("House_Name","")]:
        if col not in df_feat.columns:
            df_feat[col] = default

    
    df_feat["age_below_18"] = (df_feat["Age"] < 18).astype(int)
    df_feat["age_above_100"] = (df_feat["Age"] > 100).astype(int)
    df_feat["age_anomaly"] = ((df_feat["Age"] < 18) | (df_feat["Age"] > 120)).astype(int)

    
    df_feat["missing_name"] = (df_feat["Name"].astype(str).str.strip().str.upper() == "UNKNOWN").astype(int)
    df_feat["missing_guardian"] = (df_feat["Guardian"].astype(str).str.strip().str.upper() == "UNKNOWN").astype(int)
    df_feat["missing_house"] = (df_feat["House_No"].astype(str).str.strip().str.upper() == "UNKNOWN").astype(int)
    df_feat["missing_gender"] = (df_feat["Gender"].astype(str).str.strip().str.upper() == "U").astype(int)

    
    df_feat["id_anomaly"] = (
    (
        df_feat["ID"].astype(str).str.strip().str.upper() == "UNKNOWN"
    ) | (
        df_feat["Cleaned_ID"].astype(str).str.strip() == ""
    )
).astype(int)

    df_feat["id_length"] = df_feat["Cleaned_ID"].astype(str).apply(len)
    df_feat["invalid_id_format"] = df_feat["id_length"].apply(lambda x: 1 if x != 10 else 0)

    
    df_feat["house_name_length"] = df_feat["House_Name"].astype(str).apply(len)
    df_feat["house_name_anomaly"] = df_feat["house_name_length"].apply(lambda x: 1 if x < 3 else 0)

    
    df_feat["crowded_house"] = df_feat.groupby("House_No")["House_No"].transform("count")
    df_feat["crowded_house"] = (df_feat["crowded_house"] > 10).astype(int)

   
    df_feat["duplicate_id"] = df_feat["Cleaned_ID"].duplicated(keep=False).astype(int)
    df_feat["duplicate_combo"] = df_feat.duplicated(subset=["Name","Guardian","House_No"], keep=False).astype(int)

    
    df_feat["name_length"] = df_feat["Name"].astype(str).apply(len)
    df_feat["guardian_length"] = df_feat["Guardian"].astype(str).apply(len)

    
    df_feat["missing_houseno"] = df_feat["missing_house"]
    df_feat["anomaly_count"] = df_feat[[
    "age_below_18", "age_above_100", "age_anomaly",
    "missing_name", "missing_guardian", "missing_houseno",
    "missing_gender", "id_anomaly", "invalid_id_format",
    "house_name_anomaly", "crowded_house", "duplicate_id",
    "duplicate_combo"
]].sum(axis=1)


    print("[+] Feature engineering completed. Total features:", len(df_feat.columns))
    return df_feat


def plot_anomaly_count_graph(df_results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_results["anomaly_count"], marker="o", markersize=2)
    ax.axhline(2, linestyle="--", color="red")  # threshold = 2

    ax.set_title("Anomaly Count Graph")
    ax.set_xlabel("Record Index")
    ax.set_ylabel("Anomaly Count")

    st.pyplot(fig)




def load_and_prepare_data(filepath):

    print("[*] Loading voter data...")

    
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} voters")

    
    df = clean_voter_data(df)

   
    df = apply_rules(df)

   
    rule_cols = ["R1_Duplicate_ID", "R2_Duplicate_House", "R3_Duplicate_Serial",
                 "R4_Missing_Guardian", "R5_Invalid_Age"]

    df["Anomaly_Count"] = df[rule_cols].sum(axis=1)

    return df



def prepare_features_for_model(df_features):
    
    feature_columns = [
        'age_below_18',
        'age_above_100',
        'age_anomaly',
        'missing_name',
        'missing_guardian',
        'missing_houseno',   
        'missing_gender',
        'id_anomaly',
        'invalid_id_format',
        'house_name_anomaly',
        'crowded_house',
        'duplicate_id',
        'duplicate_combo',
        'name_length',
        'guardian_length',
        'house_name_length',
        'id_length'
    ]

   
    for col in feature_columns:
        if col not in df_features.columns:
            df_features[col] = 0

    X = df_features[feature_columns].copy()
    X = X.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"[+] Features prepared: {X_scaled.shape[1]} features for {X_scaled.shape[0]} voters")
    return X_scaled, feature_columns, scaler



def train_anomaly_detectors(X_scaled):

    print("[*] Training anomaly detection models...")
    
    
    print("  [*] Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.02,  
        random_state=42,
        n_jobs=-1
    )
    iso_predictions = iso_forest.fit_predict(X_scaled)
    iso_scores = iso_forest.score_samples(X_scaled)
    
    
    print("  [*] Training Local Outlier Factor...")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.02,
        novelty=False,
        n_jobs=-1
    )
    lof_predictions = lof.fit_predict(X_scaled)
    lof_scores = lof.negative_outlier_factor_
    
    print("[+] Models trained successfully")
    
    return iso_forest, lof, iso_scores, lof_scores, iso_predictions, lof_predictions


def create_ensemble_scores(iso_scores, lof_scores, iso_predictions, lof_predictions):
    
    print("[*] Creating ensemble anomaly scores...")
    
    
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
    
    
    ensemble_score = (iso_norm + lof_norm) / 2
    
    both_anomaly = ((iso_predictions == -1) & (lof_predictions == -1)).astype(int)
    
    return ensemble_score, both_anomaly



def analyze_thresholds(ensemble_score):
    
    print("[*] Analyzing anomaly thresholds...")
    
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    threshold_analysis = []
    
    for threshold in thresholds:
        n_anomalies = (ensemble_score >= threshold).sum()
        percentage = (n_anomalies / len(ensemble_score)) * 100
        threshold_analysis.append({
            'threshold': threshold,
            'n_anomalies': n_anomalies,
            'percentage': percentage
        })
    
    print("\n[+] Threshold Analysis:")
    print(f"{'Threshold':<12} {'Count':<10} {'Percentage':<12}")
    print("-" * 35)
    for item in threshold_analysis:
        print(f"{item['threshold']:<12.2f} {item['n_anomalies']:<10} {item['percentage']:<12.2f}%")
    
    return threshold_analysis

def apply_rules(df):
    df["R1_Duplicate_ID"] = df.duplicated("ID", keep=False)
    df["R2_Duplicate_House"] = df.duplicated("house name", keep=False)
    df["R3_Duplicate_Serial"] = df.duplicated("Serial_no", keep=False)
    df["R4_Missing_Guardian"] = df["Guardian"].isna() | (df["Guardian"] == "")
    df["R5_Invalid_Age"] = (df["Age"] < 18) | (df["Age"] > 110)
    return df


def get_triggered_rules(row):
    rules = []
    if row["R1_Duplicate_ID"]: rules.append("R1")
    if row["R2_Duplicate_House"]: rules.append("R2")
    if row["R3_Duplicate_Serial"]: rules.append("R3")
    if row["R4_Missing_Guardian"]: rules.append("R4")
    if row["R5_Invalid_Age"]: rules.append("R5")
    return ",".join(rules)


def generate_predictions(df, anomaly_count, threshold=2):

    df_results = df.copy()
    df_results['Anomaly_Count'] = anomaly_count

    df_results['Is_Potential_Fake'] = (anomaly_count >= threshold).astype(int)

    df_results['Risk_Level'] = df_results['Anomaly_Count'].apply(
        lambda x: 'Critical' if x >= 6 else
                  'High' if x >= 4 else
                  'Medium' if x >= 2 else
                  'Low'
    )

    return df_results




def create_visualizations(df_results, feature_columns, X_scaled):

    st.subheader("Anomaly Detection Visualizations")

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df_results['Anomaly_Score'], bins=50)
    ax1.axvline(0.75, color='red', linestyle='--')
    ax1.set_title("Anomaly Score Distribution")


    ax5 = plt.subplot(2, 3, 5)
    for risk in ['Low', 'Medium', 'High', 'Critical']:
        vals = df_results[df_results["Risk_Level"] == risk]["Age"]
        ax5.hist(vals, bins=15, alpha=0.5, label=risk)
    ax5.legend()
    ax5.set_title("Age Distribution by Risk")


    plt.tight_layout()

    st.pyplot(fig)

    plt.savefig("anomaly_detection_analysis.png", dpi=300)
    plt.close()

def create_feature_importance_plot(df_features, feature_columns):

    st.subheader("Feature Importance Chart")

    fig, ax = plt.subplots(figsize=(12, 8))

    variances = []
    for col in feature_columns:
        variances.append({'Feature': col, 'Variance': df_features[col].var()})

    variance_df = pd.DataFrame(variances).sort_values('Variance', ascending=False).head(15)

    ax.barh(variance_df['Feature'], variance_df['Variance'], color='teal')
    ax.set_title("Top 15 Important Features")
    ax.set_xlabel("Variance")

    plt.tight_layout()
    st.pyplot(fig)

    plt.savefig("feature_importance.png", dpi=300)
    plt.close()

def main():
    
    print("\n" + "="*70)
    print("FAKE VOTER DETECTION - UNSUPERVISED ANOMALY DETECTION")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
   
    df, df_features = load_and_prepare_data('VOTERS.csv')
    
   
    X_scaled, feature_columns, scaler = prepare_features_for_model(df_features)
    
    
    iso_forest, lof, iso_scores, lof_scores, iso_pred, lof_pred = train_anomaly_detectors(X_scaled)
    
    
    anomaly_count = detect_anomaly_count(df_features, threshold=2)
    
  
    df_results = generate_predictions(df, anomaly_count, threshold=2)

    
    
    plot_anomaly_count_graph(df_results)

    create_feature_importance_plot(df_features, feature_columns)
    
    
    print("\n[*] Saving results...")
    df_results.to_csv('VOTERS_ANOMALY_SCORES.csv', index=False)
    print("[+] Predictions saved: VOTERS_ANOMALY_SCORES.csv")
    
   
    joblib.dump({
        'iso_forest': iso_forest,
        'lof': lof,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'threshold': 0.75
    }, 'anomaly_detector_model.joblib')
    print("[+] Model saved: anomaly_detector_model.joblib")
    
    
    print("\n" + "="*70)
    print("DETECTION SUMMARY")
    print("="*70)
    print(f"Total voters analyzed: {len(df_results)}")
    print(f"Potential fakes detected: {df_results['Is_Potential_Fake'].sum()}")
    print(f"Critical risk voters: {(df_results['Risk_Level'] == 'Critical').sum()}")
    print(f"High risk voters: {(df_results['Risk_Level'] == 'High').sum()}")
    
    suspicious_voters = df_results[df_results['Anomaly_Score'] > 0.999].sort_values('Anomaly_Score', ascending=False)
    
    print(f"\n{'='*150}")
    print(f"HIGHLY SUSPICIOUS VOTERS (Anomaly Score > 0.999) - TOTAL: {len(suspicious_voters)}")
    print(f"{'='*150}\n")
    
    if len(suspicious_voters) == 0:
        print("[*] No voters found with anomaly score > 0.999")
    else:
        for idx, (_, row) in enumerate(suspicious_voters.iterrows(), 1):
            print(f"[RANK #{idx}] - Anomaly Score: {row['Anomaly_Score']:.6f} | Risk Level: {row['Risk_Level']}")
            print(f"{'-'*150}")
            
            for col in df_results.columns:
                if col not in ['Anomaly_Score', 'Risk_Level', 'Flagged_Both_Models', 'Is_Potential_Fake']:
                    print(f"  {col:<30}: {row[col]}")
            
            reasons = get_anomaly_reasons(row, df_features, feature_columns)
            print(f"\n  ANOMALY REASONS:")
            if reasons:
                for reason in reasons:
                    print(f"    - {reason}")
            print()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    print("="*70 + "\n")
def detect_anomaly_count(df_features, threshold=2):
    
    numeric_cols = ['Age', 'name_length', 'guardian_length', 'house_name_length', 'id_length']

    
    df_numeric = df_features[numeric_cols].copy()

    
    mean_vals = df_numeric.mean()
    std_vals = df_numeric.std().replace(0, 1)

    
    Z = (df_numeric - mean_vals) / std_vals

    
    anomaly_flags = (abs(Z) > threshold).astype(int)

    
    anomaly_count = anomaly_flags.sum(axis=1)

    return anomaly_count


def get_anomaly_reasons(row, df_features, feature_columns):
    
    reasons = []
    
    serial_no = int(row['Serial No'])
    
    
    matching_rows = df_features[df_features['Serial No'] == serial_no]
    if len(matching_rows) == 0:
        return ["Unable to find feature data for this voter"]
    
    feature_row = matching_rows.iloc[0]
    
   
    checks = [
        ('age_below_18', 1, lambda: f"Age below 18 years old (Age: {int(feature_row['Age'])})"),
        ('age_above_100', 1, lambda: f"Age above 100 years old (Age: {int(feature_row['Age'])})"),
        ('age_anomaly', 1, lambda: f"Age outside valid range 18-120 (Age: {int(feature_row['Age'])})"),
        ('missing_name', 1, lambda: "Name field is UNKNOWN"),
        ('missing_guardian', 1, lambda: "Guardian name field is UNKNOWN"),
        ('missing_houseno', 1, lambda: "House number field is UNKNOWN"),
        ('missing_gender', 1, lambda: "Gender field is UNKNOWN"),
        ('id_anomaly', 1, lambda: "ID Card is UNKNOWN or invalid"),
        ('invalid_id_format', 1, lambda: f"Invalid ID format detected"),
        ('house_name_anomaly', 1, lambda: f"House name is too short or generic"),
        ('crowded_house', 1, lambda: "House has unusually high number of voters (>10)"),
        ('duplicate_id', 1, lambda: f"ID is duplicated in dataset: {feature_row['Cleaned_ID']}"),
        ('duplicate_combo', 1, lambda: "Duplicate voter profile - Same Name, Guardian, and House exists"),
    ]
    
    for feature_name, expected_val, reason_func in checks:
        if feature_name in feature_row.index:
            try:
                feature_val = int(feature_row[feature_name])
                if feature_val == expected_val:
                    reasons.append(reason_func())
            except (ValueError, TypeError):
                continue
    
    return reasons if reasons else ["Multiple feature anomalies detected"]

if __name__ == "__main__":
    main()

