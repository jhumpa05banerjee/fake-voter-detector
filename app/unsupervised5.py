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

# ---------------------------------------
# ADD THIS FUNCTION HERE
def anomaly_detection(df_results):
    st.subheader("📊 Anomaly Detection Graph")

    # Score column (already correct)
    score_col = "Anomaly_Score"

    if score_col not in df_results.columns:
        st.error(f"❌ Score column '{score_col}' not found in df_results")
        return

    try:
        # Smaller graph size
        fig, ax = plt.subplots(figsize=(5, 3))

        # Scatter plot: anomaly score vs fake prediction
        ax.scatter(
            df_results[score_col],
            df_results["Is_Potential_Fake"].astype(int),
            s=12,     # smaller dots
            alpha=0.7
        )

        ax.set_xlabel("Anomaly Score")
        ax.set_ylabel("Fake (1 = Fraud, 0 = Normal)")
        ax.set_title("Anomaly Detection Scatter Plot")

        plt.tight_layout()

        st.pyplot(fig)  # no container_width → stays small

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
    
def map_columns(df):

    # Normalize all columns
    df.columns = df.columns.str.lower().str.strip().str.replace(r'[^a-z0-9 ]', '', regex=True)

    keyword_map = {
        "age": ["age", "voter age", "years"],
        "name": ["name", "voter name", "full name"],
        "guardian": ["guardian", "father", "parent", "guardian name", "father name"],
        "gender": ["gender", "sex"],
        "id": ["id", "epic", "voter id", "id card no", "epic no", "epic number", "id number"],
        "serial_no": ["serial", "serial no", "sno", "slno"],
        "house_name": ["house name", "building", "home"],
        "house_no": ["old ward name", "ward", "house no", "address"],
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


def clean_voter_data(filepath):
    print("[*] Loading raw voter data...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} voters")

    print("[*] Cleaning voter data...")

    mapping = map_columns(df)

    # Create unified standard dataframe
    df_std = pd.DataFrame()

    df_std["Age"] = df[mapping['age']] if mapping['age'] else 0
    df_std["Name"] = df[mapping['name']] if mapping['name'] else "UNKNOWN"
    df_std["Guardian"] = df[mapping['guardian']] if mapping['guardian'] else "UNKNOWN"
    df_std["Gender"] = df[mapping['gender']] if mapping['gender'] else "U"
    df_std["ID"] = df[mapping['id']] if mapping['id'] else "UNKNOWN"
    df_std["Serial_No"] = df[mapping['serial_no']] if mapping['serial_no'] else 0
    df_std["House_Name"] = df[mapping['house_name']] if mapping['house_name'] else "UNKNOWN"
    df_std["House_No"] = df[mapping['house_no']] if mapping['house_no'] else "UNKNOWN"

    # Apply type cleaning
    df_std["Age"] = pd.to_numeric(df_std["Age"], errors="coerce").fillna(0).astype(int)
    df_std["Name"] = df_std["Name"].astype(str).str.upper().str.strip()
    df_std["Guardian"] = df_std["Guardian"].astype(str).str.upper().str.strip()
    df_std["Gender"] = df_std["Gender"].astype(str).str.upper().str.strip()

    # Clean ID
    df_std["Cleaned_ID"] = df_std["ID"].astype(str).str.replace(" ", "").str.upper()

    print("[+] Cleaning completed")
    print("Columns now:", df_std.columns.tolist())
    print("Shape:", df_std.shape)

    return df_std


def engineer_anomaly_features(df):
    print("[*] Engineering features for anomaly detection...")

    df_feat = df.copy()

    # ------------ SAFE COLUMN CHECKS (no KeyError ever) ------------
    safe_cols = {
        "Age": 0,
        "Name": "",
        "Guardian": "",
        "House_No": "",
        "Gender": "",
        "Cleaned_ID": "",
        "ID": "",
        "House_Name": "",
    }

    for col, default in safe_cols.items():
        if col not in df_feat.columns:
            df_feat[col] = default

    # ============= BASIC AGE FEATURES =============
    df_feat["age_below_18"] = (df_feat["Age"] < 18).astype(int)
    df_feat["age_above_100"] = (df_feat["Age"] > 100).astype(int)
    df_feat["age_anomaly"] = ((df_feat["Age"] < 18) | (df_feat["Age"] > 120)).astype(int)

    # ============= MISSING FIELDS =============
    df_feat["missing_name"] = (df_feat["Name"].astype(str).str.strip() == "").astype(int)
    df_feat["missing_guardian"] = (df_feat["Guardian"].astype(str).str.strip() == "").astype(int)
    df_feat["missing_house"] = (df_feat["House_No"].astype(str).str.strip() == "").astype(int)
    df_feat["missing_gender"] = (df_feat["Gender"].astype(str).str.strip() == "").astype(int)

    # ============= ID BASED FEATURES =============
    df_feat["id_anomaly"] = (
        (df_feat["ID"].astype(str).str.strip() == "") |
        (df_feat["Cleaned_ID"].astype(str).str.strip() == "")
    ).astype(int)

    df_feat["id_length"] = df_feat["Cleaned_ID"].astype(str).apply(len)
    df_feat["invalid_id_format"] = df_feat["id_length"].apply(lambda x: 1 if x != 10 else 0)

    # ============= HOUSE NAME ANOMALY =============
    df_feat["house_name_length"] = df_feat["House_Name"].astype(str).apply(len)
    df_feat["house_name_anomaly"] = df_feat["house_name_length"].apply(lambda x: 1 if x < 3 else 0)

    # ============= CROWDED HOUSE =============
    df_feat["crowded_house"] = (
        df_feat.groupby("House_No")["House_No"].transform("count") > 10
    ).astype(int)

    # ============= DUPLICATE BASED =============
    df_feat["duplicate_id"] = df_feat["Cleaned_ID"].duplicated(keep=False).astype(int)

    df_feat["duplicate_combo"] = df_feat.duplicated(
        subset=["Name", "Guardian", "House_No"], keep=False
    ).astype(int)

    # ============= NAME / GUARDIAN LENGTH =============
    df_feat["name_length"] = df_feat["Name"].astype(str).apply(len)
    df_feat["guardian_length"] = df_feat["Guardian"].astype(str).apply(len)
    df_feat["missing_houseno"] = df_feat["missing_house"]


    print("[+] Feature engineering completed! Total features:", len(df_feat.columns))
    return df_feat





def load_and_prepare_data(filepath):
    
    print("[*] Loading voter data...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} voters")
    
    
    df = clean_voter_data(filepath)
    
    
    df_features = engineer_anomaly_features(df)
    
    return df, df_features



def prepare_features_for_model(df_features):
    
    print("[*] Preparing features for modeling...")
    
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
        contamination=0.02,  # Expect ~2% anomalies
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



def generate_predictions(df, df_features, ensemble_score, both_anomaly, threshold=0.75):
    
    print(f"[*] Generating predictions with threshold={threshold}...")
    
    df_results = df.copy()
    df_results['Anomaly_Score'] = ensemble_score
    df_results['Flagged_Both_Models'] = both_anomaly
    df_results['Is_Potential_Fake'] = (ensemble_score >= threshold).astype(int)
    df_results['Risk_Level'] = df_results['Anomaly_Score'].apply(
        lambda x: 'Critical' if x >= 0.85 else 'High' if x >= 0.75 else 'Medium' if x >= 0.60 else 'Low'
    )
    

    df_results = df_results.sort_values('Anomaly_Score', ascending=False)
    
    n_potential_fakes = df_results['Is_Potential_Fake'].sum()
    n_critical = (df_results['Risk_Level'] == 'Critical').sum()
    n_high = (df_results['Risk_Level'] == 'High').sum()
    
    print(f"\n[+] Prediction Results:")
    print(f"    Total voters: {len(df_results)}")
    print(f"    Potential fakes (threshold={threshold}): {n_potential_fakes} ({n_potential_fakes/len(df_results)*100:.2f}%)")
    print(f"    Critical risk: {n_critical}")
    print(f"    High risk: {n_high}")
    
    return df_results



def create_visualizations(df_results, feature_columns, X_scaled):

    st.subheader("📊 Anomaly Detection Visualizations")

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df_results['Anomaly_Score'], bins=50)
    ax1.axvline(0.75, color='red', linestyle='--')
    ax1.set_title("Anomaly Score Distribution")

    ax2 = plt.subplot(2, 3, 2)
    risk_counts = df_results['Risk_Level'].value_counts()
    ax2.barh(risk_counts.index, risk_counts.values)
    ax2.set_title("Voters by Risk Level")

    ax3 = plt.subplot(2, 3, 3)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    s = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=df_results['Anomaly_Score'], cmap="RdYlGn_r")
    plt.colorbar(s, ax=ax3)
    ax3.set_title("PCA View of Anomalies")

    ax4 = plt.subplot(2, 3, 4)
    top10 = df_results.head(10)
    ax4.barh(top10.index, top10["Anomaly_Score"], color="crimson")
    ax4.invert_yaxis()
    ax4.set_title("Top 10 Anomalies")

    ax5 = plt.subplot(2, 3, 5)
    for risk in ['Low', 'Medium', 'High', 'Critical']:
        vals = df_results[df_results["Risk_Level"] == risk]["Age"]
        ax5.hist(vals, bins=15, alpha=0.5, label=risk)
    ax5.legend()
    ax5.set_title("Age Distribution by Risk")

    ax6 = plt.subplot(2, 3, 6)
    sorted_scores = np.sort(df_results['Anomaly_Score'])
    ax6.plot(sorted_scores)
    ax6.axhline(0.75, color='red', linestyle='--')
    ax6.set_title("Cumulative Anomaly Scores")

    plt.tight_layout()

    st.pyplot(fig)

    plt.savefig("anomaly_detection_analysis.png", dpi=300)
    plt.close()

def create_feature_importance_plot(df_features, feature_columns):

    st.subheader("📌 Feature Importance Chart")

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
    
    
    ensemble_score, both_anomaly = create_ensemble_scores(iso_scores, lof_scores, iso_pred, lof_pred)
    
    
    threshold_analysis = analyze_thresholds(ensemble_score)
    
  
    df_results = generate_predictions(df, df_features, ensemble_score, both_anomaly, threshold=0.75)
    
    
    create_visualizations(df_results, feature_columns, X_scaled)
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

