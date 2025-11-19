"""
Fake Voter Detection using Unsupervised Anomaly Detection
Models: Isolation Forest + Local Outlier Factor (LOF)
Approach: Detect outliers/anomalies that deviate from normal voter patterns
No labeled data needed - works perfectly for extreme imbalance (1200:2 ratio)

DATA PIPELINE:
1. Load raw VOTERS.csv
2. Clean and standardize data (ID types, gender, names, etc.)
3. Engineer anomaly detection features
4. Train Isolation Forest + LOF models
5. Create ensemble anomaly scores
6. Output predictions and risk levels
"""

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

# ============================================================================
# DATA CLEANING SECTION (from get-id-type.py)
# ============================================================================

def get_id_type(id_str):
    """Classify ID format"""
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
    """Standardize voter ID values"""
    if pd.isna(id_str):
        return ""
    id_str = str(id_str).strip().upper()
    if re.match(r'^[A-Z]{3}\d{7}$', id_str):
        return id_str                     # already clean
    elif re.match(r'^KL/\d+/\d+/\d+$', id_str):
        return id_str.replace("/", "")     # remove slashes for uniformity
    elif re.match(r'^SECID[A-Z0-9]+$', id_str):
        return "SYS_" + id_str[-6:]        # short system ID
    else:
        return ""

def clean_voter_data(filepath):
    """
    Load raw VOTERS.csv and apply all data cleaning transformations
    """
    print("[*] Loading raw voter data...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} voters")
    
    print("[*] Cleaning voter data...")
    
    # -------- ID CARD CLEANING --------
    df['ID_Type'] = df["ID Card No."].apply(get_id_type)
    df['Cleaned_ID'] = df["ID Card No."].apply(clean_voter_id)
    
    # -------- FILL MISSING VALUES --------
    df['Age'] = df['Age'].fillna(0).astype(int)
    df['Name'] = df['Name'].fillna("UNKNOWN").str.strip().str.upper()
    df["Guardian's Name"] = df["Guardian's Name"].fillna("UNKNOWN").str.strip().str.upper()
    df["ID Card No."] = df["ID Card No."].fillna("UNKNOWN")
    df["OldWard No/ House No."] = df["OldWard No/ House No."].fillna("UNKNOWN")
    df["House Name"] = df["House Name"].fillna("UNKNOWN").str.strip().str.upper()
    df["Gender"] = df["Gender"].fillna("U")   # U = UNKNOWN
    
    # -------- CONVERT DATATYPES --------
    df['Serial No'] = pd.to_numeric(df['Serial No'], errors='coerce').fillna(0).astype(int)
    df['Name'] = df['Name'].astype(str).str.strip()
    df["Guardian's Name"] = df["Guardian's Name"].astype(str).str.strip()
    df['OldWard No/ House No.'] = df['OldWard No/ House No.'].astype(str).str.strip()
    df['House Name'] = df['House Name'].astype(str).str.strip()
    
    # -------- CLEAN AND STANDARDIZE GENDER --------
    df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
    df['Gender'] = df['Gender'].replace({
        'FEMALE': 'F',
        'MALE': 'M',
        'FEM': 'F',
        'MAL': 'M',
        'F': 'F',
        'M': 'M'
    })
    # Any non-M/F set as Unknown
    df['Gender'] = df['Gender'].apply(lambda x: x if x in ['M', 'F'] else 'U')
    
    # -------- STANDARDIZE ID CARD --------
    df['ID Card No.'] = df['ID Card No.'].astype(str).str.strip().str.upper()
    
    print("[+] Data cleaning completed")
    print(f"    Columns: {df.columns.tolist()}")
    print(f"    Shape: {df.shape}")
    
    return df

# ============================================================================
# FEATURE ENGINEERING FOR ANOMALY DETECTION
# ============================================================================

def engineer_anomaly_features(df):
    """
    Create anomaly detection features from cleaned data
    """
    print("[*] Engineering features for anomaly detection...")
    
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
        lambda row: 1 if (
            row['ID_Type'] == 'New_EPIC' and 
            not re.match(r'^[A-Z]{3}\d{7}$', str(row['Cleaned_ID']))
        ) else 0,
        axis=1
    )
    
    df_features['house_name_anomaly'] = df_features['House Name'].apply(
        lambda x: 1 if (len(str(x).strip()) < 3 or str(x).lower() in ['unknown', 'na']) else 0
    )
    
    house_counts = df_features['House Name'].value_counts()
    df_features['crowded_house'] = df_features['House Name'].apply(
        lambda x: 1 if house_counts.get(x, 0) > 10 else 0
    )
    
    df_features['duplicate_id'] = (df_features['Cleaned_ID'].duplicated(keep=False) & 
                                    (df_features['Cleaned_ID'] != '')).astype(int)
    
    df_features['duplicate_combo'] = df_features.duplicated(
        subset=['Name', "Guardian's Name", "OldWard No/ House No."], keep=False
    ).astype(int)
    
    df_features['name_length'] = df_features['Name'].astype(str).str.len()
    df_features['guardian_length'] = df_features["Guardian's Name"].astype(str).str.len()
    df_features['house_name_length'] = df_features['House Name'].astype(str).str.len()
    
    df_features['id_length'] = df_features['Cleaned_ID'].astype(str).str.len()
    
    return df_features

# ============================================================================
# 1. DATA LOADING AND FEATURE ENGINEERING
# ============================================================================

def load_and_prepare_data(filepath):
    """Load voter data and prepare features for anomaly detection"""
    print("[*] Loading voter data...")
    df = pd.read_csv(filepath)
    print(f"[+] Loaded {len(df)} voters")
    
    # Clean voter data
    df = clean_voter_data(filepath)
    
    # Engineer anomaly detection features
    df_features = engineer_anomaly_features(df)
    
    return df, df_features

# ============================================================================
# 2. FEATURE SELECTION AND SCALING
# ============================================================================

def prepare_features_for_model(df_features):
    """Select and scale features for anomaly detection"""
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
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"[+] Features prepared: {X_scaled.shape[1]} features for {X_scaled.shape[0]} voters")
    
    return X_scaled, feature_columns, scaler

# ============================================================================
# 3. ANOMALY DETECTION MODELS
# ============================================================================

def train_anomaly_detectors(X_scaled):
    """Train Isolation Forest and LOF models"""
    print("[*] Training anomaly detection models...")
    
    # Isolation Forest - excellent for high-dimensional outliers
    print("  [*] Training Isolation Forest...")
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.02,  # Expect ~2% anomalies
        random_state=42,
        n_jobs=-1
    )
    iso_predictions = iso_forest.fit_predict(X_scaled)
    iso_scores = iso_forest.score_samples(X_scaled)
    
    # Local Outlier Factor - detects local density-based outliers
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

# ============================================================================
# 4. ENSEMBLE SCORING
# ============================================================================

def create_ensemble_scores(iso_scores, lof_scores, iso_predictions, lof_predictions):
    """Create ensemble anomaly score combining both models"""
    print("[*] Creating ensemble anomaly scores...")
    
    # Normalize scores to [0, 1]
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
    lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())
    
    # Ensemble: average of both normalized scores
    ensemble_score = (iso_norm + lof_norm) / 2
    
    # Voting: flag if BOTH models agree on anomaly (-1 = anomaly in sklearn)
    both_anomaly = ((iso_predictions == -1) & (lof_predictions == -1)).astype(int)
    
    return ensemble_score, both_anomaly

# ============================================================================
# 5. THRESHOLD ANALYSIS
# ============================================================================

def analyze_thresholds(ensemble_score):
    """Analyze different anomaly thresholds"""
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

# ============================================================================
# 6. PREDICTIONS AND OUTPUT
# ============================================================================

def generate_predictions(df, df_features, ensemble_score, both_anomaly, threshold=0.75):
    """Generate final predictions and output"""
    print(f"[*] Generating predictions with threshold={threshold}...")
    
    df_results = df.copy()
    df_results['Anomaly_Score'] = ensemble_score
    df_results['Flagged_Both_Models'] = both_anomaly
    df_results['Is_Potential_Fake'] = (ensemble_score >= threshold).astype(int)
    df_results['Risk_Level'] = df_results['Anomaly_Score'].apply(
        lambda x: 'Critical' if x >= 0.85 else 'High' if x >= 0.75 else 'Medium' if x >= 0.60 else 'Low'
    )
    
    # Sort by anomaly score
    df_results = df_results.sort_values('Anomaly_Score', ascending=False)
    
    # Statistics
    n_potential_fakes = df_results['Is_Potential_Fake'].sum()
    n_critical = (df_results['Risk_Level'] == 'Critical').sum()
    n_high = (df_results['Risk_Level'] == 'High').sum()
    
    print(f"\n[+] Prediction Results:")
    print(f"    Total voters: {len(df_results)}")
    print(f"    Potential fakes (threshold={threshold}): {n_potential_fakes} ({n_potential_fakes/len(df_results)*100:.2f}%)")
    print(f"    Critical risk: {n_critical}")
    print(f"    High risk: {n_high}")
    
    return df_results

# ============================================================================
# 7. VISUALIZATION
# ============================================================================

def create_visualizations(df_results, feature_columns, X_scaled):
    """Create comprehensive visualizations"""
    print("[*] Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Anomaly Score Distribution
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(df_results['Anomaly_Score'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(0.75, color='red', linestyle='--', linewidth=2, label='Threshold (0.75)')
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Anomaly Score Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Risk Level Distribution
    ax2 = plt.subplot(2, 3, 2)
    risk_counts = df_results['Risk_Level'].value_counts()
    colors = {'Critical': '#d62728', 'High': '#ff7f0e', 'Medium': '#ffbb78', 'Low': '#2ca02c'}
    risk_colors = [colors.get(level, 'gray') for level in risk_counts.index]
    ax2.barh(risk_counts.index, risk_counts.values, color=risk_colors)
    ax2.set_xlabel('Count')
    ax2.set_title('Voters by Risk Level')
    ax2.grid(alpha=0.3, axis='x')
    
    # 3. Top Anomalous Features (PCA visualization)
    ax3 = plt.subplot(2, 3, 3)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=df_results['Anomaly_Score'], 
                         cmap='RdYlGn_r', alpha=0.6, s=50)
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax3.set_title('Anomalies in 2D Feature Space')
    plt.colorbar(scatter, ax=ax3, label='Anomaly Score')
    
    # 4. Top 10 Anomalous Voters
    ax4 = plt.subplot(2, 3, 4)
    top_10 = df_results.head(10)
    y_pos = np.arange(len(top_10))
    scores = top_10['Anomaly_Score'].values
    ax4.barh(y_pos, scores, color='crimson', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([f"#{i+1}" for i in range(len(top_10))], fontsize=9)
    ax4.set_xlabel('Anomaly Score')
    ax4.set_title('Top 10 Most Anomalous Voters')
    ax4.invert_yaxis()
    ax4.grid(alpha=0.3, axis='x')
    
    # 5. Age Distribution by Risk Level
    ax5 = plt.subplot(2, 3, 5)
    for risk in ['Low', 'Medium', 'High', 'Critical']:
        data = df_results[df_results['Risk_Level'] == risk]['Age']
        ax5.hist(data, bins=20, alpha=0.5, label=risk)
    ax5.set_xlabel('Age')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Age Distribution by Risk Level')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. Cumulative Anomaly Scores
    ax6 = plt.subplot(2, 3, 6)
    sorted_scores = np.sort(df_results['Anomaly_Score'].values)
    ax6.plot(sorted_scores, linewidth=2, color='darkblue')
    ax6.fill_between(range(len(sorted_scores)), sorted_scores, alpha=0.3, color='steelblue')
    ax6.axhline(0.75, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax6.set_xlabel('Voter Index (sorted)')
    ax6.set_ylabel('Anomaly Score')
    ax6.set_title('Cumulative Anomaly Scores')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_analysis.png', dpi=300, bbox_inches='tight')
    print("[+] Visualization saved: anomaly_detection_analysis.png")
    plt.close()

# ============================================================================
# 8. FEATURE IMPORTANCE
# ============================================================================

def create_feature_importance_plot(df_features, feature_columns):
    """Analyze which features are most important for anomalies"""
    print("[*] Analyzing feature importance...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate variance for each feature (proxy for importance in anomaly detection)
    feature_variance = []
    for col in feature_columns:
        var = df_features[col].var()
        feature_variance.append({'Feature': col, 'Variance': var})
    
    df_var = pd.DataFrame(feature_variance).sort_values('Variance', ascending=True).tail(15)
    
    ax.barh(df_var['Feature'], df_var['Variance'], color='teal', alpha=0.7)
    ax.set_xlabel('Feature Variance (Importance Proxy)')
    ax.set_title('Top 15 Most Important Features for Anomaly Detection')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("[+] Feature importance plot saved: feature_importance.png")
    plt.close()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline"""
    print("\n" + "="*70)
    print("FAKE VOTER DETECTION - UNSUPERVISED ANOMALY DETECTION")
    print("="*70 + "\n")
    
    start_time = datetime.now()
    
    # 1. Load and prepare data
    df, df_features = load_and_prepare_data('VOTERS.csv')
    
    # 2. Prepare features
    X_scaled, feature_columns, scaler = prepare_features_for_model(df_features)
    
    # 3. Train models
    iso_forest, lof, iso_scores, lof_scores, iso_pred, lof_pred = train_anomaly_detectors(X_scaled)
    
    # 4. Create ensemble scores
    ensemble_score, both_anomaly = create_ensemble_scores(iso_scores, lof_scores, iso_pred, lof_pred)
    
    # 5. Analyze thresholds
    threshold_analysis = analyze_thresholds(ensemble_score)
    
    # 6. Generate predictions (using 0.75 threshold - adjust as needed)
    df_results = generate_predictions(df, df_features, ensemble_score, both_anomaly, threshold=0.75)
    
    # 7. Create visualizations
    create_visualizations(df_results, feature_columns, X_scaled)
    create_feature_importance_plot(df_features, feature_columns)
    
    # 8. Save results
    print("\n[*] Saving results...")
    df_results.to_csv('VOTERS_ANOMALY_SCORES.csv', index=False)
    print("[+] Predictions saved: VOTERS_ANOMALY_SCORES.csv")
    
    # Save trained models
    joblib.dump({
        'iso_forest': iso_forest,
        'lof': lof,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'threshold': 0.75
    }, 'anomaly_detector_model.joblib')
    print("[+] Model saved: anomaly_detector_model.joblib")
    
    # 9. Summary Report
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
    """
    Extract and explain which features contributed to high anomaly score
    Returns a list of reasons why this voter is flagged
    """
    reasons = []
    
    serial_no = int(row['Serial No'])
    
    # Match by Serial No to find corresponding feature row
    matching_rows = df_features[df_features['Serial No'] == serial_no]
    if len(matching_rows) == 0:
        return ["Unable to find feature data for this voter"]
    
    feature_row = matching_rows.iloc[0]
    
    # Check each anomaly feature and add reason if flagged
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
