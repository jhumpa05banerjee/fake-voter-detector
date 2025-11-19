import numpy as np
import pandas as pd
import re

# --- Load VOTERS.csv ---
file = 'C:/Users/Jhumpa/Desktop/minor/VOTERS.csv'
filedata = pd.read_csv(file, sep=',', header=0)

print("\n=== Before Cleaning ===")
print(filedata.info())

# --- Define helper functions ---
def get_id_type(id_str):
    """Classify ID format."""
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
    """Standardize voter ID values."""
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

# --- Apply voter ID cleaning ---
filedata['ID_Type'] = filedata["ID Card No."].apply(get_id_type)
filedata['Cleaned_ID'] = filedata["ID Card No."].apply(clean_voter_id)

# --- Fill missing values ---
#commented out mean for age
'''mean_age = filedata['Age'].mean()
filedata['Missing_Age_Flag'] = filedata['Age'].isna().astype(int)
filedata['Age'] = filedata['Age'].fillna(mean_age).astype(int)'''


filedata['Age'] = filedata['Age'].fillna(0).astype(int)
filedata['Name'] = filedata['Name'].fillna("UNKNOWN").str.strip().str.upper()
filedata["Guardian's Name"] = filedata["Guardian's Name"].fillna("UNKNOWN").str.strip().str.upper()
filedata["ID Card No."] = filedata["ID Card No."].fillna("UNKNOWN")
filedata["OldWard No/ House No."] = filedata["OldWard No/ House No."].fillna("UNKNOWN")
filedata["House Name"] = filedata["House Name"].fillna("UNKNOWN").str.strip().str.upper()
filedata["Gender"] = filedata["Gender"].fillna("U")   # U = UNKNOWN

# --- Convert columns to correct datatypes ---
filedata['Serial No'] = pd.to_numeric(filedata['Serial No'], errors='coerce').fillna(0).astype(int)
filedata['Name'] = filedata['Name'].astype(str).str.strip()
filedata["Guardian's Name"] = filedata["Guardian's Name"].astype(str).str.strip()
filedata['OldWard No/ House No.'] = filedata['OldWard No/ House No.'].astype(str).str.strip()
filedata['House Name'] = filedata['House Name'].astype(str).str.strip()
# --- Clean and standardize Gender values ---
filedata['Gender'] = filedata['Gender'].astype(str).str.strip().str.upper()

filedata['Gender'] = filedata['Gender'].replace({
    'FEMALE': 'F',
    'MALE': 'M',
    'FEM': 'F',
    'MAL': 'M',
    'F': 'F',
    'M': 'M'
})

# Any non-M/F set as Unknown
filedata['Gender'] = filedata['Gender'].apply(lambda x: x if x in ['M', 'F'] else 'U')


filedata['ID Card No.'] = filedata['ID Card No.'].astype(str).str.strip().str.upper()

print("\n Data Cleaning & Type Conversion Done ===")
print(filedata.dtypes)

# --- Save cleaned data ---
output_file_clean = 'C:/Users/Jhumpa/Desktop/minor/VOTERS_CLEANED.csv'
filedata.to_csv(output_file_clean, index=False)
print(f"\n Cleaned dataset saved to: {output_file_clean}")


# --- Pre-calculated how many voters share the same house ---
house_counts = filedata['House Name'].value_counts()
filedata['Frequent_House'] = filedata['House Name'].apply(lambda x: house_counts[x] > 30)  # flag if 10+ voters in one house

# --- Function to detect anomalies in each row ---
def detect_anomalies(row):
    anomalies = []

    # Age anomalies
    if (row['Age'] < 18) or (row['Age'] > 120):
        anomalies.append('Age_Anomaly')

    # Missing or unknown name/guardian/id
    if row['Name'] == 'UNKNOWN':
        anomalies.append('Missing_Name')
    if row["Guardian's Name"] == 'UNKNOWN':
        anomalies.append('Missing_Guardian')

    
    if row["ID Card No."] == 'UNKNOWN' or row["Cleaned_ID"] == '':
        anomalies.append('ID_Anomaly')
    if row['OldWard No/ House No.'] == 'UNKNOWN':
        anomalies.append('Missing_HouseNo')
    
    

    # Invalid gender
    if row['Gender'] not in ['M', 'F']:
        anomalies.append('Missing_Gender')

    # Weird address (too short or placeholder-like)
    if len(str(row['House Name']).strip()) < 3 or row['House Name'].lower() in ['unknown', 'na']:
        anomalies.append('House_Name_Anomaly')

    #Incorrect format for id
    if row['ID_Type'] == 'New_EPIC' and not re.match(r'^[A-Z]{3}\d{7}$', str(row['Cleaned_ID'])):
        anomalies.append('Invalid_ID_Format')
    
    # Crowded house anomaly (too many voters registered at same house)
    if row['Frequent_House']:
        anomalies.append('Crowded_House')

    return anomalies

# --- Apply anomaly detection per row ---
filedata['Anomaly_Types'] = filedata.apply(detect_anomalies, axis=1)

# --- Duplicate-based anomalies ---
# Duplicate IDs
duplicate_id_mask = filedata['Cleaned_ID'].duplicated(keep=False)
filedata.loc[duplicate_id_mask, 'Anomaly_Types'] = filedata.loc[duplicate_id_mask, 'Anomaly_Types'].apply(
    lambda lst: lst + ['Duplicate_ID'] if 'Duplicate_ID' not in lst else lst)

# Duplicate Name + Guardian + House No combo
duplicate_combo_mask = filedata.duplicated(subset=['Name', "Guardian's Name", "OldWard No/ House No."], keep=False)
filedata.loc[duplicate_combo_mask, 'Anomaly_Types'] = filedata.loc[duplicate_combo_mask, 'Anomaly_Types'].apply(
    lambda lst: lst + ['Duplicate_Combo'] if 'Duplicate_Combo' not in lst else lst)

# --- Create only final anomaly summary columns ---
filedata['Anomaly_Count'] = filedata['Anomaly_Types'].apply(len)
filedata['Anomaly_Types'] = filedata['Anomaly_Types'].apply(lambda x: ', '.join(x) if x else 'None')

# --- Flag fake voters based on threshold ---
threshold = 1
filedata['Fake_Voter_Flag'] = (filedata['Anomaly_Count'] >= threshold).astype(int)

# --- Remove any helper columns if present ---
columns_to_drop = ['Frequent_House', 'Crowded_House', 'Duplicate_ID_Flag']
for col in columns_to_drop:
    if col in filedata.columns:
        filedata.drop(columns=[col], inplace=True)

# --- Keep only required final columns ---
columns_to_keep = [
    'Serial No', 'Name', "Guardian's Name", 'OldWard No/ House No.', 'House Name',
    'Gender', 'Age', 'ID Card No.', 'ID_Type', 'Cleaned_ID',
    'Anomaly_Types', 'Anomaly_Count', 'Fake_Voter_Flag'
]
filedata = filedata[[col for col in columns_to_keep if col in filedata.columns]]

# --- Save anomaly summary ---
output_file_anomaly = 'C:/Users/Jhumpa/Desktop/minor/VOTERS_COMPACT_ANOMALIES.csv'
filedata.to_csv(output_file_anomaly, index=False)
print(f"\n Anomaly summary saved to: {output_file_anomaly}")

# --- Display summary ---
print("\n=== Sample Output ===")
print(filedata[['Serial No', 'Name', 'Gender', 'Age', 'Anomaly_Count', 'Anomaly_Types', 'Fake_Voter_Flag']].head(10))

print("\n=== Anomaly Flag Summary ===")
print(filedata['Fake_Voter_Flag'].value_counts())


# === UNSUPERVISED MODEL: ISOLATION FOREST ===
print("\n=== Starting Isolation Forest Anomaly Model ===")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Select relevant features (ignore text columns) ---
features = ['Age', 'Gender', 'ID_Type', 'Anomaly_Count']
data_model = filedata[features].copy()

# --- Encode categorical columns ---
le_gender = LabelEncoder()
le_idtype = LabelEncoder()

data_model['Gender'] = le_gender.fit_transform(data_model['Gender'])
data_model['ID_Type'] = le_idtype.fit_transform(data_model['ID_Type'])

# --- Scale numeric values ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_model)

# --- Train Isolation Forest ---
iso = IsolationForest(
    n_estimators=200,
    contamination=0.02,   # 2% of records assumed as anomalies (tune if needed)
    random_state=42
)
iso.fit(data_scaled)

# --- Predict anomalies ---
preds = iso.predict(data_scaled)
filedata['ML_Anomaly_Flag'] = np.where(preds == -1, 1, 0)

# --- Compare with rule-based fake voter flag ---
comparison = pd.crosstab(filedata['Fake_Voter_Flag'], filedata['ML_Anomaly_Flag'], rownames=['Rule-Based'], colnames=['ML-Model'])
print("\n=== Comparison: Rule-Based vs Isolation Forest ===")
print(comparison)

# --- Add final combined decision ---
filedata['Final_Fake_Voter'] = np.where(
    (filedata['Fake_Voter_Flag'] == 1) | (filedata['ML_Anomaly_Flag'] == 1), 1, 0
)

print("\n=== Final Fake Voter Count (Combined) ===")
print(filedata['Final_Fake_Voter'].value_counts())

# --- Save final output ---
output_file_final = 'C:/Users/Jhumpa/Desktop/minor/VOTERS_FINAL_DETECTION.csv'
filedata.to_csv(output_file_final, index=False)
print(f"\n Final detection file saved to: {output_file_final}")

