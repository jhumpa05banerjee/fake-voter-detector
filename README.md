

✅ Voter Data Cleaning & Fake Voter Detection

✅ Requirements
Make sure the following Python packages are installed:

```bash
pip install pandas numpy scikit-learn
````

---
📌 Project Overview

This script loads a raw voters dataset, cleans and standardizes important fields, detects anomalies using rule-based logic, and identifies potential **fake voters** using an **Isolation Forest (unsupervised ML model)**.

---
📂 Input

Place your input CSV named:

```
VOTERS.csv
```

Required columns:

```
Serial No, Name, Guardian's Name, ID Card No., OldWard No/ House No., House Name, Gender, Age
```

Update the file path inside the script if needed.
🧹 Data Cleaning Features
✅ Standardizes & fixes missing values

* `Name`, `Guardian's Name`, `House Name` → filled with `"UNKNOWN"`, uppercased
* `Age` → missing replaced with `0`
* `Gender` → standardized to `M`, `F`, or `U`
* `ID Card No.` → cleaned and normalized
✅ Voter ID Classification

| Format Example | Recognized As | Cleaned Output |
| -------------- | ------------- | -------------- |
| `ABC1234567`   | New_EPIC      | As is          |
| `KL/1234/12/1` | Old_EPIC      | `KL1234121`    |
| `SECIDAXYZ123` | System_ID     | `SYS_123XYZ`   |
| Anything else  | UNKNOWN       | `""`           |

---

 🔍 Rule-Based Anomaly Detection

A voter is flagged if any of these issues are detected:

* Age < 18 or > 120
* Missing/unknown name or guardian
* Invalid / unknown voter ID
* Duplicate Voter ID
* Duplicate (Name + Guardian + House No.)
* Too many voters in same house (default: > 30)
* House name too short or placeholder

Creates columns:

```
Anomaly_Types
Anomaly_Count
Fake_Voter_Flag   → 1 if anomaly_count ≥ threshold (default: 1)
```

---

🤖 ML-Based Detection (Isolation Forest)

* Uses features: `Age`, `Gender`, `ID_Type`, `Anomaly_Count`
* Converts categorical → numeric with LabelEncoder
* Scales with StandardScaler
* Model marks unusual patterns as anomalies
* Adds:

```
ML_Anomaly_Flag
Final_Fake_Voter  → 1 if rule-based OR ML says anomaly
```

---

📤 Generated Output Files

| File                           | Description                        |
| ------------------------------ | ---------------------------------- |
| `VOTERS_CLEANED.csv`           | Cleaned dataset                    |
| `VOTERS_COMPACT_ANOMALIES.csv` | Selected columns + anomaly summary |
| `VOTERS_FINAL_DETECTION.csv`   | Final fake voter detection results |



✅ Final Columns in Output

```
Serial No
Name
Guardian's Name
OldWard No/ House No.
House Name
Gender
Age
ID Card No.
ID_Type
Cleaned_ID
Anomaly_Types
Anomaly_Count
Fake_Voter_Flag
ML_Anomaly_Flag
Final_Fake_Voter
```

---

▶️ How to Run

1️⃣ Save script as `voter_cleaning.py`
2️⃣ Make sure CSV path inside script is correct
3️⃣ Run:

```bash
python voter_cleaning.py


