import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

st.set_page_config(page_title="Fake Voter Detector", layout="wide")

page_bg = """
<style>
    body {
        background: linear-gradient(to bottom right, #dbe9ff, #e8f0ff, #eef5ff);
    }
    .stApp {
        background: linear-gradient(to bottom right, #dcecff, #eaf3ff) !important;
    }
    header, footer { visibility: hidden; }
    .block-container {
        background: rgba(255,255,255,0.12);
        padding: 1.6rem;
        border-radius: 12px;
        backdrop-filter: blur(6px);
    }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.title("Fake Voter Detection")


def normalize_columns(df):
    cols = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9]+', '_', regex=True)
        .str.replace(r'_{2,}', '_', regex=True)
        .str.strip('_')
    )
    df.columns = cols
    return df


def map_and_standardize(df):
    df = df.copy()
    df = normalize_columns(df)

    def find_col(keys):
        for k in keys:
            for c in df.columns:
                if k in c:
                    return c
        return None

    mapping = {
        "age": find_col(["age", "voter_age", "age_years"]),
        "name": find_col(["name", "voter_name", "full_name"]),
        "guardian": find_col(["guardian", "father", "husband", "parent"]),
        "gender": find_col(["gender", "sex"]),
        "id": find_col(["id", "epic", "voter_id", "id_card_no", "epic_no"]),
        "serial_no": find_col(["serial", "serial_no", "s_no", "sno", "sl_no"]),
        "house_name": find_col(["house_name", "building", "home", "house"]),
        "house_no": find_col(["house_no", "address", "ward", "old_ward_name"])
    }

    std = pd.DataFrame()
    std["Age"] = df[mapping["age"]] if mapping["age"] else 0
    std["Name"] = df[mapping["name"]] if mapping["name"] else "UNKNOWN"
    std["Guardian"] = df[mapping["guardian"]] if mapping["guardian"] else "UNKNOWN"
    std["Gender"] = df[mapping["gender"]] if mapping["gender"] else "U"
    std["ID"] = df[mapping["id"]] if mapping["id"] else "UNKNOWN"
    std["Serial_No"] = df[mapping["serial_no"]] if mapping["serial_no"] else 0
    std["House_Name"] = df[mapping["house_name"]] if mapping["house_name"] else "UNKNOWN"
    std["House_No"] = df[mapping["house_no"]] if mapping["house_no"] else "UNKNOWN"

    std["Age"] = pd.to_numeric(std["Age"], errors="coerce").fillna(0).astype(int)
    std["Name"] = std["Name"].astype(str).str.strip().str.upper()
    std["Guardian"] = std["Guardian"].astype(str).str.strip().str.upper()
    std["Gender"] = std["Gender"].astype(str).str.strip().str.upper()
    std["ID"] = std["ID"].astype(str).str.strip().str.upper()
    std["Serial_No"] = pd.to_numeric(std["Serial_No"], errors="coerce").fillna(0).astype(int)
    std["House_Name"] = std["House_Name"].astype(str).str.strip().str.upper()
    std["House_No"] = std["House_No"].astype(str).str.strip().str.upper()

    std["Cleaned_ID"] = std["ID"].str.replace(r'\s+', '', regex=True).str.upper()

    return std


def compute_anomaly_count(df_std):
    df = df_std.copy()

    df["r_age_below_18"] = (df["Age"] < 18).astype(int)

    
    df["r_duplicate_house"] = 0

    
    df["r_duplicate_id"] = df["Cleaned_ID"].duplicated(keep=False).astype(int)

    df["anomaly_count"] = df[["r_age_below_18", "r_duplicate_id"]].sum(axis=1).astype(int)

    df["Anomaly_Count"] = df["anomaly_count"]
    df["Is_Potential_Fake"] = (df["anomaly_count"] >= 1).astype(int)

    df["Risk_Level"] = df["anomaly_count"].apply(
    lambda x: 
        "High Risk" if x >= 2 else
        ("Suspicious" if x == 1 else "Normal")
)


    def triggered_rules(row):
        triggers = []
        if row["r_age_below_18"]:
            triggers.append("R1:Age<18")
        if row["r_duplicate_id"]:
            triggers.append("R3:DuplicateID")
        return ",".join(triggers) if triggers else "None"

    df["Triggered_Rules"] = df.apply(triggered_rules, axis=1)

    return df



uploaded = st.file_uploader("Upload Voter CSV File", type=["csv"])

fallback_local_path = "/mnt/data/final_book1.csv"

if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        source_label = "Uploaded file"
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    try:
        raw = pd.read_csv(fallback_local_path)
        source_label = f"Local sample: {fallback_local_path}"
        st.info(f"No file uploaded — using fallback local CSV: {fallback_local_path}")
    except Exception as e:
        st.error("Please upload a CSV file.")
        st.stop()



df_clean = map_and_standardize(raw)
st.subheader("Cleaned Data")
st.caption(f"Source: {source_label}")
st.dataframe(df_clean.head(50))

df_with_anomalies = compute_anomaly_count(df_clean)


cols_to_show = [
    "Age", "Name", "Guardian", "Gender", "ID", "Serial_No",
    "House_Name", "House_No", "Cleaned_ID", "Anomaly_Count",
    "Risk_Level", "Is_Potential_Fake", "Triggered_Rules"
]

st.subheader("Detection Results (sample)")
st.dataframe(df_with_anomalies[cols_to_show].head(200))


st.header("Anomaly Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Anomaly Count Distribution")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.hist(df_with_anomalies["Anomaly_Count"],
            bins=range(0, df_with_anomalies["Anomaly_Count"].max() + 2),
            edgecolor="black")
    ax.axvline(2, color="red", linestyle="--")
    ax.set_xlabel("Anomaly Count", fontsize=8)
    ax.set_ylabel("Records", fontsize=8)
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution by Risk Level")
    fig2, ax2 = plt.subplots(figsize=(4, 2.5))
    for risk in df_with_anomalies["Risk_Level"].unique():
        subset = df_with_anomalies[df_with_anomalies["Risk_Level"] == risk]
        ax2.hist(subset["Age"], alpha=0.5, bins=10, label=risk)
    ax2.set_xlabel("Age", fontsize=8)
    ax2.set_ylabel("Count", fontsize=8)
    ax2.legend(fontsize=6)
    st.pyplot(fig2)

st.markdown("""
### Meaning of Labels

- **Is_Potential_Fake = 0** → *Not Suspicious*
- **Is_Potential_Fake = 1** →  *Suspicious (Age < 18 OR Duplicate ID)*
- **Anomaly_Count = 2+** → *High Risk / Strong Fake Probability*

- **Risk_Level**  
  - **Normal** → No strong evidence  
  - **Suspicious** → At least 1 anomaly  
  - **High Risk** → Multiple anomalies detected
""")



st.header("Suspicious Voters")

suspicious = df_with_anomalies[df_with_anomalies["Anomaly_Count"] > 0] \
                .sort_values("Anomaly_Count", ascending=False)

st.table(suspicious[cols_to_show].reset_index(drop=True))


csv = df_with_anomalies.to_csv(index=False)
st.download_button("Download Full Results (CSV)", data=csv, file_name="VOTERS_ANOMALY_COUNTS.csv", mime="text/csv")


st.header("Inspect a Record")
row_index = st.number_input("Record index", min_value=1, max_value=len(df_with_anomalies)-1, value=1)
if st.button("Show record details"):
    r = df_with_anomalies.iloc[int(row_index)-1]
    st.write(r[cols_to_show].to_dict())

st.success("Analysis complete.")
