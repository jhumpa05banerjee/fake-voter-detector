import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Fake Voter Detector", layout="wide")

st.title("🗳️ Fake Voter Detector — Streamlit App")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your VOTERS.csv file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    if st.button("Run Fake Voter Detection"):
        with st.spinner("Processing..."):
            try:
                subprocess.run(["python", "CLEAN4.py"], cwd=ROOT)
                subprocess.run(["python", "model-training.py"], cwd=ROOT)
                final_path = ROOT / "VOTERS_FINAL_DETECTION.csv"
                if final_path.exists():
                    result = pd.read_csv(final_path)
                    st.success("✅ Detection complete!")
                    st.write("### Results", result.head(30))
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button("Download Results", csv, "VOTERS_FINAL_DETECTION.csv", "text/csv")
                else:
                    st.error("VOTERS_FINAL_DETECTION.csv not found! Make sure scripts generate it.")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("👈 Upload your VOTERS.csv to start detection.")
