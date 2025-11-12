import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Fake Voter Detector", layout="wide")

# -------------------------------
# Header
# -------------------------------
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🗳️ Fake Voter Detector</h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload your VOTERS.csv file", type=["csv"])
show_charts = st.sidebar.checkbox("Show Charts", value=True)

# -------------------------------
# Main content
# -------------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Uploaded file: **{uploaded_file.name}** ({df.shape[0]} rows)")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("🚀 Run Fake Voter Detection", use_container_width=True):
        with st.spinner("Running analysis..."):
            subprocess.run(["python", "CLEAN4.py"], cwd=ROOT)
            subprocess.run(["python", "model-training.py"], cwd=ROOT)

            out_path = ROOT / "VOTERS_FINAL_DETECTION.csv"
            if out_path.exists():
                result = pd.read_csv(out_path)
                st.success("✅ Detection complete!")
                st.metric("Total Records", len(result))
                if "Final_Fake_Voter" in result.columns:
                    fakes = int(result["Final_Fake_Voter"].sum())
                    st.metric("Flagged as Fake", fakes)

                tab1, tab2 = st.tabs(["📋 Flagged Voters", "📊 Summary Charts"])
                with tab1:
                    if "Final_Fake_Voter" in result.columns:
                        flagged = result[result["Final_Fake_Voter"] == 1]
                        st.dataframe(flagged, use_container_width=True)
                    else:
                        st.dataframe(result.head(50))

                with tab2:
                    if show_charts and "Final_Fake_Voter" in result.columns:
                        st.bar_chart(result["Final_Fake_Voter"].value_counts())

                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Results", csv, "VOTERS_FINAL_DETECTION.csv", "text/csv")
            else:
                st.error("❌ Output file not found! Check scripts.")
else:
    st.info("👈 Upload a CSV file from the sidebar to start.")

