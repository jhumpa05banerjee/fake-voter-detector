import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend import process_voter_csv


# --------------------- CUSTOM DARK THEME CSS ---------------------
page_bg = """
<style>

[data-testid="stAppViewContainer"] {
    background-color: #0d0f15;
    color: #ffffff;
    padding: 20px;
}

[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}

h1, h2, h3, h4 {
    color: #7aa2ff;
}

/* File uploader box */
[data-testid="stFileUploader"] > div:nth-child(1) > div {
    border: 2px dashed #4a4c5a;
    background-color: #1a1c25;
    color: #cfd2dc;
}

/* Browse file button */
.css-1cpxqw2 edgvbvh3 {
    background-color: #2b3a55 !important;
    border-radius: 8px;
    padding: 6px 12px;
    color: white !important;
}

/* Download button */
.stDownloadButton button {
    background-color: #2d5cf6 !important;
    color: white !important;
    border-radius: 10px;
    padding: 10px 18px;
    font-size: 16px;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# --------------------- MAIN APP ---------------------

st.title("🗳️ Fake Voter Detection System")
st.write("Your uploaded CSV will be analyzed for duplicate or fake voter entries.")

uploaded = st.file_uploader("Upload VOTERS.csv", type=["csv"])

if uploaded:
    st.success("File uploaded successfully!")
    df = process_voter_csv(uploaded)

    st.subheader("📊 Processed Voter Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("📌 Summary")
    st.write(df['Final_Fake'].value_counts())

    st.download_button(
        "Download Final Detection CSV",
        df.to_csv(index=False),
        file_name="FAKE_VOTERS_OUTPUT.csv",
        mime="text/csv"
    )
