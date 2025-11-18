import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend import process_voter_csv

st.title("🗳️ Fake Voter Detection System")

uploaded = st.file_uploader("Upload VOTERS.csv", type=["csv"])

if uploaded:
    st.success("File uploaded successfully!")

    uploaded.seek(0)   # ← VERY IMPORTANT
    df = process_voter_csv(uploaded)


    st.subheader("Processed Voter Data")
    st.dataframe(df)

    st.subheader("Summary")
    st.write(df['Final_Fake'].value_counts())

    st.download_button(
        "Download Final Detection CSV",
        df.to_csv(index=False),
        file_name="FAKE_VOTERS_OUTPUT.csv",
        mime="text/csv"
    )


