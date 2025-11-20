import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fake Voter Detector", layout="wide")

# Title
st.title("🗳️ Fake Voter Detection - Unsupervised ML Model")

# Load model
@st.cache_resource
def load_model():
    try:
        data = joblib.load("anomaly_detector_model.joblib")
        return data
    except:
        st.error("Model file not found! Please run unsupervised.py first.")
        return None

model_data = load_model()

if model_data:

    threshold = model_data["threshold"]
    scaler = model_data["scaler"]
    feature_cols = model_data["feature_columns"]
    iso_forest = model_data["iso_forest"]
    lof = model_data["lof"]

    st.success("Model Loaded Successfully!")

    uploaded = st.file_uploader("Upload voter CSV file", type=["csv"])

    if uploaded:

        df = pd.read_csv(uploaded)
        st.subheader("📌 Uploaded Data")
        st.dataframe(df.head())

        st.info("Cleaning data... Please wait...")

        from unsupervised5 import (
            clean_voter_data, engineer_anomaly_features,
            prepare_features_for_model, create_ensemble_scores,
            generate_predictions
        )

        df_clean = clean_voter_data(uploaded)   # FIXED
        df_feat = engineer_anomaly_features(df_clean)
        X_scaled, feature_cols, scaler = prepare_features_for_model(df_feat)

        # --------- MODEL SCORES ----------
        iso_scores = iso_forest.score_samples(X_scaled)

        lof.fit(X_scaled)               # ✔ LOF must be fitted before predicting
        lof_scores = lof.negative_outlier_factor_

        iso_pred = iso_forest.predict(X_scaled)
        lof_pred = lof.fit_predict(X_scaled)

        # --------- ENSEMBLE -------------
        ensemble_score, both_anomaly = create_ensemble_scores(
            iso_scores, lof_scores, iso_pred, lof_pred
        )

        df_results = generate_predictions(df_clean, df_feat, ensemble_score, both_anomaly)

        st.subheader("🔍 Detection Results")
        st.dataframe(df_results.head(50))

        st.download_button(
            "Download Full Result CSV",
            df_results.to_csv(index=False),
            file_name="VOTERS_ANOMALY_RESULTS.csv",
            mime="text/csv"
        )

        st.success("✓ Analysis Complete!")

else:
    st.warning("Please train the model by running unsupervised.py")
