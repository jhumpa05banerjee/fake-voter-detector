import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fake Voter Detector", layout="wide")


@st.cache_resource
def load_model():
    try:
        data = joblib.load("anomaly_detector_model.joblib")
        return data
    except Exception:
        st.error("❌ Model file not found! Please run unsupervised.py first to generate the model.")
        return None



st.title("🗳️ Fake Voter Detection - Unsupervised ML Model")

model_data = load_model()

if not model_data:
    st.warning("⚠ Model not loaded. Train using unsupervised.py")
    st.stop()

st.success("✓ Model Loaded Successfully!")

threshold = model_data["threshold"]
scaler = model_data["scaler"]
feature_cols = model_data["feature_columns"]
iso_forest = model_data["iso_forest"]
lof = model_data["lof"]

uploaded = st.file_uploader("📤 Upload Voter CSV File", type=["csv"])

if not uploaded:
    st.info("Please upload a CSV file to continue.")
    st.stop()



from unsupervised5 import (
    clean_voter_data,
    engineer_anomaly_features,
    prepare_features_for_model,
    create_ensemble_scores,
    generate_predictions,
)



st.info("⚙ Cleaning data... Please wait...")


df_clean = clean_voter_data(uploaded)

st.subheader("📌 Cleaned Data Sample")
st.dataframe(df_clean.head())


df_feat = engineer_anomaly_features(df_clean)


X_scaled, feature_cols_new, scaler_new = prepare_features_for_model(df_feat)




iso_scores = iso_forest.score_samples(X_scaled)
iso_pred = iso_forest.predict(X_scaled)


lof.fit(X_scaled)  
lof_scores = lof.negative_outlier_factor_
lof_pred = lof.fit_predict(X_scaled)


ensemble_score, both_anomaly = create_ensemble_scores(
    iso_scores, lof_scores, iso_pred, lof_pred
)


df_results = generate_predictions(
    df_clean, df_feat, ensemble_score, both_anomaly, threshold=threshold
)

-

st.subheader("🔍 Detection Results (Top 50 Risky Records)")
st.dataframe(df_results.head(50))


st.download_button(
    "⬇ Download Full Result CSV",
    df_results.to_csv(index=False),
    file_name="VOTERS_ANOMALY_RESULTS.csv",
    mime="text/csv",
)

st.success("🎉 Analysis Complete!")
