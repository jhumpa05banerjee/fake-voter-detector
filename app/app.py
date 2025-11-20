import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Fake Voter Detector", layout="wide")


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("anomaly_detector_model.joblib")
    except:
        st.error("❌ Model file not found! Run unsupervised.py first.")
        return None


st.title("🗳️ Fake Voter Detection - Unsupervised ML Model")

model_data = load_model()
if not model_data:
    st.stop()

st.success("✓ Model Loaded Successfully!")

threshold = model_data["threshold"]
scaler = model_data["scaler"]
feature_cols = model_data["feature_columns"]
iso_forest = model_data["iso_forest"]
lof = model_data["lof"]


# ---------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------
uploaded = st.file_uploader("📤 Upload Voter CSV File", type=["csv"])
if not uploaded:
    st.info("Upload CSV to continue.")
    st.stop()


# ---------------------------------------------------------
# IMPORT FUNCTIONS FROM unsupervised5
# ---------------------------------------------------------
from unsupervised5 import (
    clean_voter_data,
    engineer_anomaly_features,
    prepare_features_for_model,
    create_ensemble_scores,
    generate_predictions
)


# ---------------------------------------------------------
# CLEANING + FEATURE ENGINEERING
# ---------------------------------------------------------
df_clean = clean_voter_data(uploaded)
st.write("COLUMNS FOUND:", df_clean.columns.tolist())

st.subheader("📌 Cleaned Data Sample")
st.dataframe(df_clean.head())

df_feat = engineer_anomaly_features(df_clean)

X_scaled, feature_cols_new, scaler_new = prepare_features_for_model(df_feat)


# ---------------------------------------------------------
# MODEL SCORES
# ---------------------------------------------------------
iso_scores = iso_forest.score_samples(X_scaled)
iso_pred = iso_forest.predict(X_scaled)

lof.fit(X_scaled)
lof_scores = lof.negative_outlier_factor_
lof_pred = lof.fit_predict(X_scaled)

ensemble_score, both_anomaly = create_ensemble_scores(
    iso_scores, lof_scores, iso_pred, lof_pred
)

df_results = generate_predictions(
    df_clean, df_feat, ensemble_score, both_anomaly, threshold
)


# ---------------------------------------------------------
# ---------------------- GRAPHS SECTION -------------------
# ---------------------------------------------------------
st.header("📊 Anomaly Detection Visualizations")

# Small graph size
small = (5, 3)


# ------------------ 1) ANOMALY SCORE HISTOGRAM ------------------
fig1, ax1 = plt.subplots(figsize=small)
ax1.hist(df_results["Anomaly_Score"], bins=40, edgecolor="black")
ax1.set_title("Anomaly Score Distribution")
st.pyplot(fig1)


# ------------------ 2) RISK LEVEL COUNTS ------------------
fig2, ax2 = plt.subplots(figsize=small)
risk_counts = df_results["Risk_Level"].value_counts()
ax2.bar(risk_counts.index, risk_counts.values)
ax2.set_title("Risk Level Count")
st.pyplot(fig2)


# ------------------ 3) PCA VISUALIZATION ------------------
try:
    X_scaled_plot = scaler.transform(df_feat[feature_cols])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled_plot)

    fig3, ax3 = plt.subplots(figsize=small)
    sc = ax3.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=df_results["Anomaly_Score"],
        cmap="viridis",
        s=20
    )
    plt.colorbar(sc, ax=ax3)
    ax3.set_title("PCA - Anomaly Distribution")
    st.pyplot(fig3)

except Exception as e:
    st.error(f"PCA plotting error: {e}")


# ------------------ 4) TOP 10 SUSPICIOUS ------------------
top10 = df_results.sort_values("Anomaly_Score", ascending=False).head(10)
fig4, ax4 = plt.subplots(figsize=small)
ax4.barh(top10["Serial_No"], top10["Anomaly_Score"])

ax4.invert_yaxis()
ax4.set_title("Top 10 Suspicious Voters")
st.pyplot(fig4)


# ------------------ 5) AGE VS RISK LEVEL ------------------
fig5, ax5 = plt.subplots(figsize=small)
for r in df_results["Risk_Level"].unique():
    ax5.hist(df_results[df_results["Risk_Level"] == r]["Age"], alpha=0.4, label=r)
ax5.legend(fontsize=7)
ax5.set_title("Age Distribution by Risk")
st.pyplot(fig5)


# ------------------ 6) SCORE CURVE ------------------
fig6, ax6 = plt.subplots(figsize=small)
sorted_scores = np.sort(df_results["Anomaly_Score"])
ax6.plot(sorted_scores)
ax6.set_title("Sorted Anomaly Score Curve")
st.pyplot(fig6)


# ---------------------------------------------------------
# RESULT TABLE
# ---------------------------------------------------------
st.subheader("🔍 Detection Results (Top 50 Risky Records)")
st.dataframe(df_results.head(50))

st.download_button(
    "⬇ Download Full Result CSV",
    df_results.to_csv(index=False),
    file_name="VOTERS_ANOMALY_RESULTS.csv",
    mime="text/csv",
)

st.success("🎉 Analysis Complete!")


