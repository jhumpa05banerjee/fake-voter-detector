import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Fake Voter Detector", layout="wide")

page_bg = """
<style>
    body {
        background: linear-gradient(to bottom right, #dbe9ff, #e8f0ff, #eef5ff);
    }

    .stApp {
        background: linear-gradient(to bottom right, #dcecff, #eaf3ff) !important;
    }

    .css-18e3th9 {
        background: rgba(255, 255, 255, 0.0) !important;
    }

    header, footer {
        visibility: hidden;
    }

    /* Card-like containers */
    .block-container {
        background: rgba(255, 255, 255, 0.15);
        padding: 2rem;
        border-radius: 15px;
        backdrop-filter: blur(8px);
    }

    /* File uploader box */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.3) !important;
        padding: 15px;
        border-radius: 12px;
    }
</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)




@st.cache_resource
def load_model():
    try:
        return joblib.load("anomaly_detector_model.joblib")
    except:
        st.error("Model file not found! Run unsupervised.py first.")
        return None


st.title("Fake Voter Detection")

model_data = load_model()
if not model_data:
    st.stop()

st.success("✓ Model Loaded Successfully!")

threshold = model_data["threshold"]
scaler = model_data["scaler"]
feature_cols = model_data["feature_columns"]
iso_forest = model_data["iso_forest"]
lof = model_data["lof"]



uploaded = st.file_uploader("Upload Voter CSV File", type=["csv"])
if not uploaded:
    st.info("Upload CSV to continue.")
    st.stop()



from unsupervised5 import (
    clean_voter_data,
    engineer_anomaly_features,
    prepare_features_for_model,
    create_ensemble_scores,
    generate_predictions
)



df_clean = clean_voter_data(uploaded)
st.write("COLUMNS FOUND:", df_clean.columns.tolist())

st.subheader("Cleaned Data Sample")
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
    df_clean, df_feat, ensemble_score, both_anomaly, threshold
)



st.header("Anomaly Detection Visualizations")


center = st.columns([1, 2, 1])


small = (4, 2.5)



with center[1]:
    fig1, ax1 = plt.subplots(figsize=small)
    ax1.hist(df_results["Anomaly_Score"], bins=40, edgecolor="black")
    ax1.set_title("Anomaly Score Distribution")
    st.pyplot(fig1)



with center[1]:
    fig2, ax2 = plt.subplots(figsize=small)
    risk_counts = df_results["Risk_Level"].value_counts()
    ax2.bar(risk_counts.index, risk_counts.values)
    ax2.set_title("Risk Level Count")
    st.pyplot(fig2)



with center[1]:
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



with center[1]:
    top10 = df_results.sort_values("Anomaly_Score", ascending=False).head(10)
    fig4, ax4 = plt.subplots(figsize=small)
    ax4.barh(top10["Serial_No"], top10["Anomaly_Score"])
    ax4.invert_yaxis()
    ax4.set_title("Top 10 Suspicious Voters")
    st.pyplot(fig4)



with center[1]:
    fig5, ax5 = plt.subplots(figsize=small)
    for r in df_results["Risk_Level"].unique():
        ax5.hist(df_results[df_results["Risk_Level"] == r]["Age"], alpha=0.4, label=r)
    ax5.legend(fontsize=6)
    ax5.set_title("Age Distribution by Risk")
    st.pyplot(fig5)



with center[1]:
    fig6, ax6 = plt.subplots(figsize=small)
    sorted_scores = np.sort(df_results["Anomaly_Score"])
    ax6.plot(sorted_scores)
    ax6.set_title("Sorted Anomaly Score Curve")
    st.pyplot(fig6)




st.subheader("Detection Results (Top 50 Risky Records)")
st.dataframe(df_results.head(50))

st.download_button(
    "⬇ Download Full Result CSV",
    df_results.to_csv(index=False),
    file_name="VOTERS_ANOMALY_RESULTS.csv",
    mime="text/csv",
)

st.success("Analysis Complete!")



