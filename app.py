# ---------------- train_speech_model.py ----------------
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import os

os.makedirs("models", exist_ok=True)

X = pd.DataFrame({
    'mfcc_mean': np.random.rand(100),
    'zcr': np.random.rand(100),
    'spectral_centroid': np.random.rand(100)
})
y = np.random.randint(0, 2, size=100)

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "models/speech_model.pkl")
print("✅ Speech model trained and saved.")

# ---------------- app.py ----------------
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import librosa
import joblib
from fpdf import FPDF
import os
# ---------------- Page Configuration ----------------
st.set_page_config(page_title="PD Severity Dashboard", layout="wide")
st.title("🧠 Parkinson’s Disease Multi-Omic Dashboard")
st.markdown("Upload and analyse speech, molecular, wearable, and environmental data for PD severity prediction.")

# ---------------- Load model ----------------
def load_model(model_path):
    return joblib.load(model_path)


# ---------------- Generate PDF Report ----------------
def generate_pdf_report(results, filename="pd_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title (Unicode-safe)
    pdf.cell(200, 10, txt="Parkinson's Disease Prediction Report", ln=True, align='C')

    # Add prediction results (cleaned for encoding)
    for key, value in results.items():
        safe_key = str(key).replace("’", "'")
        safe_value = str(value).replace("’", "'")
        pdf.cell(200, 10, txt=f"{safe_key}: {safe_value}", ln=True)

    pdf.output(filename)
    return filename

# ---------------- SHAP Visualisation ----------------
def display_shap(model, X, feature_names):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    st.subheader("Feature Importance (SHAP)")
    shape_info = np.array(shap_values.values).shape
    st.write("SHAP value shape:", shape_info)

    try:
        # If multiple outputs (e.g., shape [n_samples, n_features, n_classes])
        if shap_values.values.ndim == 3:
            st.warning("Multi-output SHAP detected. Visualizing class 1 only.")
            class1_values = shap_values[:, :, 1]  # Take SHAP values for class 1
            shap.plots.waterfall(shap.Explanation(
                values=class1_values[0],
                base_values=shap_values.base_values[0][1],
                data=X.iloc[0],
                feature_names=feature_names
            ))
        else:
            # Single-output SHAP
            shap.plots.waterfall(shap_values[0])
    except Exception as e:
        st.error(f"SHAP plotting failed: {e}")
        st.write("SHAP explanation object:")
        st.write(shap_values)

    return shap_values

# ---------------- Tabs Layout ----------------
tabs = st.tabs(["🎙 Speech", "🧬 Molecular", "💧 Wearable", "🌍 Environmental"])

# ---------------- 1. Speech Tab ----------------
with tabs[0]:
    st.header("🎙 Speech Analysis")
    wav_file = st.file_uploader("Upload a speech .wav file", type=["wav"])
    if wav_file:
        with open("temp_speech.wav", "wb") as f:
            f.write(wav_file.read())
        y, sr = librosa.load("temp_speech.wav", sr=22050)
        features = {
            'mfcc_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr)),
            'zcr': np.mean(librosa.feature.zero_crossing_rate(y)),
            'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        }
        X = pd.DataFrame([features])
        model = load_model("models/speech_model.pkl")
        prediction = model.predict(X)[0]
        st.success(f"Predicted PD Severity (Speech): {prediction}")
        display_shap(model, X, X.columns)
        if st.button("📄 Download PDF Report", key="speech_pdf"):
            pdf_path = generate_pdf_report({"Speech Score": prediction})
            st.download_button("Download", open(pdf_path, "rb"), file_name="pd_speech_report.pdf")

# ---------------- 2. Molecular Tab ----------------
with tabs[1]:
    st.header("🧬 Molecular Biomarkers")
    csv_file = st.file_uploader("Upload molecular .csv file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Preview:", df.head())
        model = load_model("models/molecular_model.pkl")
        prediction = model.predict(df)[0]
        st.success(f"Predicted PD Severity (Molecular): {prediction}")
        display_shap(model, df, df.columns)
        if st.button("📄 Download PDF Report", key="mol_pdf"):
            pdf_path = generate_pdf_report({"Molecular Score": prediction})
            st.download_button("Download", open(pdf_path, "rb"), file_name="pd_molecular_report.pdf")

# ---------------- 3. Wearable Tab ----------------
with tabs[2]:
    st.header("💧 Wearable Biosensor Data")
    csv_file = st.file_uploader("Upload wearable sensor .csv file", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Preview:", df.head())

        model = load_model("models/wearable_model.pkl")

        # ✅ Fix: Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.empty:
            st.error("No numeric features found for prediction.")
        else:
            prediction = model.predict(df_numeric)[0]
            st.success(f"Predicted PD Severity (Wearable): {prediction}")
            display_shap(model, df_numeric, df_numeric.columns)

            if st.button("📄 Download PDF Report", key="wear_pdf"):
                pdf_path = generate_pdf_report({"Wearable Score": prediction})
                st.download_button("Download", open(pdf_path, "rb"), file_name="pd_wearable_report.pdf")


# ---------------- 4. Environmental Tab ----------------
with tabs[3]:
    st.header("🌍 Environmental Exposure Data")
    csv_file = st.file_uploader("Upload environmental .csv file", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write("Preview:", df.head())
        model = load_model("models/environmental_model.pkl")
        prediction = model.predict(df)[0]
        st.success(f"Predicted Environmental Risk Score: {prediction}")
        display_shap(model, df, df.columns)
        if st.button("📄 Download PDF Report", key="env_pdf"):
            pdf_path = generate_pdf_report({"Environmental Score": prediction})
            st.download_button("Download", open(pdf_path, "rb"), file_name="pd_environment_report.pdf")

