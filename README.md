# 🧠 Parkinson’s Disease Multi-Omic Dashboard

This project is a **Streamlit dashboard** designed for predicting the severity of Parkinson’s Disease (PD) using multiple data modalities:
- 🎙 Speech data
- 🧬 Molecular biomarkers
- 💧 Wearable sensor data
- 🌍 Environmental exposure

## 🚀 Features

- Upload `.wav` files to analyze speech characteristics
- Upload `.csv` files for molecular, wearable, and environmental datasets
- Visual explanations using **SHAP** (SHapley Additive exPlanations)
- Generate downloadable **PDF reports** for each prediction
- Modular design with support for separate models for each data modality

## 📁 Project Structure
├── app.py # Main Streamlit application
├── train_speech_model.py # Script to train speech-based model
├── models/ # Folder for storing trained models
│ ├── speech_model.pkl
│ ├── molecular_model.pkl
│ ├── wearable_model.pkl
│ └── environmental_model.pkl
├── requirements.txt # Required packages
└── README.md # Project documentation


## 🧪 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/efexu00/parkinsons_dashboard.git
cd parkinsons_dashboard


pip install -r requirements.txt
python train_speech_model.py
streamlit run app.py

 Requirements
Python 3.8+

Streamlit

NumPy, Pandas, Scikit-learn

SHAP, librosa, FPDF, joblib, matplotlib

See requirements.txt for a complete list.

📄 Generate PDF Reports
Each tab provides a download button to generate a PDF summary of the prediction.

📊 Explainability
This app uses SHAP to visualize feature contributions for each prediction, giving clinicians or researchers interpretable insights.

📬 Contributing
PRs and feedback are welcome! Please open an issue if you'd like to suggest improvements or add functionality.



