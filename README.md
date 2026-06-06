<div align="center">

# ⚡ ChurnIQ — Telecom Customer Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

### 🚀 Predict customer churn in real time with an interactive ML-powered dashboard

[![Live App](https://img.shields.io/badge/🌐_Live_App-Click_Here-00e5ff?style=for-the-badge)](https://churnprediction-zmwbrsnrdfajtydbpk6ril.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/📁_GitHub-Source_Code-181717?style=for-the-badge)](https://github.com/YOUR_USERNAME/ChurnPrediction)

</div>

---

## 📌 Overview

**ChurnIQ** is an end-to-end Machine Learning project that predicts whether a telecom customer will churn (leave the service) or stay — before it happens.

Telecom companies lose **thousands of customers every month**. Identifying at-risk customers early can save **millions in revenue**. This dashboard makes that prediction instant, visual, and actionable.

> Built entirely from scratch — raw CSV → data preprocessing → model training → interactive web dashboard → public deployment.

---

## 🎯 Features

| Feature | Description |
|---|---|
| ⚡ **Real-time Prediction** | Instantly predicts Churn or No Churn from 19 customer inputs |
| 📊 **Risk Gauge Chart** | Visual gauge showing churn probability from 0–100% |
| 🔍 **Key Risk Factors** | Bar chart highlighting which features drive the prediction |
| 📈 **Session Trend Line** | Tracks churn probability across multiple predictions |
| 🍩 **Session Overview** | Donut chart showing churn vs safe ratio for the session |
| 🧾 **Prediction History** | Logs last 10 predictions with timestamp, contract & probability |
| 📋 **Customer Profile** | Live summary of all entered customer attributes |
| 📥 **Metric Cards** | Real-time counters — total analyzed, churn risk, safe, churn rate |

---

## 🖥️ Dashboard Preview

```
┌─────────────────────────────────────────────────────────┐
│  ChurnIQ · Dashboard                      LIVE  19 May  │
├──────────────┬──────────────┬─────────────┬─────────────┤
│ TOTAL        │ CHURN RISK   │ SAFE        │ CHURN RATE  │
│ ANALYZED     │              │ CUSTOMERS   │             │
│     12       │      4       │     8       │   33.3%     │
├──────────────┴──────────────┴─────────────┴─────────────┤
│                                                          │
│  PREDICTION RESULT        │  CUSTOMER PROFILE           │
│  ┌─────────────────────┐  │  Gender        Male         │
│  │  🔴 CHURN RISK      │  │  Tenure        12 months    │
│  │  Confidence: 82.4%  │  │  Contract      Month-month  │
│  │  ████████░░ 82.4%  │  │  Monthly       $65.00       │
│  └─────────────────────┘  │  Internet      Fiber optic  │
│                            │                             │
│  RISK GAUGE               │  SESSION OVERVIEW           │
│      [Plotly Gauge]        │      [Donut Chart]          │
│                            │                             │
│  KEY RISK FACTORS         │  RECENT PREDICTIONS         │
│      [Bar Chart]           │      [History Log]          │
├────────────────────────────────────────────────────────-─┤
│  CHURN PROBABILITY TREND · SESSION                       │
│      [Line Chart with 50% threshold]                     │
└──────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **ML Model** | Scikit-learn — Logistic Regression |
| **Data Processing** | Pandas, NumPy |
| **Dashboard UI** | Streamlit |
| **Charts & Visualization** | Plotly |
| **Model Serialization** | Joblib / Pickle |
| **Version Control** | Git & GitHub |
| **Deployment** | Streamlit Community Cloud |

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Rows:** 7,043 customers
- **Features:** 21 columns (19 used for prediction)
- **Target:** `Churn` — Yes / No

### Key Features Used

| Category | Features |
|---|---|
| **Demographics** | Gender, SeniorCitizen, Partner, Dependents |
| **Account** | Tenure, Contract, PaperlessBilling, PaymentMethod |
| **Services** | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| **Billing** | MonthlyCharges, TotalCharges |

---

## 🔑 Key Insights from the Data

- 📌 **Month-to-month contracts** have the highest churn rate (~43%)
- 📌 **Fiber optic internet** customers churn more than DSL users
- 📌 Customers with **no tech support** are significantly more likely to leave
- 📌 **Short tenure** (< 12 months) is a strong churn signal
- 📌 **High monthly charges** combined with no long-term contract = highest risk

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.10+
- Git

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ChurnPrediction.git
cd ChurnPrediction
```

### Step 2 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the model and save it
Open `ChurnPrediction.ipynb` in Jupyter and run all cells. This will generate `churn_pipeline.pkl`.

```python
import joblib
joblib.dump(lr, "churn_pipeline.pkl")
```

### Step 5 — Run the dashboard
```bash
streamlit run app.py
```

### Step 6 — Open in browser
```
http://localhost:8501
```

Upload `churn_pipeline.pkl` in the sidebar and start predicting!

---

## 📁 Project Structure

```
ChurnPrediction/
│
├── app.py                  # Streamlit dashboard (main app)
├── ChurnPrediction.ipynb   # Jupyter notebook (EDA + model training)
├── churn_pipeline.pkl      # Trained Logistic Regression model
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

---

## 📦 Requirements

```
streamlit>=1.32.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.2.0
joblib>=1.3.0
plotly>=5.20.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🌐 Live Deployment

The app is deployed on **Streamlit Community Cloud** and accessible publicly:

🔗 **[https://churnprediction-zmwbrsnrdfajtydbpk6ril.streamlit.app](https://churnprediction-zmwbrsnrdfajtydbpk6ril.streamlit.app)**

No installation needed — open the link, upload the model, and start predicting!

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Algorithm** | Logistic Regression |
| **Train/Test Split** | 80% / 20% |
| **Features** | 44 (after one-hot encoding) |

> Run the notebook to see full classification report, confusion matrix, and ROC curve.

---

## 🙋‍♂️ Author

**Ankit**
- 🌐 LinkedIn: [www.linkedin.com/in/iankityadav03]
- 💻 GitHub: [https://github.com/Ankit-zone]
- 🚀 Live App: [https://churnprediction-zmwbrsnrdfajtydbpk6ril.streamlit.app](https://churnprediction-zmwbrsnrdfajtydbpk6ril.streamlit.app)

---

## ⭐ Support

If you found this project useful, please consider giving it a **star ⭐** on GitHub — it helps others find it too!

---

<div align="center">

Made with ❤️ by Ankit &nbsp;|&nbsp; Built with Python & Streamlit

</div>
