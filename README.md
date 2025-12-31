# 🧠 Digital Habits vs Mental Health Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mental-wellbeing-analytics.streamlit.app/)

A machine learning–based analytics project that explores the relationship between digital habits such as screen time, social media usage, and sleep patterns and mental wellbeing indicators including stress levels and mood scores.

**🚀 [View Live Demo](https://mental-wellbeing-analytics.streamlit.app/)**

This project was developed collaboratively, following a team-based approach to machine learning development.

This repository demonstrates an end-to-end ML workflow, including data preprocessing, exploratory data analysis, supervised and unsupervised learning, and deployment through an interactive web application.

## 🎯 Project Scope

*   Analyze digital behavior data using machine learning techniques
*   Apply EDA and feature engineering to structured datasets
*   Build predictive and clustering models
*   Visualize insights using interactive dashboards
*   Practice collaborative, project-based ML development

## ✨ Features

### 🔬 Data Analysis
*   Exploratory Data Analysis (EDA) with statistical summaries and visualizations
*   Data cleaning, preprocessing, and feature engineering

### 🤖 Machine Learning
*   **Random Forest Classifier** – Stress level prediction
*   **XGBoost Classifier** – Mood severity classification
*   **K-Means Clustering** – Lifestyle behavior segmentation
*   **Isolation Forest** – Detection of anomalous digital behavior

### 📊 Visualization
*   Interactive charts using Plotly
*   Correlation heatmaps
*   Feature importance analysis
*   Cluster visualizations

### 🌐 Web Application
*   Streamlit-based interactive dashboard
*   Real-time predictions based on user input
*   Visual exploration of model outputs

## 🛠️ Technology Stack

*   **Language**: Python
*   **Data Processing**: Pandas, NumPy
*   **Machine Learning**: Scikit-learn, XGBoost
*   **Visualization**: Plotly
*   **Web Framework**: Streamlit

## 📁 Project Structure

```
digital-habits-mental-health/
├── digital_habits_vs_mental_health.csv
├── requirements.txt
├── README.md
├── app.py
├── simple_setup.py
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_models.py
│   └── main_analysis.py
├── models/
│   ├── rf_stress.joblib
│   ├── xgb_mood.joblib
│   ├── kmeans.joblib
│   └── isolation_forest.joblib
└── reports/
```

## � Results & Observations

*   Digital habits show notable correlations with mental wellbeing indicators
*   Screen time and sleep duration are influential features in stress prediction
*   Clustering reveals distinct digital lifestyle behavior groups
*   Anomaly detection highlights unusual usage patterns for further analysis

> **Note**: The dataset and results are used for educational and training purposes only and do not represent medical or clinical conclusions.

## � Getting Started

**1️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

**2️⃣ Run model setup**
```bash
python simple_setup.py
```

**3️⃣ Launch the web app**
```bash
streamlit run app.py
```

## � Learning Outcomes

*   Understanding of end-to-end machine learning pipelines
*   Practical experience with supervised and unsupervised models
*   Hands-on feature engineering and evaluation
*   Introductory ML deployment using Streamlit
*   Team-based collaborative development

## 👥 Contributors

This project was developed collaboratively as part of a Placement & Training.

## ⚠️ Disclaimer

This project is intended solely for educational and demonstration purposes. It is not a medical diagnostic or advisory system.

---
**Happy Learning & Building! 🧠📊**
