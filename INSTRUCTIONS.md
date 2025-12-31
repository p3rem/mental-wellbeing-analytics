# 🚀 Quick Start Instructions

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- The CSV file: `digital_habits_vs_mental_health.csv`

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify setup:**
   ```bash
   python test_setup.py
   ```

## 🎯 Usage

### Step 1: Run the Complete Analysis Pipeline
```bash
python main_analysis.py
```

This will:
- ✅ Load and analyze the dataset (100,000 records)
- ✅ Perform feature engineering
- ✅ Train all machine learning models
- ✅ Generate visualizations and reports
- ✅ Save trained models for the web app

**Expected output:**
- Models saved in `models/` directory
- Visualizations saved in `plots/` directory
- Analysis report saved in `reports/` directory

### Step 2: Start the Interactive Web Application
```bash
streamlit run app.py
```

This will:
- 🌐 Launch the web app in your browser (usually http://localhost:8501)
- 🔮 Provide interactive prediction interface
- 📊 Show data visualizations and analysis
- 💡 Give personalized recommendations

## 📊 What You'll Get

### 🤖 Trained Models
- **Random Forest**: Stress prediction (92.75% accuracy)
- **XGBoost**: Mood severity classification
- **K-Means**: Lifestyle clustering (4 clusters)
- **Isolation Forest**: Anomaly detection

### 📈 Visualizations
- Distribution plots for all features
- Correlation heatmap
- Stress-mood relationship analysis
- Digital habits analysis
- Cluster visualizations
- Feature importance plots

### 🌐 Web App Features
- **Home**: Dataset overview and key metrics
- **Predictions**: Interactive mental health predictions
- **Analysis**: Model performance and feature importance
- **Visualizations**: Interactive data exploration
- **About**: Project documentation

## 🎮 Using the Web App

1. **Navigate to "🔮 Predictions"**
2. **Enter your digital habits:**
   - Screen time (hours/day)
   - Social media platforms used
   - Hours on Reels/short videos
   - Sleep hours
   - Current stress level
   - Current mood score

3. **Get instant results:**
   - Stress level prediction
   - Mood severity classification
   - Lifestyle cluster assignment
   - Anomaly detection
   - Personalized recommendations

## 📁 Project Files

```
MLPROJECTFSP/
├── digital_habits_vs_mental_health.csv    # Your dataset
├── requirements.txt                       # Dependencies
├── README.md                             # Full documentation
├── INSTRUCTIONS.md                       # This file
├── main_analysis.py                      # Main pipeline
├── eda.py                               # Data analysis
├── preprocessing.py                     # Data preprocessing
├── train_models.py                      # Model training
├── app.py                               # Web application
├── test_setup.py                        # Setup verification
├── models/                              # Generated models
├── plots/                               # Generated visualizations
└── reports/                             # Generated reports
```

## 🔍 Troubleshooting

### Common Issues

**❌ "Model files not found"**
```bash
# Solution: Run the analysis pipeline first
python main_analysis.py
```

**❌ "ModuleNotFoundError"**
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**❌ "Data file not found"**
```bash
# Solution: Ensure CSV file is in the current directory
ls digital_habits_vs_mental_health.csv
```

**❌ Streamlit app not loading**
```bash
# Solution: Try different port
streamlit run app.py --server.port 8502
```

### Performance Tips
- Use a machine with at least 8GB RAM
- Close other applications when running analysis
- For faster web app loading, ensure models are pre-trained

## 📈 Expected Results

### Model Performance
- **Random Forest (Stress)**: ~92.75% accuracy
- **XGBoost (Mood)**: ~80-85% accuracy
- **K-Means Clustering**: Good silhouette scores
- **Anomaly Detection**: Identifies ~10% of data as anomalous

### Key Insights
1. Digital habits significantly impact mental health
2. Screen time and sleep patterns predict stress levels
3. Social media usage correlates with mood scores
4. Lifestyle clustering reveals behavioral patterns
5. Anomaly detection identifies unusual digital behavior

## 🎉 Success Indicators

✅ **Setup Complete**: `python test_setup.py` shows all tests passed

✅ **Analysis Complete**: `models/` directory contains trained models

✅ **Web App Running**: Streamlit app opens in browser

✅ **Predictions Working**: Can input data and get predictions

## 📞 Need Help?

1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify the dataset file is present

---

**Happy Analyzing! 🧠📊**

*This project demonstrates the relationship between digital habits and mental health using machine learning techniques.*
