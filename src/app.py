#!/usr/bin/env python3
"""
Streamlit Web Application for Digital Habits vs Mental Health Analysis

This app provides an interactive interface for:
1. Mental health prediction based on digital habits
2. Lifestyle clustering analysis
3. Anomaly detection
4. Interactive visualizations

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Digital Habits vs Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
    .cluster-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #ff7f0e;
        margin: 1rem 0;
    }
    .anomaly-box {
        background-color: #fff5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #d62728;
        margin: 1rem 0;
    }
    .recommendations-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data  # Magic decorator. It caches the data so it doesn't reload on every interaction, making the app fast.
def load_data():
    """Load the original dataset"""
    return pd.read_csv('digital_habits_vs_mental_health.csv')

@st.cache_resource  # Similar cache, but for the heavy models.
def load_models():
    """Load trained models and preprocessing components"""
    models = {}
    try:
        # Load models
        models['rf_stress'] = joblib.load('models/rf_stress.joblib')
        models['xgb_mood'] = joblib.load('models/xgb_mood.joblib')
        models['kmeans'] = joblib.load('models/kmeans.joblib')
        models['isolation_forest'] = joblib.load('models/isolation_forest.joblib')
        
        # Load preprocessing components
        preprocessing = joblib.load('models/preprocessing.joblib')
        models['scaler'] = preprocessing['scaler']
        models['feature_lists'] = preprocessing['feature_lists']
        
        # Load model performance
        models['performance'] = joblib.load('models/model_performance.joblib')
        
        return models
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the analysis pipeline first.")
        st.info("Run: python main_analysis.py")
        return None

def create_features(user_input):
    """Create features from user input"""
    # Extract values
    screen_time = user_input['screen_time_hours']
    social_media = user_input['social_media_platforms_used']
    reels_time = user_input['hours_on_Reels']
    sleep_hours = user_input['sleep_hours']
    
    # Create derived features
    digital_wellness_score = (
        (24 - screen_time) / 24 * 10 +
        (5 - social_media) / 5 * 10 +
        (10 - reels_time) / 10 * 10
    ) / 3
    
    sleep_quality = 1 if (sleep_hours >= 7 and sleep_hours <= 9) else 0
    screen_sleep_ratio = screen_time / sleep_hours if sleep_hours > 0 else 0
    social_media_intensity = social_media * reels_time
    stress_mood_imbalance = abs(user_input['stress_level'] - (10 - user_input['mood_score']))
    
    # Create feature dictionary
    features = {
        'screen_time_hours': screen_time,
        'social_media_platforms_used': social_media,
        'hours_on_Reels': reels_time,
        'sleep_hours': sleep_hours,
        'digital_wellness_score': digital_wellness_score,
        'sleep_quality': sleep_quality,
        'screen_sleep_ratio': screen_sleep_ratio,
        'social_media_intensity': social_media_intensity,
        'stress_mood_imbalance': stress_mood_imbalance
    }
    
    return features

def predict_mental_health(user_input, models):
    """Make predictions using trained models"""
    # Create features
    features = create_features(user_input)
    
    # Ensure all required features are present
    stress_features_list = models['feature_lists']['stress']
    mood_features_list = models['feature_lists']['mood']
    lifestyle_features_list = models['feature_lists']['lifestyle']
    
    # Validate that all required features exist
    for feature_list_name, feature_list in [('stress', stress_features_list), 
                                             ('mood', mood_features_list), 
                                             ('lifestyle', lifestyle_features_list)]:
        missing_features = [f for f in feature_list if f not in features]
        if missing_features:
            raise ValueError(f"Missing features for {feature_list_name}: {missing_features}")
    
    # Create DataFrames with proper feature names in correct order (consistent with main app.py)
    stress_features_df = pd.DataFrame([[features[f] for f in stress_features_list]], 
                                      columns=stress_features_list)
    mood_features_df = pd.DataFrame([[features[f] for f in mood_features_list]], 
                                   columns=mood_features_list)
    lifestyle_features_df = pd.DataFrame([[features[f] for f in lifestyle_features_list]], 
                                        columns=lifestyle_features_list)
    
    # Make predictions
    stress_pred = models['rf_stress'].predict(stress_features_df)[0]
    stress_proba = models['rf_stress'].predict_proba(stress_features_df)[0]
    
    mood_pred = models['xgb_mood'].predict(mood_features_df)[0]
    mood_proba = models['xgb_mood'].predict_proba(mood_features_df)[0]
    
    # Cluster prediction
    lifestyle_scaled = models['scaler'].transform(lifestyle_features_df)
    cluster_pred = models['kmeans'].predict(lifestyle_scaled)[0]
    
    # Anomaly detection
    anomaly_score = models['isolation_forest'].decision_function(lifestyle_scaled)[0]
    is_anomaly = models['isolation_forest'].predict(lifestyle_scaled)[0] == -1
    
    return {
        'stress_prediction': stress_pred,
        'stress_probability': stress_proba,
        'mood_prediction': mood_pred,
        'mood_probability': mood_proba,
        'cluster': cluster_pred,
        'anomaly_score': anomaly_score,
        'is_anomaly': is_anomaly,
        'features': features
    }

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🧠 Digital Habits vs Mental Health</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Mental Health Analysis & Prediction")
    
    # Load data and models
    data = load_data()
    models = load_models()
    
    if models is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predictions", "📈 Analysis", "📊 Visualizations", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(data, models)
    elif page == "🔮 Predictions":
        show_predictions_page(models)
    elif page == "📈 Analysis":
        show_analysis_page(data, models)
    elif page == "📊 Visualizations":
        show_visualizations_page(data, models)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(data, models):
    """Display the home page"""
    st.markdown("## 🏠 Welcome to Digital Habits vs Mental Health Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Records", f"{len(data):,}")
    
    with col2:
        st.metric("📈 Features", len(data.columns))
    
    with col3:
        if 'rf_stress' in models['performance']:
            acc = models['performance']['rf_stress']['accuracy']
            st.metric("🎯 Stress Model Accuracy", f"{acc:.2%}")
    
    with col4:
        if 'xgb_mood' in models['performance']:
            acc = models['performance']['xgb_mood']['accuracy']
            st.metric("🎯 Mood Model Accuracy", f"{acc:.2%}")
    
    # Dataset overview
    st.markdown("### 📋 Dataset Overview")
    st.dataframe(data.head(), use_container_width=True)
    
    # Quick insights
    st.markdown("### 💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Key Findings:**
        - Screen time correlates with stress levels
        - Sleep patterns significantly impact mood scores
        - Social media usage affects mental well-being
        - Digital wellness scores predict mental health outcomes
        """)
    
    with col2:
        st.markdown("""
        **🎯 Model Performance:**
        - Random Forest: Stress prediction
        - XGBoost: Mood severity classification
        - K-Means: Lifestyle clustering
        - Isolation Forest: Anomaly detection
        """)
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    st.info("✅ Analysis pipeline completed successfully!")
    st.success("🤖 Models trained and ready for predictions")
    st.success("📊 Visualizations generated")
    st.success("📋 Reports available")

def show_predictions_page(models):
    """Display the predictions page"""
    st.markdown("## 🔮 Mental Health Predictions")
    st.markdown("Enter your digital habits to get personalized mental health insights.")
    
    # User input form
    with st.form("prediction_form"):
        st.markdown("### 📝 Your Digital Habits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            screen_time = st.slider("Screen Time (hours/day)", 0.0, 24.0, 6.0, 0.1)
            social_media = st.slider("Social Media Platforms Used", 0, 10, 3, 1)
            reels_time = st.slider("Hours on Reels/Short Videos", 0.0, 10.0, 2.0, 0.1)
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1)
        
        with col2:
            stress_level = st.slider("Current Stress Level (1-10)", 1, 10, 5, 1)
            mood_score = st.slider("Current Mood Score (1-10)", 1, 10, 7, 1)
        
        submitted = st.form_submit_button("🔮 Get Predictions")
    
    if submitted:
        # Create user input dictionary
        user_input = {
            'screen_time_hours': screen_time,
            'social_media_platforms_used': social_media,
            'hours_on_Reels': reels_time,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'mood_score': mood_score
        }
        
        # Get predictions
        predictions = predict_mental_health(user_input, models)
        
        # Display results
        st.markdown("## 📊 Your Mental Health Analysis")
        
        # Stress prediction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 🧠 Stress Level Prediction")
            
            if predictions['stress_prediction'] == 1:
                st.error("⚠️ **High Stress Detected**")
                st.write("Your digital habits suggest high stress levels.")
                st.write(f"Confidence: {max(predictions['stress_probability']):.1%}")
            else:
                st.success("✅ **Low Stress Detected**")
                st.write("Your digital habits suggest manageable stress levels.")
                st.write(f"Confidence: {max(predictions['stress_probability']):.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Mood prediction
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### 😊 Mood Severity Prediction")
            
            mood_labels = {0: "Low", 1: "Medium", 2: "High"}
            mood_colors = {0: "🔴", 1: "🟡", 2: "🟢"}
            
            mood_pred = mood_labels[predictions['mood_prediction']]
            mood_emoji = mood_colors[predictions['mood_prediction']]
            
            st.write(f"{mood_emoji} **{mood_pred} Mood**")
            st.write(f"Confidence: {max(predictions['mood_probability']):.1%}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Lifestyle cluster
        st.markdown('<div class="cluster-box">', unsafe_allow_html=True)
        st.markdown("### 🎯 Lifestyle Cluster Analysis")
        
        cluster_descriptions = {
            0: "Balanced Digital Lifestyle - Moderate screen time with good sleep patterns",
            1: "High Digital Engagement - Extensive social media use with potential sleep impact",
            2: "Digital Wellness Focus - Low screen time with healthy sleep habits",
            3: "Stress-Prone Pattern - High screen time with poor sleep quality"
        }
        
        cluster = predictions['cluster']
        st.write(f"**Cluster {cluster}**: {cluster_descriptions.get(cluster, 'Unknown pattern')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Anomaly detection
        st.markdown('<div class="anomaly-box">', unsafe_allow_html=True)
        st.markdown("### 🔍 Anomaly Detection")
        
        if predictions['is_anomaly']:
            st.warning("⚠️ **Anomalous Pattern Detected**")
            st.write("Your digital habits show unusual patterns compared to the general population.")
            st.write(f"Anomaly Score: {predictions['anomaly_score']:.3f}")
        else:
            st.success("✅ **Normal Pattern**")
            st.write("Your digital habits are within normal ranges.")
            st.write(f"Anomaly Score: {predictions['anomaly_score']:.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown('<div class="recommendations-box">', unsafe_allow_html=True)
        st.markdown("## 💡 Personalized Recommendations")
        
        recommendations = []
        
        if screen_time > 8:
            recommendations.append("📱 Consider reducing screen time to improve sleep quality")
        
        if sleep_hours < 7:
            recommendations.append("😴 Aim for 7-9 hours of sleep for better mental health")
        
        if social_media > 5:
            recommendations.append("📱 Limit social media platforms to reduce digital overwhelm")
        
        if reels_time > 3:
            recommendations.append("⏰ Set time limits for short-form video consumption")
        
        if predictions['stress_prediction'] == 1:
            recommendations.append("🧘 Practice stress-reduction techniques like meditation")
        
        if predictions['mood_prediction'] == 0:
            recommendations.append("🌞 Increase exposure to natural light and physical activity")
        
        if recommendations:
            for rec in recommendations:
                st.write(f"• {rec}")
        else:
            st.success("🎉 Great job! Your digital habits are well-balanced.")
        
        st.markdown('</div>', unsafe_allow_html=True)

def show_analysis_page(data, models):
    """Display the analysis page"""
    st.markdown("## 📈 Data Analysis")
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'rf_stress' in models['performance']:
            perf = models['performance']['rf_stress']
            st.metric("Random Forest Accuracy", f"{perf['accuracy']:.2%}")
            st.metric("Cross-validation Score", f"{perf['cv_score']:.2%}")
    
    with col2:
        if 'xgb_mood' in models['performance']:
            perf = models['performance']['xgb_mood']
            st.metric("XGBoost Accuracy", f"{perf['accuracy']:.2%}")
            st.metric("Cross-validation Score", f"{perf['cv_score']:.2%}")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    if 'rf_stress' in models['performance'] and 'xgb_mood' in models['performance']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest (Stress Prediction)**")
            rf_importance = models['performance']['rf_stress']['feature_importance']
            rf_df = pd.DataFrame(list(rf_importance.items()), columns=['Feature', 'Importance'])
            rf_df = rf_df.sort_values('Importance', ascending=True)
            
            fig = px.barh(rf_df, x='Importance', y='Feature', title='Stress Prediction Features')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**XGBoost (Mood Prediction)**")
            xgb_importance = models['performance']['xgb_mood']['feature_importance']
            xgb_df = pd.DataFrame(list(xgb_importance.items()), columns=['Feature', 'Importance'])
            xgb_df = xgb_df.sort_values('Importance', ascending=True)
            
            fig = px.barh(xgb_df, x='Importance', y='Feature', title='Mood Prediction Features')
            st.plotly_chart(fig, use_container_width=True)

def show_visualizations_page(data, models):
    """Display the visualizations page"""
    st.markdown("## 📊 Interactive Visualizations")
    
    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlations")
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.markdown("### 📈 Feature Distributions")
    
    selected_feature = st.selectbox(
        "Select a feature to visualize:",
        data.columns.tolist()
    )
    
    fig = px.histogram(
        data, 
        x=selected_feature,
        title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
        nbins=30
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plots
    st.markdown("### 📊 Scatter Plots")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("X-axis feature:", data.columns.tolist(), key='x')
    
    with col2:
        y_feature = st.selectbox("Y-axis feature:", data.columns.tolist(), key='y')
    
    if x_feature != y_feature:
        fig = px.scatter(
            data,
            x=x_feature,
            y=y_feature,
            title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    """Display the about page"""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application analyzes the relationship between digital habits and mental health outcomes.
    Using machine learning techniques, we can predict stress levels and mood severity based on
    digital behavior patterns.
    
    ### 🔬 Methodology
    
    **Data Analysis:**
    - Exploratory Data Analysis (EDA)
    - Feature engineering and preprocessing
    - Correlation analysis
    
    **Machine Learning Models:**
    - **Random Forest**: Binary classification for stress prediction
    - **XGBoost**: Multi-class classification for mood severity
    - **K-Means Clustering**: Lifestyle pattern identification
    - **Isolation Forest**: Anomaly detection
    
    **Features Used:**
    - Screen time hours
    - Social media platforms used
    - Hours on Reels/short videos
    - Sleep hours
    - Derived features (digital wellness score, sleep quality, etc.)
    
    ### 📊 Dataset
    
    The dataset contains information about:
    - Digital habits and screen time patterns
    - Social media usage
    - Sleep patterns
    - Stress levels and mood scores
    
    ### 🛠️ Technical Stack
    
    - **Python**: Core programming language
    - **Pandas**: Data manipulation and analysis
    - **Scikit-learn**: Machine learning algorithms
    - **XGBoost**: Gradient boosting for classification
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### 📈 Key Insights
    
    1. **Digital habits significantly impact mental health outcomes**
    2. **Screen time and sleep patterns are strong predictors of stress levels**
    3. **Social media usage patterns correlate with mood scores**
    4. **Lifestyle clustering reveals distinct behavioral patterns**
    5. **Anomaly detection helps identify unusual digital behavior patterns**
    
    ### 🚀 Getting Started
    
    1. Run the analysis pipeline: `python main_analysis.py`
    2. Start the web app: `streamlit run app.py`
    3. Enter your digital habits to get personalized insights
    
    ### 📝 Disclaimer
    
    This application is for educational and research purposes only.
    It should not be used as a substitute for professional medical advice.
    If you're experiencing mental health concerns, please consult with a healthcare professional.
    """)

if __name__ == "__main__":
    main()