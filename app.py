#!/usr/bin/env python3
"""
Digital Habits vs Mental Health - Enhanced Streamlit App
Main application file with improved UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# Get base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'digital_habits_vs_mental_health.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Page configuration
st.set_page_config(
    page_title="Digital Habits vs Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #eff6ff; /* Soft Blue 50 - Calming & Fresh */
        color: #1f2937;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Content Text - Dark */
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4, 
    .main .block-container h5, 
    .main .block-container h6, 
    .main .block-container p, 
    .main .block-container label, 
    .main .block-container .stMarkdown, 
    .main .block-container .stText, 
    .main .block-container .stSelectbox, 
    .main .block-container .stNumberInput {
        color: #1f2937 !important;
        font-family: 'Outfit', sans-serif;
    }

    /* Metrics and Tabs */
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: #1f2937 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #4b5563 !important; /* Gray-600 */
    }
    
    .stTabs [aria-selected="true"] {
        color: #4f46e5 !important; /* Primary Brand Color */
    }

    /* Widget Labels */
    .stSlider label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stMarkdown label, 
    div[data-testid="stMarkdownContainer"] p {
        color: #1f2937 !important;
    }
    
    /* Slider Tick Labels */
    [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
        color: #4b5563 !important;
    }

    /* Sleek Sidebar - Dark Premium Theme */
    section[data-testid="stSidebar"] {
        background-color: #1e1b4b; /* Deep Indigo - Ultra Premium */
        border-right: 1px solid #312e81;
    }
    
    /* Sidebar Text - Light */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4, 
    section[data-testid="stSidebar"] h5, 
    section[data-testid="stSidebar"] h6, 
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] .stMarkdown {
        color: #f8fafc !important;
        font-family: 'Outfit', sans-serif;
    }

    
    /* Header - Sleek & Modern */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 3rem 2rem;
        border-radius: 24px;
        color: white !important;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 25px -5px rgba(79, 70, 229, 0.4), 0 8px 10px -6px rgba(79, 70, 229, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    /* Header - Text Styles (High Specificity) */
    
    .stApp .main .block-container .main-header h1 {
        font-size: 3.5rem !important;
        font-weight: 800 !important;
        color: white !important;
        margin-bottom: 1rem !important;
    }

    .stApp .main .block-container .main-header h3 {
        font-size: 2rem !important; /* Much bigger as requested */
        font-weight: 700 !important;
        opacity: 1;
        margin-bottom: 0.75rem;
        color: #e0e7ff !important;
    }
    
    .stApp .main .block-container .main-header p,
    .header-description {
        font-size: 1.25rem !important;
        opacity: 1;
        color: #ffffff !important; /* Forced White with High Specificity */
        font-weight: 500 !important;
    }

    /* Glass Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 16px;
        color: #1f2937;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-card h3 {
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    .metric-card h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin: 0;
    }
    
    /* Result Boxes */
    .prediction-box, .cluster-box, .anomaly-box {
        padding: 2rem;
        border-radius: 20px;
        color: white;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    /* Stress Prediction - Soft Violet/Indigo */
    .prediction-box {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.4);
    }
    
    /* Cluster - Ocean Blue */
    .cluster-box {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        box-shadow: 0 10px 20px -5px rgba(14, 165, 233, 0.4);
    }
    
    /* Anomaly - Rose/Orange */
    .anomaly-box {
        background: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
        box-shadow: 0 10px 20px -5px rgba(244, 63, 94, 0.4);
    }
    
    /* Recommendations */
    .recommendation-card {
        background: white;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 2px 4px 0 rgba(0,0,0,0.05);
        color: #374151;
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f3f4f6;
    }
    
    /* Buttons */
    div[data-testid="stButton"] > button, 
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #4f46e5 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3) !important;
    }

    div[data-testid="stButton"] > button p, 
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stButton"] > button *, 
    div[data-testid="stFormSubmitButton"] > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    div[data-testid="stButton"] > button:hover, 
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #4338ca !important;
        color: #ffffff !important;
        box-shadow: 0 6px 8px -1px rgba(79, 70, 229, 0.4) !important;
        transform: translateY(-2px);
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background-color: #4f46e5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the original dataset"""
    if not os.path.exists(CSV_PATH):
        st.error(f"❌ Data file not found at: {CSV_PATH}")
        st.info("Please ensure 'digital_habits_vs_mental_health.csv' is in the same directory as app.py")
        return None
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def train_models_if_missing():
    """Train models if they don't exist (for cloud deployment)"""
    if not os.path.exists(os.path.join(MODELS_DIR, 'rf_stress.joblib')):
        with st.spinner("🔧 First-time setup: Training models... This may take a minute."):
            import subprocess
            import sys
            result = subprocess.run([sys.executable, os.path.join(BASE_DIR, 'simple_setup.py')], 
                                   capture_output=True, text=True, cwd=BASE_DIR)
            if result.returncode != 0:
                st.error(f"Model training failed: {result.stderr}")
                return False
        st.success("✅ Models trained successfully!")
        st.cache_resource.clear()
    return True

@st.cache_resource
def load_models():
    """Load trained models"""
    # First ensure models exist
    train_models_if_missing()
    
    try:
        models = {}
        models['rf_stress'] = joblib.load(os.path.join(MODELS_DIR, 'rf_stress.joblib'))
        models['xgb_mood'] = joblib.load(os.path.join(MODELS_DIR, 'xgb_mood.joblib'))
        models['kmeans'] = joblib.load(os.path.join(MODELS_DIR, 'kmeans.joblib'))
        models['isolation_forest'] = joblib.load(os.path.join(MODELS_DIR, 'isolation_forest.joblib'))
        
        preprocessing = joblib.load(os.path.join(MODELS_DIR, 'preprocessing.joblib'))
        models['scaler'] = preprocessing['scaler']
        models['feature_lists'] = preprocessing['feature_lists']
        
        models['performance'] = joblib.load(os.path.join(MODELS_DIR, 'model_performance.joblib'))
        
        return models
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found in: {MODELS_DIR}")
        st.info("Please refresh the page to retry model training.")
        return None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def create_features(user_input):
    """Create features from user input"""
    screen_time = user_input['screen_time_hours']
    social_media = user_input['social_media_platforms_used']
    reels_time = user_input['hours_on_Reels']
    sleep_hours = user_input['sleep_hours']
    
    digital_wellness_score = (
        (24 - screen_time) / 24 * 10 +
        (5 - social_media) / 5 * 10 +
        (10 - reels_time) / 10 * 10
    ) / 3
    
    sleep_quality = 1 if (sleep_hours >= 7 and sleep_hours <= 9) else 0
    screen_sleep_ratio = screen_time / sleep_hours if sleep_hours > 0 else 0
    social_media_intensity = social_media * reels_time
    stress_mood_imbalance = abs(user_input['stress_level'] - (10 - user_input['mood_score']))
    
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
    """Make predictions"""
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
    
    # Create DataFrames with proper feature names in correct order
    stress_features_df = pd.DataFrame([[features[f] for f in stress_features_list]], 
                                      columns=stress_features_list)
    mood_features_df = pd.DataFrame([[features[f] for f in mood_features_list]], 
                                   columns=mood_features_list)
    lifestyle_features_df = pd.DataFrame([[features[f] for f in lifestyle_features_list]], 
                                        columns=lifestyle_features_list)
    
    stress_pred = models['rf_stress'].predict(stress_features_df)[0]
    stress_proba = models['rf_stress'].predict_proba(stress_features_df)[0]
    
    mood_pred = models['xgb_mood'].predict(mood_features_df)[0]
    mood_proba = models['xgb_mood'].predict_proba(mood_features_df)[0]
    
    lifestyle_scaled = models['scaler'].transform(lifestyle_features_df)
    cluster_pred = models['kmeans'].predict(lifestyle_scaled)[0]
    
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

def create_digital_wellness_gauge(score):
    """Create a gauge chart for digital wellness score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Digital Wellness Score"},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 7], 'color': "yellow"},
                {'range': [7, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_feature_importance_chart(importance_dict):
    """Create feature importance chart"""
    df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=True)
    
    fig = px.bar(df, x='Importance', y='Feature', orientation='h',
                 title="Feature Importance for Stress Prediction",
                 color='Importance', color_continuous_scale='viridis')
    fig.update_layout(height=400)
    return fig

def main():
    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Digital Habits vs Mental Health</h1>
        <h3>Interactive Mental Health Analysis & Prediction</h3>
        <p class="header-description" style="color: white !important;">Discover how your digital habits impact your mental well-being</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    data = load_data()
    models = load_models()
    
    if models is None:
        st.error("❌ Failed to load models. Please run the setup first.")
        st.info("Run: python simple_setup.py")
        return
    
    # Sidebar navigation with improved styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                padding: 1.5rem; border-radius: 16px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
        <h3 style="margin:0; font-weight: 800; font-size: 1.5rem;">📊 Navigation</h3>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🔮 Predictions", "📈 Analysis", "📊 Data Explorer", "ℹ️ About"]
    )
    
    if page == "🏠 Home":
        show_home_page(data, models)
    elif page == "🔮 Predictions":
        show_predictions_page(models)
    elif page == "📈 Analysis":
        show_analysis_page(data, models)
    elif page == "📊 Data Explorer":
        show_data_explorer_page(data)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(data, models):
    st.markdown("## 🏠 Welcome to Digital Habits vs Mental Health Analysis")
    
    # Key metrics with gradient cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Total Records</h3>
            <h2>{len(data):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Features</h3>
            <h2>{len(data.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if 'rf_stress' in models['performance']:
            acc = models['performance']['rf_stress']['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Stress Model</h3>
                <h2>{acc:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'xgb_mood' in models['performance']:
            acc = models['performance']['xgb_mood']['accuracy']
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎯 Mood Model</h3>
                <h2>{acc:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Dataset overview with tabs
    st.markdown("### 📋 Dataset Overview")
    tab1, tab2, tab3 = st.tabs(["📊 Sample Data", "📈 Statistics", "🔍 Quick Insights"])
    
    with tab1:
        st.dataframe(data.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(data.describe(), use_container_width=True)
    
    with tab3:
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

def show_predictions_page(models):
    st.markdown("## 🔮 Mental Health Predictions")
    st.markdown("Enter your digital habits to get personalized mental health insights.")
    
    # User input form with improved styling
    with st.form("prediction_form"):
        st.markdown("### 📝 Your Digital Habits")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📱 Digital Usage**")
            screen_time = st.slider("Screen Time (hours/day)", 0.0, 24.0, 6.0, 0.1,
                                   help="Total time spent on screens including phone, computer, tablet")
            social_media = st.slider("Social Media Platforms Used", 0, 10, 3, 1,
                                   help="Number of different social media platforms you use regularly")
            reels_time = st.slider("Hours on Reels/Short Videos", 0.0, 10.0, 2.0, 0.1,
                                 help="Time spent on short-form video content")
        
        with col2:
            st.markdown("**😴 Sleep & Wellness**")
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.1,
                                  help="Average hours of sleep per night")
            stress_level = st.slider("Current Stress Level (1-10)", 1, 10, 5, 1,
                                   help="Rate your current stress level from 1 (low) to 10 (high)")
            mood_score = st.slider("Current Mood Score (1-10)", 1, 10, 7, 1,
                                 help="Rate your current mood from 1 (poor) to 10 (excellent)")
        
        submitted = st.form_submit_button("🔮 Get Predictions")
    
    if submitted:
        user_input = {
            'screen_time_hours': screen_time,
            'social_media_platforms_used': social_media,
            'hours_on_Reels': reels_time,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'mood_score': mood_score
        }
        
        predictions = predict_mental_health(user_input, models)
        
        st.markdown("## 📊 Your Mental Health Analysis")
        
        # Digital Wellness Gauge
        wellness_score = predictions['features']['digital_wellness_score']
        st.plotly_chart(create_digital_wellness_gauge(wellness_score), use_container_width=True)
        
        # Predictions in cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="prediction-box">
                <h3>🧠 Stress Level Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if predictions['stress_prediction'] == 1:
                st.error("⚠️ **High Stress Detected**")
                st.write("Your digital habits suggest high stress levels.")
            else:
                st.success("✅ **Low Stress Detected**")
                st.write("Your digital habits suggest manageable stress levels.")
            st.write(f"Confidence: {max(predictions['stress_probability']):.1%}")
        
        with col2:
            st.markdown("""
            <div class="prediction-box">
                <h3>😊 Mood Prediction</h3>
            </div>
            """, unsafe_allow_html=True)
            
            mood_labels = {0: "Low", 1: "Medium", 2: "High"}
            mood_pred = mood_labels.get(predictions['mood_prediction'], "Unknown")
            st.write(f"**{mood_pred} Mood**")
            st.write(f"Confidence: {max(predictions['mood_probability']):.1%}")
        
        # Lifestyle cluster
        st.markdown("""
        <div class="cluster-box">
            <h3>🎯 Lifestyle Cluster Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        cluster_descriptions = {
            0: "Balanced Digital Lifestyle - Moderate screen time with good sleep patterns",
            1: "High Digital Engagement - Extensive social media use with potential sleep impact",
            2: "Digital Wellness Focus - Low screen time with healthy sleep habits",
            3: "Stress-Prone Pattern - High screen time with poor sleep quality"
        }
        cluster = predictions['cluster']
        st.write(f"**Cluster {cluster}**: {cluster_descriptions.get(cluster, 'Unknown pattern')}")
        
        # Anomaly detection
        st.markdown("""
        <div class="anomaly-box">
            <h3>🔍 Anomaly Detection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if predictions['is_anomaly']:
            st.warning("⚠️ **Anomalous Pattern Detected**")
            st.write("Your digital habits show unusual patterns compared to the general population.")
        else:
            st.success("✅ **Normal Pattern**")
            st.write("Your digital habits are within normal ranges.")
        st.write(f"Anomaly Score: {predictions['anomaly_score']:.3f}")
        
        # Recommendations
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
                st.markdown(f"""
                <div class="recommendation-card">
                    <p>{rec}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🎉 Great job! Your digital habits are well-balanced.")

def show_analysis_page(data, models):
    st.markdown("## 📈 Data Analysis")
    
    # Model performance
    st.markdown("### 🎯 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'rf_stress' in models['performance']:
            perf = models['performance']['rf_stress']
            st.metric("Random Forest Accuracy", f"{perf['accuracy']:.2%}")
    
    with col2:
        if 'xgb_mood' in models['performance']:
            perf = models['performance']['xgb_mood']
            st.metric("XGBoost Accuracy", f"{perf['accuracy']:.2%}")
    
    # Feature importance
    st.markdown("### 🔍 Feature Importance")
    
    if 'rf_stress' in models['performance']:
        rf_importance = models['performance']['rf_stress']['feature_importance']
        st.plotly_chart(create_feature_importance_chart(rf_importance), use_container_width=True)

def show_data_explorer_page(data):
    st.markdown("## 📊 Data Explorer")
    
    # Correlation heatmap
    st.markdown("### 🔗 Feature Correlations")
    corr_matrix = data.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap of Features",
        color_continuous_scale='RdBu'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive scatter plots
    st.markdown("### 📈 Interactive Scatter Plots")
    
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
            opacity=0.6,
            color=y_feature,
            color_continuous_scale='viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.markdown("### 📊 Feature Distributions")
    selected_feature = st.selectbox("Select a feature to visualize:", data.columns.tolist())
    
    fig = px.histogram(
        data, 
        x=selected_feature,
        title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
        nbins=30,
        color_discrete_sequence=['#667eea']
    )
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This application analyzes the relationship between digital habits and mental health outcomes.
    Using machine learning techniques, we can predict stress levels and mood severity based on
    digital behavior patterns.
    
    ### 🔬 Methodology
    
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
    
    1. Run the setup: `python simple_setup.py`
    2. Start the web app: `streamlit run app.py`
    3. Enter your digital habits to get personalized insights
    
    ### 📝 Disclaimer
    
    This application is for educational and research purposes only.
    It should not be used as a substitute for professional medical advice.
    If you're experiencing mental health concerns, please consult with a healthcare professional.
    """)

if __name__ == "__main__":
    main()
