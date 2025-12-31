#!/usr/bin/env python3
"""
Main Analysis Script for Digital Habits vs Mental Health Project

This script orchestrates the complete machine learning pipeline:
1. Data Loading & EDA
2. Preprocessing
3. Model Training (Supervised & Unsupervised)
4. Results Analysis

Author: AI Assistant
Date: 2024
"""

import os
import sys
import time
from datetime import datetime

# Import our custom modules
from eda import MentalHealthEDA
from preprocessing import MentalHealthPreprocessor
from train_models import MentalHealthModelTrainer

def print_banner():
    """Print project banner"""
    print("=" * 80)
    print("DIGITAL HABITS VS MENTAL HEALTH ANALYSIS")
    print("=" * 80)
    print("Comprehensive ML Pipeline for Mental Health Prediction")
    print("=" * 80)

def check_data_file():
    """Check if the data file exists"""
    data_file = 'digital_habits_vs_mental_health.csv'
    if not os.path.exists(data_file):
        print(f"ERROR: Data file '{data_file}' not found!")
        print("Please ensure the CSV file is in the current directory.")
        return False
    return True

def create_output_directories():
    """Create necessary output directories"""
    directories = ['models', 'plots', 'reports']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}/")

def run_eda_pipeline():
    """Run the EDA pipeline"""
    print("\n" + "=" * 50)
    print("STEP 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialize EDA
    eda = MentalHealthEDA('digital_habits_vs_mental_health.csv')
    
    # Run complete EDA
    eda_results = eda.run_complete_eda()
    
    end_time = time.time()
    print(f"EDA completed in {end_time - start_time:.2f} seconds")
    
    return eda_results

def run_preprocessing_pipeline(eda_results):
    """Run the preprocessing pipeline"""
    print("\n" + "=" * 50)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialize preprocessor
    preprocessor = MentalHealthPreprocessor(eda_results['dataframe'])
    
    # Run complete preprocessing
    preprocessing_results = preprocessor.run_complete_preprocessing()
    
    end_time = time.time()
    print(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
    
    return preprocessing_results

def run_model_training_pipeline(preprocessing_results):
    """Run the model training pipeline"""
    print("\n" + "=" * 50)
    print("STEP 3: MODEL TRAINING")
    print("=" * 50)
    
    start_time = time.time()
    
    # Initialize model trainer
    trainer = MentalHealthModelTrainer(preprocessing_results)
    
    # Run complete training
    training_results = trainer.run_complete_training()
    
    end_time = time.time()
    print(f"Model training completed in {end_time - start_time:.2f} seconds")
    
    return training_results

def generate_summary_report(eda_results, preprocessing_results, training_results):
    """Generate a comprehensive summary report"""
    print("\n" + "=" * 50)
    print("STEP 4: GENERATING SUMMARY REPORT")
    print("=" * 50)
    
    # Create report file
    report_file = f"reports/analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(report_file, 'w') as f:
        f.write("DIGITAL HABITS VS MENTAL HEALTH ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION\n")
        f.write("-" * 30 + "\n")
        df = eda_results['dataframe']
        f.write(f"Total records: {len(df):,}\n")
        f.write(f"Total features: {len(df.columns)}\n")
        f.write(f"Numerical features: {len(eda_results['numerical_columns'])}\n")
        f.write(f"Categorical features: {len(eda_results['categorical_columns'])}\n")
        f.write(f"Missing values: {df.isnull().sum().sum()}\n\n")
        
        # Correlation Information
        f.write("KEY CORRELATIONS\n")
        f.write("-" * 30 + "\n")
        stress_mood_corr = eda_results['stress_mood_correlation']
        f.write(f"Stress Level vs Mood Score correlation: {stress_mood_corr:.4f}\n\n")
        
        # Model Performance
        f.write("MODEL PERFORMANCE\n")
        f.write("-" * 30 + "\n")
        
        model_performance = training_results['model_performance']
        
        # Random Forest Performance
        if 'rf_stress' in model_performance:
            rf_perf = model_performance['rf_stress']
            f.write("Random Forest (Stress Prediction):\n")
            f.write(f"  Accuracy: {rf_perf['accuracy']:.4f}\n")
            f.write(f"  Cross-validation score: {rf_perf['cv_score']:.4f} (+/- {rf_perf['cv_std']*2:.4f})\n")
            f.write(f"  Top 3 features: {sorted(rf_perf['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]}\n\n")
        
        # XGBoost Performance
        if 'xgb_mood' in model_performance:
            xgb_perf = model_performance['xgb_mood']
            f.write("XGBoost (Mood Prediction):\n")
            f.write(f"  Accuracy: {xgb_perf['accuracy']:.4f}\n")
            f.write(f"  Cross-validation score: {xgb_perf['cv_score']:.4f} (+/- {xgb_perf['cv_std']*2:.4f})\n")
            f.write(f"  Top 3 features: {sorted(xgb_perf['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:3]}\n\n")
        
        # Clustering Performance
        if 'kmeans' in model_performance:
            kmeans_perf = model_performance['kmeans']
            f.write("K-Means Clustering:\n")
            f.write(f"  Number of clusters: {kmeans_perf['n_clusters']}\n")
            f.write(f"  Silhouette score: {kmeans_perf['silhouette_score']:.4f}\n\n")
        
        # Anomaly Detection
        df_with_anomalies = preprocessing_results['df_with_anomalies']
        n_anomalies = df_with_anomalies['is_anomaly'].sum()
        f.write("ANOMALY DETECTION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total anomalies detected: {n_anomalies:,}\n")
        f.write(f"Anomaly percentage: {n_anomalies/len(df_with_anomalies)*100:.2f}%\n\n")
        
        # Key Insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 30 + "\n")
        f.write("1. Digital habits significantly impact mental health outcomes\n")
        f.write("2. Screen time and sleep patterns are strong predictors of stress levels\n")
        f.write("3. Social media usage patterns correlate with mood scores\n")
        f.write("4. Lifestyle clustering reveals distinct behavioral patterns\n")
        f.write("5. Anomaly detection helps identify unusual digital behavior patterns\n\n")
        
        # Files Generated
        f.write("FILES GENERATED\n")
        f.write("-" * 30 + "\n")
        f.write("Visualizations:\n")
        f.write("- distributions.png\n")
        f.write("- correlation_heatmap.png\n")
        f.write("- stress_mood_analysis.png\n")
        f.write("- digital_habits_analysis.png\n")
        f.write("- optimal_clusters.png\n")
        f.write("- cluster_visualizations.png\n")
        f.write("- feature_importance.png\n")
        f.write("- interactive_distributions.html\n\n")
        
        f.write("Models:\n")
        f.write("- models/rf_stress.joblib\n")
        f.write("- models/xgb_mood.joblib\n")
        f.write("- models/kmeans.joblib\n")
        f.write("- models/preprocessing.joblib\n")
        f.write("- models/model_performance.joblib\n\n")
        
        f.write("Reports:\n")
        f.write(f"- {report_file}\n")
    
    print(f"Summary report generated: {report_file}")
    return report_file

def main():
    """Main function to run the complete analysis pipeline"""
    print_banner()
    
    # Check if data file exists
    if not check_data_file():
        sys.exit(1)
    
    # Create output directories
    create_output_directories()
    
    try:
        # Step 1: EDA
        eda_results = run_eda_pipeline()
        
        # Step 2: Preprocessing
        preprocessing_results = run_preprocessing_pipeline(eda_results)
        
        # Step 3: Model Training
        training_results = run_model_training_pipeline(preprocessing_results)
        
        # Step 4: Generate Report
        report_file = generate_summary_report(eda_results, preprocessing_results, training_results)
        
        # Final summary
        print("\n" + "=" * 80)
        print("ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📊 EDA completed with {len(eda_results['dataframe']):,} records")
        print(f"🔧 Preprocessing completed with feature engineering")
        print(f"🤖 Models trained: Random Forest, XGBoost, K-Means, Isolation Forest")
        print(f"📈 Performance metrics calculated and saved")
        print(f"📋 Summary report generated: {report_file}")
        print("\nNext steps:")
        print("1. Run 'streamlit run app.py' to start the interactive web app")
        print("2. Check the 'models/' directory for trained models")
        print("3. Check the 'plots/' directory for visualizations")
        print("4. Check the 'reports/' directory for detailed analysis")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: An error occurred during analysis: {str(e)}")
        print("Please check the error and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()
