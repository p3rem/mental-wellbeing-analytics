import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    silhouette_score, mean_squared_error, r2_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class MentalHealthModelTrainer:
    def __init__(self, preprocessing_results):
        """Initialize model trainer with preprocessing results"""
        self.preprocessing_results = preprocessing_results
        self.models = {}
        self.model_performance = {}
        
    def train_random_forest_stress(self):
        """Train Random Forest for stress prediction"""
        print("Training Random Forest for stress prediction...")
        
        # Get data splits
        data_splits = self.preprocessing_results['data_splits']['stress']
        X_train = data_splits['X_train']
        X_test = data_splits['X_test']
        y_train = data_splits['y_train']
        y_test = data_splits['y_test']
        
        # Initialize and train Random Forest
        rf_stress = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Train the model
        rf_stress.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_stress.predict(X_test)
        y_pred_proba = rf_stress.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(rf_stress, X_train, y_train, cv=5)
        
        print(f"Random Forest Stress Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(classification_rep)
        
        # Store model and performance
        self.models['rf_stress'] = rf_stress
        self.model_performance['rf_stress'] = {
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'feature_importance': dict(zip(X_train.columns, rf_stress.feature_importances_))
        }
        
        return rf_stress
    
    def train_xgboost_mood(self):
        """Train XGBoost for mood prediction"""
        print("Training XGBoost for mood prediction...")
        
        # Get data splits
        data_splits = self.preprocessing_results['data_splits']['mood']
        X_train = data_splits['X_train']
        X_test = data_splits['X_test']
        y_train = data_splits['y_train']
        y_test = data_splits['y_test']
        
        # Initialize and train XGBoost
        xgb_mood = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train the model
        xgb_mood.fit(X_train, y_train)
        
        # Make predictions
        y_pred = xgb_mood.predict(X_test)
        y_pred_proba = xgb_mood.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(xgb_mood, X_train, y_train, cv=5)
        
        print(f"XGBoost Mood Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print("\nClassification Report:")
        print(classification_rep)
        
        # Store model and performance
        self.models['xgb_mood'] = xgb_mood
        self.model_performance['xgb_mood'] = {
            'accuracy': accuracy,
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'feature_importance': dict(zip(X_train.columns, xgb_mood.feature_importances_))
        }
        
        return xgb_mood
    
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using Elbow Method and Silhouette Score"""
        print("Finding optimal number of clusters...")
        
        scaled_df = self.preprocessing_results['scaled_df']
        lifestyle_features = self.preprocessing_results['feature_lists']['lifestyle']
        
        # Calculate inertia and silhouette scores for different numbers of clusters
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_df[lifestyle_features])
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(scaled_df[lifestyle_features], kmeans.labels_))
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow plot
        ax1.plot(K_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score plot
        ax2.plot(K_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score for Optimal k')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find optimal k (highest silhouette score)
        optimal_k = K_range[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette score: {max(silhouette_scores):.4f}")
        
        return optimal_k, inertias, silhouette_scores
    
    def train_kmeans_clustering(self, n_clusters=4):
        """Train K-Means clustering model"""
        print(f"Training K-Means clustering with {n_clusters} clusters...")
        
        scaled_df = self.preprocessing_results['scaled_df']
        lifestyle_features = self.preprocessing_results['feature_lists']['lifestyle']
        
        # Train K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_df[lifestyle_features])
        
        # Add cluster labels to dataframe
        processed_df = self.preprocessing_results['processed_df'].copy()
        processed_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = processed_df.groupby('cluster')[lifestyle_features].mean()
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(scaled_df[lifestyle_features], cluster_labels)
        
        print(f"K-Means Clustering Results:")
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette score: {silhouette_avg:.4f}")
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        # Store model and results
        self.models['kmeans'] = kmeans
        self.model_performance['kmeans'] = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_stats': cluster_stats,
            'cluster_labels': cluster_labels
        }
        
        return kmeans, processed_df
    
    def create_cluster_visualizations(self, processed_df):
        """Create visualizations for clustering results"""
        print("Creating cluster visualizations...")
        
        lifestyle_features = self.preprocessing_results['feature_lists']['lifestyle']
        
        # Create scatter plots for different feature combinations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Screen time vs Sleep hours
        scatter1 = axes[0, 0].scatter(
            processed_df['screen_time_hours'], 
            processed_df['sleep_hours'], 
            c=processed_df['cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[0, 0].set_xlabel('Screen Time Hours')
        axes[0, 0].set_ylabel('Sleep Hours')
        axes[0, 0].set_title('Clusters: Screen Time vs Sleep Hours')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Social media platforms vs Hours on Reels
        scatter2 = axes[0, 1].scatter(
            processed_df['social_media_platforms_used'], 
            processed_df['hours_on_Reels'], 
            c=processed_df['cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[0, 1].set_xlabel('Social Media Platforms Used')
        axes[0, 1].set_ylabel('Hours on Reels')
        axes[0, 1].set_title('Clusters: Social Media vs Reels Time')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Digital wellness vs Sleep quality
        scatter3 = axes[1, 0].scatter(
            processed_df['digital_wellness_score'], 
            processed_df['sleep_quality'], 
            c=processed_df['cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[1, 0].set_xlabel('Digital Wellness Score')
        axes[1, 0].set_ylabel('Sleep Quality')
        axes[1, 0].set_title('Clusters: Digital Wellness vs Sleep Quality')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Screen sleep ratio vs Social media intensity
        scatter4 = axes[1, 1].scatter(
            processed_df['screen_sleep_ratio'], 
            processed_df['social_media_intensity'], 
            c=processed_df['cluster'], 
            cmap='viridis', 
            alpha=0.6
        )
        axes[1, 1].set_xlabel('Screen Sleep Ratio')
        axes[1, 1].set_ylabel('Social Media Intensity')
        axes[1, 1].set_title('Clusters: Screen Sleep Ratio vs Social Media Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, ax=axes, shrink=0.8)
        cbar.set_label('Cluster')
        
        plt.tight_layout()
        plt.savefig('cluster_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance for both models"""
        print("Analyzing feature importance...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest feature importance
        if 'rf_stress' in self.model_performance:
            rf_importance = self.model_performance['rf_stress']['feature_importance']
            rf_features = list(rf_importance.keys())
            rf_scores = list(rf_importance.values())
            
            # Sort by importance
            rf_sorted = sorted(zip(rf_features, rf_scores), key=lambda x: x[1], reverse=True)
            rf_features_sorted, rf_scores_sorted = zip(*rf_sorted)
            
            ax1.barh(range(len(rf_features_sorted)), rf_scores_sorted)
            ax1.set_yticks(range(len(rf_features_sorted)))
            ax1.set_yticklabels([f.replace('_', ' ').title() for f in rf_features_sorted])
            ax1.set_xlabel('Feature Importance')
            ax1.set_title('Random Forest - Stress Prediction')
            ax1.grid(True, alpha=0.3)
        
        # XGBoost feature importance
        if 'xgb_mood' in self.model_performance:
            xgb_importance = self.model_performance['xgb_mood']['feature_importance']
            xgb_features = list(xgb_importance.keys())
            xgb_scores = list(xgb_importance.values())
            
            # Sort by importance
            xgb_sorted = sorted(zip(xgb_features, xgb_scores), key=lambda x: x[1], reverse=True)
            xgb_features_sorted, xgb_scores_sorted = zip(*xgb_sorted)
            
            ax2.barh(range(len(xgb_features_sorted)), xgb_scores_sorted)
            ax2.set_yticks(range(len(xgb_features_sorted)))
            ax2.set_yticklabels([f.replace('_', ' ').title() for f in xgb_features_sorted])
            ax2.set_xlabel('Feature Importance')
            ax2.set_title('XGBoost - Mood Prediction')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_models(self, filepath='models/'):
        """Save all trained models"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        print("Saving models...")
        
        # Save individual models
        for model_name, model in self.models.items():
            model_file = os.path.join(filepath, f'{model_name}.joblib')
            joblib.dump(model, model_file)
            print(f"Saved {model_name} to {model_file}")
        
        # Save preprocessing components
        preprocessing_file = os.path.join(filepath, 'preprocessing.joblib')
        preprocessing_components = {
            'scaler': self.preprocessing_results['scaler'],
            'label_encoders': self.preprocessing_results['label_encoders'],
            'feature_lists': self.preprocessing_results['feature_lists']
        }
        joblib.dump(preprocessing_components, preprocessing_file)
        print(f"Saved preprocessing components to {preprocessing_file}")
        
        # Save isolation forest model
        iso_forest_file = os.path.join(filepath, 'isolation_forest.joblib')
        joblib.dump(self.preprocessing_results['isolation_forest'], iso_forest_file)
        print(f"Saved isolation forest model to {iso_forest_file}")
        
        # Save model performance
        performance_file = os.path.join(filepath, 'model_performance.joblib')
        joblib.dump(self.model_performance, performance_file)
        print(f"Saved model performance to {performance_file}")
        
        print("All models saved successfully!")
    
    def run_complete_training(self):
        """Run complete model training pipeline"""
        print("=" * 50)
        print("STARTING COMPLETE MODEL TRAINING PIPELINE")
        print("=" * 50)
        
        # Step 1: Train Random Forest for stress prediction
        self.train_random_forest_stress()
        
        # Step 2: Train XGBoost for mood prediction
        self.train_xgboost_mood()
        
        # Step 3: Find optimal number of clusters
        optimal_k, inertias, silhouette_scores = self.find_optimal_clusters()
        
        # Step 4: Train K-Means clustering
        kmeans, processed_df = self.train_kmeans_clustering(optimal_k)
        
        # Step 5: Create cluster visualizations
        self.create_cluster_visualizations(processed_df)
        
        # Step 6: Analyze feature importance
        self.analyze_feature_importance()
        
        # Step 7: Save models
        self.save_models()
        
        print("\n" + "=" * 50)
        print("MODEL TRAINING PIPELINE COMPLETE")
        print("=" * 50)
        
        return {
            'models': self.models,
            'model_performance': self.model_performance,
            'processed_df_with_clusters': processed_df,
            'optimal_clusters': optimal_k
        }

if __name__ == "__main__":
    # Test model training
    from eda import MentalHealthEDA
    from preprocessing import MentalHealthPreprocessor
    
    # Load and preprocess data
    eda = MentalHealthEDA('digital_habits_vs_mental_health.csv')
    preprocessor = MentalHealthPreprocessor(eda.df)
    preprocessing_results = preprocessor.run_complete_preprocessing()
    
    # Train models
    trainer = MentalHealthModelTrainer(preprocessing_results)
    results = trainer.run_complete_training()
    
    print("Model training completed successfully!")
