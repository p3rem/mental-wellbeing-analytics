import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

class MentalHealthPreprocessor:
    def __init__(self, df):
        """Initialize preprocessor with dataframe"""
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_columns = None
        
    def create_target_variables(self):
        """Create target variables for classification tasks"""
        print("Creating target variables...")
        
        # Create binary classification target for stress level
        # High stress: stress_level >= 7, Low stress: stress_level < 7
        self.df['high_stress'] = (self.df['stress_level'] >= 7).astype(int)
        
        # Create multi-class target for mood severity
        # Low mood: mood_score <= 5, Medium mood: 6-7, High mood: >= 8
        self.df['mood_severity'] = pd.cut(
            self.df['mood_score'], 
            bins=[0, 5, 7, 10], 
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        # Convert to numeric for modeling
        mood_severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        self.df['mood_severity_numeric'] = self.df['mood_severity'].map(mood_severity_map)
        
        print(f"Target variables created:")
        print(f"- High stress distribution: {self.df['high_stress'].value_counts().to_dict()}")
        print(f"- Mood severity distribution: {self.df['mood_severity'].value_counts().to_dict()}")
        
        return self.df
    
    def create_features(self):
        """Create additional features for better modeling"""
        print("Creating additional features...")
        
        # Digital wellness score (inverse relationship with screen time and social media)
        self.df['digital_wellness_score'] = (
            (24 - self.df['screen_time_hours']) / 24 * 10 +  # More screen time = lower wellness
            (5 - self.df['social_media_platforms_used']) / 5 * 10 +  # More platforms = lower wellness
            (10 - self.df['hours_on_Reels']) / 10 * 10  # More reels time = lower wellness
        ) / 3
        
        # Sleep quality indicator
        self.df['sleep_quality'] = np.where(
            (self.df['sleep_hours'] >= 7) & (self.df['sleep_hours'] <= 9),
            1,  # Good sleep
            0   # Poor sleep
        )
        
        # Screen time to sleep ratio (handle division by zero)
        self.df['screen_sleep_ratio'] = np.where(
            self.df['sleep_hours'] > 0,
            self.df['screen_time_hours'] / self.df['sleep_hours'],
            0
        )
        
        # Social media intensity (platforms * hours on reels)
        self.df['social_media_intensity'] = (
            self.df['social_media_platforms_used'] * self.df['hours_on_Reels']
        )
        
        # Stress-mood imbalance
        self.df['stress_mood_imbalance'] = abs(
            self.df['stress_level'] - (10 - self.df['mood_score'])
        )
        
        print("Additional features created:")
        print("- digital_wellness_score")
        print("- sleep_quality")
        print("- screen_sleep_ratio")
        print("- social_media_intensity")
        print("- stress_mood_imbalance")
        
        return self.df
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Checking for missing values...")
        
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found!")
            return self.df
        
        print("Missing values found:")
        print(missing_values[missing_values > 0])
        
        # For numerical columns, fill with median
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # For categorical columns, fill with mode
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
        
        print("Missing values handled!")
        return self.df
    
    def encode_categorical_variables(self):
        """Encode categorical variables"""
        print("Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['mood_severity']:  # Skip the original mood_severity column
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col])
                self.label_encoders[col] = le
                print(f"Encoded {col} -> {col}_encoded")
        
        return self.df
    
    def select_features(self):
        """Select features for modeling"""
        print("Selecting features for modeling...")
        
        # Features for lifestyle clustering
        self.lifestyle_features = [
            'screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels',
            'sleep_hours', 'digital_wellness_score', 'sleep_quality',
            'screen_sleep_ratio', 'social_media_intensity'
        ]
        
        # Features for stress prediction
        self.stress_features = [
            'screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels',
            'sleep_hours', 'digital_wellness_score', 'sleep_quality',
            'screen_sleep_ratio', 'social_media_intensity', 'stress_mood_imbalance'
        ]
        
        # Features for mood prediction
        self.mood_features = [
            'screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels',
            'sleep_hours', 'digital_wellness_score', 'sleep_quality',
            'screen_sleep_ratio', 'social_media_intensity', 'stress_mood_imbalance'
        ]
        
        print(f"Selected {len(self.lifestyle_features)} lifestyle features for clustering")
        print(f"Selected {len(self.stress_features)} features for stress prediction")
        print(f"Selected {len(self.mood_features)} features for mood prediction")
        
        return self.df
    
    def scale_features(self):
        """Scale numerical features"""
        print("Scaling numerical features...")
        
        # Scale lifestyle features for clustering
        self.lifestyle_features_scaled = self.scaler.fit_transform(
            self.df[self.lifestyle_features]
        )
        
        # Create scaled dataframe
        self.df_scaled = pd.DataFrame(
            self.lifestyle_features_scaled,
            columns=self.lifestyle_features,
            index=self.df.index
        )
        
        print("Features scaled successfully!")
        return self.df_scaled
    
    def prepare_data_splits(self, test_size=0.2, random_state=42):
        """Prepare train-test splits for different models"""
        print("Preparing data splits...")
        
        # Prepare data for stress prediction
        X_stress = self.df[self.stress_features]
        y_stress = self.df['high_stress']
        
        X_stress_train, X_stress_test, y_stress_train, y_stress_test = train_test_split(
            X_stress, y_stress, test_size=test_size, random_state=random_state, stratify=y_stress
        )
        
        # Prepare data for mood prediction
        X_mood = self.df[self.mood_features]
        y_mood = self.df['mood_severity_numeric']
        
        X_mood_train, X_mood_test, y_mood_train, y_mood_test = train_test_split(
            X_mood, y_mood, test_size=test_size, random_state=random_state, stratify=y_mood
        )
        
        print("Data splits prepared:")
        print(f"Stress prediction - Train: {X_stress_train.shape}, Test: {X_stress_test.shape}")
        print(f"Mood prediction - Train: {X_mood_train.shape}, Test: {X_mood_test.shape}")
        
        return {
            'stress': {
                'X_train': X_stress_train, 'X_test': X_stress_test,
                'y_train': y_stress_train, 'y_test': y_stress_test
            },
            'mood': {
                'X_train': X_mood_train, 'X_test': X_mood_test,
                'y_train': y_mood_train, 'y_test': y_mood_test
            }
        }
    
    def detect_anomalies(self, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        print("Detecting anomalies using Isolation Forest...")
        
        # Use lifestyle features for anomaly detection
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(self.df_scaled)
        anomaly_scores = iso_forest.decision_function(self.df_scaled)
        
        # Add anomaly information to dataframe
        self.df['anomaly_label'] = anomaly_labels
        self.df['anomaly_score'] = anomaly_scores
        self.df['is_anomaly'] = (anomaly_labels == -1).astype(int)
        
        n_anomalies = self.df['is_anomaly'].sum()
        print(f"Detected {n_anomalies} anomalies ({n_anomalies/len(self.df)*100:.2f}% of data)")
        
        return iso_forest, self.df
    
    def run_complete_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("=" * 50)
        print("STARTING COMPLETE PREPROCESSING PIPELINE")
        print("=" * 50)
        
        # Step 1: Create target variables
        self.create_target_variables()
        
        # Step 2: Create additional features
        self.create_features()
        
        # Step 3: Handle missing values
        self.handle_missing_values()
        
        # Step 4: Encode categorical variables
        self.encode_categorical_variables()
        
        # Step 5: Select features
        self.select_features()
        
        # Step 6: Scale features
        self.scale_features()
        
        # Step 7: Prepare data splits
        data_splits = self.prepare_data_splits()
        
        # Step 8: Detect anomalies
        iso_forest, df_with_anomalies = self.detect_anomalies()
        
        print("\n" + "=" * 50)
        print("PREPROCESSING PIPELINE COMPLETE")
        print("=" * 50)
        
        return {
            'processed_df': self.df,
            'scaled_df': self.df_scaled,
            'data_splits': data_splits,
            'isolation_forest': iso_forest,
            'df_with_anomalies': df_with_anomalies,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_lists': {
                'lifestyle': self.lifestyle_features,
                'stress': self.stress_features,
                'mood': self.mood_features
            }
        }

if __name__ == "__main__":
    # Test preprocessing
    from eda import MentalHealthEDA
    
    # Load data
    eda = MentalHealthEDA('digital_habits_vs_mental_health.csv')
    df = eda.df
    
    # Run preprocessing
    preprocessor = MentalHealthPreprocessor(df)
    results = preprocessor.run_complete_preprocessing()
    
    print("Preprocessing completed successfully!")
