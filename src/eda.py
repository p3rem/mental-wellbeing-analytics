import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class MentalHealthEDA:
    def __init__(self, data_path):
        """Initialize EDA with data path"""
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the CSV data"""
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully! Shape: {self.df.shape}")
    
    def basic_info(self):
        """Display basic dataset information"""
        print("=" * 50)
        print("BASIC DATASET INFORMATION")
        print("=" * 50)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        
        print("\nColumn Information:")
        print(self.df.info())
        
        print("\nMissing Values:")
        missing_values = self.df.isnull().sum()
        if missing_values.sum() == 0:
            print("No missing values found!")
        else:
            print(missing_values[missing_values > 0])
        
        print("\nData Types:")
        print(self.df.dtypes)
    
    def numerical_summary(self):
        """Display numerical column summaries"""
        print("\n" + "=" * 50)
        print("NUMERICAL COLUMN SUMMARIES")
        print("=" * 50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        return numerical_cols
    
    def categorical_summary(self):
        """Display categorical column summaries"""
        print("\n" + "=" * 50)
        print("CATEGORICAL COLUMN SUMMARIES")
        print("=" * 50)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) == 0:
            print("No categorical columns found!")
            return []
        
        for col in categorical_cols:
            print(f"\n{col}:")
            print(self.df[col].value_counts())
        
        return categorical_cols
    
    def create_distribution_plots(self):
        """Create distribution plots for key variables"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution of Key Variables', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Plot distributions
        columns = ['screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels', 
                  'sleep_hours', 'stress_level', 'mood_score']
        
        for i, col in enumerate(columns):
            if i < len(axes):
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col.replace("_", " ").title()}')
                axes[i].set_xlabel(col.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap for numerical features"""
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix
        corr_matrix = self.df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Correlation Heatmap of Numerical Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def create_interactive_plots(self):
        """Create interactive plots using plotly"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Screen Time Hours', 'Social Media Platforms', 'Hours on Reels',
                          'Sleep Hours', 'Stress Level', 'Mood Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add histograms
        columns = ['screen_time_hours', 'social_media_platforms_used', 'hours_on_Reels', 
                  'sleep_hours', 'stress_level', 'mood_score']
        
        for i, col in enumerate(columns):
            row = (i // 3) + 1
            col_pos = (i % 3) + 1
            
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col.replace('_', ' ').title()),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title_text="Interactive Distribution Plots",
            showlegend=False,
            height=800
        )
        
        fig.write_html('interactive_distributions.html')
        return fig
    
    def stress_mood_analysis(self):
        """Analyze relationship between stress and mood"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(self.df['stress_level'], self.df['mood_score'], alpha=0.6)
        axes[0].set_xlabel('Stress Level')
        axes[0].set_ylabel('Mood Score')
        axes[0].set_title('Stress Level vs Mood Score')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        self.df.boxplot(column='mood_score', by='stress_level', ax=axes[1])
        axes[1].set_xlabel('Stress Level')
        axes[1].set_ylabel('Mood Score')
        axes[1].set_title('Mood Score Distribution by Stress Level')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stress_mood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Correlation
        correlation = self.df['stress_level'].corr(self.df['mood_score'])
        print(f"\nCorrelation between Stress Level and Mood Score: {correlation:.3f}")
        
        return correlation
    
    def digital_habits_analysis(self):
        """Analyze digital habits patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Screen time vs Sleep hours
        axes[0, 0].scatter(self.df['screen_time_hours'], self.df['sleep_hours'], alpha=0.6)
        axes[0, 0].set_xlabel('Screen Time Hours')
        axes[0, 0].set_ylabel('Sleep Hours')
        axes[0, 0].set_title('Screen Time vs Sleep Hours')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Social media platforms vs Hours on Reels
        axes[0, 1].scatter(self.df['social_media_platforms_used'], self.df['hours_on_Reels'], alpha=0.6)
        axes[0, 1].set_xlabel('Social Media Platforms Used')
        axes[0, 1].set_ylabel('Hours on Reels')
        axes[0, 1].set_title('Social Media Platforms vs Hours on Reels')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Screen time distribution by stress level
        self.df.boxplot(column='screen_time_hours', by='stress_level', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Stress Level')
        axes[1, 0].set_ylabel('Screen Time Hours')
        axes[1, 0].set_title('Screen Time by Stress Level')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sleep hours distribution by mood score
        self.df.boxplot(column='sleep_hours', by='mood_score', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Mood Score')
        axes[1, 1].set_ylabel('Sleep Hours')
        axes[1, 1].set_title('Sleep Hours by Mood Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('digital_habits_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_eda(self):
        """Run complete EDA analysis"""
        print("Starting Comprehensive EDA Analysis...")
        
        # Basic information
        self.basic_info()
        
        # Summaries
        numerical_cols = self.numerical_summary()
        categorical_cols = self.categorical_summary()
        
        # Visualizations
        print("\nCreating distribution plots...")
        self.create_distribution_plots()
        
        print("Creating correlation heatmap...")
        corr_matrix = self.create_correlation_heatmap()
        
        print("Creating interactive plots...")
        self.create_interactive_plots()
        
        print("Analyzing stress-mood relationship...")
        stress_mood_corr = self.stress_mood_analysis()
        
        print("Analyzing digital habits...")
        self.digital_habits_analysis()
        
        print("\nEDA Analysis Complete!")
        print("Generated files:")
        print("- distributions.png")
        print("- correlation_heatmap.png")
        print("- interactive_distributions.html")
        print("- stress_mood_analysis.png")
        print("- digital_habits_analysis.png")
        
        return {
            'dataframe': self.df,
            'numerical_columns': numerical_cols,
            'categorical_columns': categorical_cols,
            'correlation_matrix': corr_matrix,
            'stress_mood_correlation': stress_mood_corr
        }

if __name__ == "__main__":
    # Run EDA
    eda = MentalHealthEDA('digital_habits_vs_mental_health.csv')
    results = eda.run_complete_eda()
