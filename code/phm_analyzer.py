"""
PHM Challenge 2010 - Tool Wear Prediction Model
================================================

This script extracts features from sensor data and builds a predictive model
for tool wear estimation.

Dataset Structure:
- c1_wear: Target wear values (315 cuts × 3 flutes)
- c_1_XXX: Sensor data for each cut (220k samples × 7 sensors)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

class PHMToolWearPredictor:
    """
    Complete tool wear prediction system
    """
    
    def __init__(self, data_path):
        """
        Initialize predictor
        
        Parameters:
        -----------
        data_path : str
            Path to PHM dataset directory
        """
        self.data_path = Path(data_path)
        self.wear_data = None
        self.features_df = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_wear_data(self):
        """
        Load the wear measurement file (c1_wear)
        """
        print("=" * 80)
        print("LOADING TOOL WEAR DATA")
        print("=" * 80)
        
        wear_file = self.data_path / "c1_wear.csv"
        
        if not wear_file.exists():
            print(f"❌ Wear file not found: {wear_file}")
            return False
        
        self.wear_data = pd.read_csv(wear_file)
        print(f"\n✓ Loaded wear data: {self.wear_data.shape}")
        print(f"\nWear statistics:")
        print(self.wear_data.describe())
        
        # Calculate average wear across flutes
        self.wear_data['avg_wear'] = self.wear_data[['flute_1', 'flute_2', 'flute_3']].mean(axis=1)
        
        return True
    
    def extract_features_from_cut(self, cut_number):
        """
        Extract statistical and frequency features from one cutting operation
        
        Parameters:
        -----------
        cut_number : int
            Cut number (1-315)
            
        Returns:
        --------
        dict : Feature dictionary
        """
        # Load sensor data for this cut
        sensor_file = self.data_path / f"c_1_{cut_number}.csv"
        
        if not sensor_file.exists():
            print(f"⚠️ File not found: {sensor_file.name}")
            return None
        
        try:
            # Load sensor data
            df = pd.read_csv(sensor_file)
            
            features = {'cut': cut_number}
            
            # For each sensor column
            for col_idx, col in enumerate(df.columns):
                col_name = f"sensor_{col_idx+1}"
                data = df[col].values
                
                # Time domain features
                features[f'{col_name}_mean'] = np.mean(data)
                features[f'{col_name}_std'] = np.std(data)
                features[f'{col_name}_min'] = np.min(data)
                features[f'{col_name}_max'] = np.max(data)
                features[f'{col_name}_range'] = np.max(data) - np.min(data)
                features[f'{col_name}_rms'] = np.sqrt(np.mean(data**2))
                features[f'{col_name}_skewness'] = stats.skew(data)
                features[f'{col_name}_kurtosis'] = stats.kurtosis(data)
                features[f'{col_name}_peak'] = np.max(np.abs(data))
                
                # Percentiles
                features[f'{col_name}_p25'] = np.percentile(data, 25)
                features[f'{col_name}_p75'] = np.percentile(data, 75)
                
                # Frequency domain features (using first 10000 samples for speed)
                sample_data = data[:10000]
                fft_vals = np.abs(fft(sample_data))
                fft_vals = fft_vals[:len(fft_vals)//2]  # Take positive frequencies
                
                features[f'{col_name}_fft_mean'] = np.mean(fft_vals)
                features[f'{col_name}_fft_max'] = np.max(fft_vals)
                features[f'{col_name}_fft_energy'] = np.sum(fft_vals**2)
                
            return features
            
        except Exception as e:
            print(f"❌ Error processing cut {cut_number}: {str(e)}")
            return None
    
    def extract_all_features(self, max_cuts=None):
        """
        Extract features from all cutting operations
        
        Parameters:
        -----------
        max_cuts : int, optional
            Maximum number of cuts to process (for testing)
        """
        print("\n" + "=" * 80)
        print("EXTRACTING FEATURES FROM SENSOR DATA")
        print("=" * 80)
        
        if self.wear_data is None:
            print("❌ Please load wear data first!")
            return False
        
        n_cuts = len(self.wear_data) if max_cuts is None else min(max_cuts, len(self.wear_data))
        
        print(f"\nProcessing {n_cuts} cuts...")
        print("This may take a while...\n")
        
        features_list = []
        
        for i, cut_num in enumerate(range(1, n_cuts + 1)):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{n_cuts} cuts...")
            
            features = self.extract_features_from_cut(cut_num)
            if features is not None:
                features_list.append(features)
        
        # Create features dataframe
        self.features_df = pd.DataFrame(features_list)
        
        # Merge with wear data
        self.features_df = self.features_df.merge(
            self.wear_data[['cut', 'flute_1', 'flute_2', 'flute_3', 'avg_wear']], 
            on='cut'
        )
        
        print(f"\n✓ Feature extraction complete!")
        print(f"   Features shape: {self.features_df.shape}")
        print(f"   Total features: {len(self.features_df.columns) - 5}")  # Exclude cut and wear columns
        
        return True
    
    def save_features(self, filename='phm_features.csv'):
        """
        Save extracted features to CSV
        """
        if self.features_df is not None:
            self.features_df.to_csv(filename, index=False)
            print(f"✓ Features saved to {filename}")
    
    def load_features(self, filename='phm_features.csv'):
        """
        Load previously extracted features
        """
        if Path(filename).exists():
            self.features_df = pd.read_csv(filename)
            print(f"✓ Features loaded from {filename}")
            print(f"   Shape: {self.features_df.shape}")
            return True
        else:
            print(f"❌ Features file not found: {filename}")
            return False
    
    def visualize_wear_progression(self):
        """
        Visualize tool wear over time
        """
        if self.wear_data is None:
            print("❌ No wear data loaded!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Wear progression for each flute
        ax = axes[0, 0]
        ax.plot(self.wear_data['cut'], self.wear_data['flute_1'], label='Flute 1', marker='o', markersize=2)
        ax.plot(self.wear_data['cut'], self.wear_data['flute_2'], label='Flute 2', marker='s', markersize=2)
        ax.plot(self.wear_data['cut'], self.wear_data['flute_3'], label='Flute 3', marker='^', markersize=2)
        ax.set_xlabel('Cut Number')
        ax.set_ylabel('Wear (μm)')
        ax.set_title('Tool Wear Progression - Individual Flutes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Average wear with confidence band
        ax = axes[0, 1]
        avg_wear = self.wear_data[['flute_1', 'flute_2', 'flute_3']].mean(axis=1)
        std_wear = self.wear_data[['flute_1', 'flute_2', 'flute_3']].std(axis=1)
        ax.plot(self.wear_data['cut'], avg_wear, 'b-', linewidth=2, label='Average Wear')
        ax.fill_between(self.wear_data['cut'], avg_wear-std_wear, avg_wear+std_wear, alpha=0.3)
        ax.set_xlabel('Cut Number')
        ax.set_ylabel('Wear (μm)')
        ax.set_title('Average Tool Wear with Std Dev')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Wear rate (derivative)
        ax = axes[1, 0]
        wear_rate = np.gradient(avg_wear)
        ax.plot(self.wear_data['cut'], wear_rate, 'r-', linewidth=1)
        ax.set_xlabel('Cut Number')
        ax.set_ylabel('Wear Rate (μm/cut)')
        ax.set_title('Tool Wear Rate Over Time')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Wear distribution
        ax = axes[1, 1]
        ax.hist(self.wear_data['flute_1'], alpha=0.5, bins=30, label='Flute 1')
        ax.hist(self.wear_data['flute_2'], alpha=0.5, bins=30, label='Flute 2')
        ax.hist(self.wear_data['flute_3'], alpha=0.5, bins=30, label='Flute 3')
        ax.set_xlabel('Wear (μm)')
        ax.set_ylabel('Frequency')
        ax.set_title('Wear Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wear_progression_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Wear progression plot saved as 'wear_progression_analysis.png'")
        plt.show()
    
    def train_models(self, target='avg_wear', test_size=0.2):
        """
        Train multiple machine learning models
        
        Parameters:
        -----------
        target : str
            Target column to predict ('avg_wear', 'flute_1', 'flute_2', or 'flute_3')
        test_size : float
            Proportion of data for testing
        """
        print("\n" + "=" * 80)
        print(f"TRAINING MODELS TO PREDICT: {target}")
        print("=" * 80)
        
        if self.features_df is None:
            print("❌ No features available. Please extract features first!")
            return
        
        # Prepare data
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['cut', 'flute_1', 'flute_2', 'flute_3', 'avg_wear']]
        
        X = self.features_df[feature_cols]
        y = self.features_df[target]
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Samples: {len(X)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = []
        
        print("\nTraining models...\n")
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train
            if name == 'Linear Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            })
            
            # Store model
            self.models[name] = model
            
            print(f"  RMSE: {rmse:.4f} μm")
            print(f"  MAE:  {mae:.4f} μm")
            print(f"  R²:   {r2:.4f}\n")
        
        # Display results
        results_df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        # Visualize predictions
        self._visualize_predictions(X_test, y_test, target)
        
        return results_df
    
    def _visualize_predictions(self, X_test, y_test, target):
        """
        Visualize model predictions vs actual values
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, model) in enumerate(self.models.items()):
            ax = axes[idx]
            
            if name == 'Linear Regression':
                X_test_scaled = self.scaler.transform(X_test)
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            # Scatter plot
            ax.scatter(y_test, y_pred, alpha=0.5, s=30)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            # Calculate R²
            r2 = r2_score(y_test, y_pred)
            
            ax.set_xlabel('Actual Wear (μm)')
            ax.set_ylabel('Predicted Wear (μm)')
            ax.set_title(f'{name}\nR² = {r2:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Prediction plots saved as 'model_predictions.png'")
        plt.show()
    
    def feature_importance_analysis(self, model_name='Random Forest', top_n=20):
        """
        Analyze and visualize feature importance
        """
        if model_name not in self.models:
            print(f"❌ Model '{model_name}' not found!")
            return
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"❌ Model '{model_name}' doesn't have feature importance!")
            return
        
        # Get feature names
        feature_cols = [col for col in self.features_df.columns 
                       if col not in ['cut', 'flute_1', 'flute_2', 'flute_3', 'avg_wear']]
        
        # Get importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_cols[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features - {model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved as 'feature_importance.png'")
        plt.show()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main execution function
    """
    print("\n" + "🔧" * 40)
    print("PHM TOOL WEAR PREDICTION SYSTEM")
    print("🔧" * 40 + "\n")
    
    # Set dataset path
    dataset_path = r"E:/Collaboration Work/With Farooq/phm dataset/PHM Challange 2010 Milling/c1"
    
    # Initialize predictor
    predictor = PHMToolWearPredictor(dataset_path)
    
    # Load wear data
    if not predictor.load_wear_data():
        print("Failed to load wear data. Exiting.")
        return
    
    # Visualize wear progression
    predictor.visualize_wear_progression()
    
    # Extract features (use max_cuts parameter for testing with subset)
    # For full dataset, remove max_cuts parameter or set to None
    print("\n⚠️ NOTE: Feature extraction may take 10-30 minutes for all 315 cuts")
    print("For testing, using first 50 cuts. Remove max_cuts parameter for full dataset.\n")
    
    if not predictor.extract_all_features(max_cuts=50):  # Change to None for all cuts
        print("Failed to extract features. Exiting.")
        return
    
    # Save features
    predictor.save_features('phm_features_sample.csv')
    
    # Train models
    results = predictor.train_models(target='avg_wear')
    
    # Feature importance
    predictor.feature_importance_analysis()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - wear_progression_analysis.png")
    print("  - model_predictions.png")
    print("  - feature_importance.png")
    print("  - phm_features_sample.csv")


if __name__ == "__main__":
    main()