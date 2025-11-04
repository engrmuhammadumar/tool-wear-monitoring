"""
Data Loader for PHM 2010 Milling Dataset
Handles loading, preprocessing, and organizing cutter data
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PHMDataLoader:
    """
    Data loader for PHM Society 2010 Milling Dataset
    
    Dataset Structure:
    - c1-c6: Different cutting conditions/experiments
    - Each folder contains:
        - Multiple CSV files with sensor measurements
        - Wear measurements file
    """
    
    def __init__(self, root_path: str, config: Dict = None):
        """
        Initialize data loader
        
        Args:
            root_path: Path to PHM dataset root directory
            config: Configuration dictionary
        """
        self.root_path = Path(root_path)
        self.config = config or {}
        
        # Expected sensors in PHM 2010
        self.sensor_columns = [
            'smcAC', 'smcDC', 'vib_table', 'vib_spindle', 
            'AE_table', 'AE_spindle'
        ]
        
        # Storage for loaded data
        self.data_cache = {}
        self.wear_data = {}
        
    def load_cutter_data(self, cutter_id: str, verbose: bool = True) -> Dict:
        """
        Load all data for a specific cutter
        
        Args:
            cutter_id: Cutter identifier (e.g., 'c1', 'c2')
            verbose: Print loading information
            
        Returns:
            Dictionary containing sensor data and wear measurements
        """
        if cutter_id in self.data_cache:
            if verbose:
                print(f"✓ Loading {cutter_id} from cache")
            return self.data_cache[cutter_id]
        
        if verbose:
            print(f"📂 Loading data for cutter: {cutter_id}")
        
        cutter_path = self.root_path / cutter_id
        
        if not cutter_path.exists():
            raise FileNotFoundError(f"Cutter directory not found: {cutter_path}")
        
        # Load sensor data files
        sensor_files = sorted(list(cutter_path.glob("*.csv")))
        
        # Check for Excel files (wear measurements usually in Excel)
        wear_files = list(cutter_path.glob("*wear*.xlsx")) + list(cutter_path.glob("*wear*.xls"))
        
        if len(sensor_files) == 0:
            # Try .txt files
            sensor_files = sorted(list(cutter_path.glob("*.txt")))
        
        if verbose:
            print(f"  Found {len(sensor_files)} sensor data files")
            print(f"  Found {len(wear_files)} wear measurement files")
        
        # Load all sensor data
        sensor_data_list = []
        cut_numbers = []
        
        for i, file_path in enumerate(sensor_files):
            try:
                # Try reading as CSV
                df = pd.read_csv(file_path)
                
                # If no headers, assign column names
                if df.columns[0] == 0 or 'Unnamed' in str(df.columns[0]):
                    if len(df.columns) == 6:
                        df.columns = self.sensor_columns
                    elif len(df.columns) == 7:  # Some files have time column
                        df.columns = ['time'] + self.sensor_columns
                
                sensor_data_list.append(df)
                cut_numbers.append(i + 1)
                
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Warning: Could not load {file_path.name}: {e}")
                continue
        
        # Load wear measurements
        wear_measurements = None
        if wear_files:
            try:
                wear_measurements = pd.read_excel(wear_files[0])
                if verbose:
                    print(f"  ✓ Loaded wear measurements: {wear_measurements.shape}")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Could not load wear file: {e}")
        
        # Package data
        cutter_data = {
            'cutter_id': cutter_id,
            'sensor_data': sensor_data_list,
            'cut_numbers': cut_numbers,
            'wear_measurements': wear_measurements,
            'num_cuts': len(sensor_data_list),
            'sensor_columns': self.sensor_columns
        }
        
        # Cache the data
        self.data_cache[cutter_id] = cutter_data
        
        if verbose:
            print(f"  ✓ Successfully loaded {len(sensor_data_list)} cuts for {cutter_id}\n")
        
        return cutter_data
    
    def load_all_cutters(self, cutter_list: List[str]) -> Dict:
        """
        Load data for multiple cutters
        
        Args:
            cutter_list: List of cutter IDs to load
            
        Returns:
            Dictionary mapping cutter_id to cutter data
        """
        all_data = {}
        
        print("=" * 60)
        print("🔄 LOADING PHM 2010 DATASET")
        print("=" * 60)
        
        for cutter_id in cutter_list:
            try:
                data = self.load_cutter_data(cutter_id)
                all_data[cutter_id] = data
            except Exception as e:
                print(f"❌ Error loading {cutter_id}: {e}\n")
                continue
        
        print("=" * 60)
        print(f"✓ Successfully loaded {len(all_data)}/{len(cutter_list)} cutters")
        print("=" * 60)
        
        return all_data
    
    def prepare_sequences(
        self, 
        cutter_data: Dict, 
        window_size: int = 50,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sliding window sequences for RUL prediction
        
        Args:
            cutter_data: Data dictionary from load_cutter_data
            window_size: Size of sliding window
            stride: Stride for sliding window
            
        Returns:
            X: Input sequences (samples, window_size, features)
            y: RUL labels (samples,)
        """
        sensor_data_list = cutter_data['sensor_data']
        num_cuts = len(sensor_data_list)
        
        sequences = []
        labels = []
        
        for cut_idx in range(num_cuts):
            # Get sensor data for this cut
            sensor_df = sensor_data_list[cut_idx]
            
            # Calculate RUL (remaining useful life)
            # RUL = total_cuts - current_cut
            rul = num_cuts - cut_idx - 1
            
            # Create sliding windows within this cut
            num_windows = (len(sensor_df) - window_size) // stride + 1
            
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + window_size
                
                window = sensor_df.iloc[start_idx:end_idx][self.sensor_columns].values
                
                sequences.append(window)
                labels.append(rul)
        
        X = np.array(sequences)
        y = np.array(labels)
        
        return X, y
    
    def get_train_test_split(
        self,
        train_cutters: List[str],
        test_cutters: List[str],
        window_size: int = 50,
        stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train-test split for cross-condition validation
        
        Args:
            train_cutters: List of cutter IDs for training
            test_cutters: List of cutter IDs for testing
            window_size: Window size for sequences
            stride: Stride for sliding window
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        print("\n" + "=" * 60)
        print("🔀 PREPARING TRAIN-TEST SPLIT")
        print("=" * 60)
        
        # Load training data
        print(f"Training cutters: {train_cutters}")
        train_data = self.load_all_cutters(train_cutters)
        
        # Load test data
        print(f"\nTest cutters: {test_cutters}")
        test_data = self.load_all_cutters(test_cutters)
        
        # Prepare training sequences
        X_train_list, y_train_list = [], []
        for cutter_id in train_cutters:
            if cutter_id in train_data:
                X, y = self.prepare_sequences(
                    train_data[cutter_id], 
                    window_size, 
                    stride
                )
                X_train_list.append(X)
                y_train_list.append(y)
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        # Prepare test sequences
        X_test_list, y_test_list = [], []
        for cutter_id in test_cutters:
            if cutter_id in test_data:
                X, y = self.prepare_sequences(
                    test_data[cutter_id], 
                    window_size, 
                    stride
                )
                X_test_list.append(X)
                y_test_list.append(y)
        
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        
        print("\n" + "=" * 60)
        print("✓ DATA SPLIT SUMMARY")
        print("=" * 60)
        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Test samples: {X_test.shape[0]:,}")
        print(f"Input shape: {X_train.shape[1:]}")
        print(f"RUL range (train): [{y_train.min()}, {y_train.max()}]")
        print(f"RUL range (test): [{y_test.min()}, {y_test.max()}]")
        print("=" * 60 + "\n")
        
        return X_train, y_train, X_test, y_test
    
    def get_cutter_info(self) -> pd.DataFrame:
        """
        Get summary information about all cutters
        
        Returns:
            DataFrame with cutter information
        """
        if not self.data_cache:
            print("No data loaded yet. Load cutters first.")
            return None
        
        info_list = []
        for cutter_id, data in self.data_cache.items():
            info_list.append({
                'Cutter ID': cutter_id,
                'Number of Cuts': data['num_cuts'],
                'Sensors': len(data['sensor_columns']),
                'Has Wear Data': data['wear_measurements'] is not None
            })
        
        return pd.DataFrame(info_list)


def load_phm_dataset(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load PHM dataset based on config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    loader = PHMDataLoader(
        root_path=config['data']['root_path'],
        config=config
    )
    
    return loader.get_train_test_split(
        train_cutters=config['data']['train_cutters'],
        test_cutters=config['data']['test_cutters'],
        window_size=config.get('window_size', 50),
        stride=config.get('stride', 1)
    )