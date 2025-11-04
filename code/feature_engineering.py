"""
Physics-Informed Feature Engineering for Tool Wear Prediction
Novel Contributions: Physics-based features respecting degradation mechanisms
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """
    Comprehensive feature extraction from raw sensor signals
    Includes time-domain, frequency-domain, and wavelet features
    """
    
    def __init__(self, sensor_columns: List[str]):
        """
        Initialize feature extractor
        
        Args:
            sensor_columns: List of sensor channel names
        """
        self.sensor_columns = sensor_columns
        self.scaler = StandardScaler()
        
    def extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain statistical features
        
        Args:
            signal_data: 1D signal array
            
        Returns:
            Dictionary of time-domain features
        """
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(signal_data)
        features['std'] = np.std(signal_data)
        features['var'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak_to_peak'] = np.ptp(signal_data)
        
        # Shape factors
        features['kurtosis'] = stats.kurtosis(signal_data)
        features['skewness'] = stats.skew(signal_data)
        
        # Advanced shape factors
        abs_signal = np.abs(signal_data)
        features['crest_factor'] = np.max(abs_signal) / features['rms'] if features['rms'] > 0 else 0
        features['shape_factor'] = features['rms'] / np.mean(abs_signal) if np.mean(abs_signal) > 0 else 0
        features['impulse_factor'] = np.max(abs_signal) / np.mean(abs_signal) if np.mean(abs_signal) > 0 else 0
        
        # Percentiles
        features['p10'] = np.percentile(signal_data, 10)
        features['p25'] = np.percentile(signal_data, 25)
        features['p50'] = np.percentile(signal_data, 50)
        features['p75'] = np.percentile(signal_data, 75)
        features['p90'] = np.percentile(signal_data, 90)
        
        return features
    
    def extract_frequency_domain_features(
        self, 
        signal_data: np.ndarray, 
        sampling_rate: float = 50000
    ) -> Dict[str, float]:
        """
        Extract frequency-domain features using FFT
        
        Args:
            signal_data: 1D signal array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary of frequency-domain features
        """
        features = {}
        
        # Compute FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_freq = fftfreq(n, 1/sampling_rate)
        
        # Use only positive frequencies
        pos_mask = fft_freq > 0
        fft_freq = fft_freq[pos_mask]
        fft_mag = np.abs(fft_vals[pos_mask])
        
        # Power spectral density
        psd = fft_mag ** 2
        
        # Frequency domain statistics
        features['fft_mean'] = np.mean(fft_mag)
        features['fft_std'] = np.std(fft_mag)
        features['fft_max'] = np.max(fft_mag)
        
        # Dominant frequency
        dominant_idx = np.argmax(psd)
        features['dominant_frequency'] = fft_freq[dominant_idx]
        features['dominant_magnitude'] = fft_mag[dominant_idx]
        
        # Spectral centroid (center of mass of spectrum)
        features['spectral_centroid'] = np.sum(fft_freq * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
        
        # Spectral entropy
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
        features['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))
        
        # Frequency band energies (for tool wear detection)
        # Low: 0-1kHz, Mid: 1k-5kHz, High: 5k-15kHz, Very High: 15k+
        freq_bands = {
            'low': (0, 1000),
            'mid': (1000, 5000),
            'high': (5000, 15000),
            'very_high': (15000, sampling_rate/2)
        }
        
        for band_name, (f_low, f_high) in freq_bands.items():
            band_mask = (fft_freq >= f_low) & (fft_freq < f_high)
            features[f'energy_{band_name}'] = np.sum(psd[band_mask])
        
        return features
    
    def extract_wavelet_features(
        self, 
        signal_data: np.ndarray,
        wavelet: str = 'db4',
        level: int = 5
    ) -> Dict[str, float]:
        """
        Extract wavelet-based features
        
        Args:
            signal_data: 1D signal array
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Dictionary of wavelet features
        """
        features = {}
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # Extract features from each level
        for i, coeff in enumerate(coeffs):
            prefix = f'wavelet_l{i}'
            features[f'{prefix}_energy'] = np.sum(coeff ** 2)
            features[f'{prefix}_std'] = np.std(coeff)
            features[f'{prefix}_mean_abs'] = np.mean(np.abs(coeff))
        
        return features
    
    def extract_all_features(
        self, 
        signal_window: np.ndarray,
        sampling_rate: float = 50000
    ) -> np.ndarray:
        """
        Extract all features from a multi-channel signal window
        
        Args:
            signal_window: (window_length, num_channels) array
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Feature vector
        """
        all_features = []
        
        for channel_idx in range(signal_window.shape[1]):
            channel_data = signal_window[:, channel_idx]
            
            # Time domain
            time_features = self.extract_time_domain_features(channel_data)
            all_features.extend(time_features.values())
            
            # Frequency domain
            freq_features = self.extract_frequency_domain_features(channel_data, sampling_rate)
            all_features.extend(freq_features.values())
            
            # Wavelet
            wavelet_features = self.extract_wavelet_features(channel_data)
            all_features.extend(wavelet_features.values())
        
        return np.array(all_features)


class PhysicsInformedFeatures:
    """
    NOVEL: Physics-informed features based on tool wear mechanisms
    
    Tool wear mechanisms:
    1. Abrasive wear - proportional to cutting force and distance
    2. Adhesive wear - related to temperature and contact pressure
    3. Diffusion wear - temperature-dependent
    4. Fatigue wear - cyclic loading patterns
    """
    
    def __init__(self, sensor_columns: List[str]):
        """
        Initialize physics-informed feature extractor
        
        Args:
            sensor_columns: List of sensor channel names
        """
        self.sensor_columns = sensor_columns
        
        # Map sensors to physical quantities
        self.force_indicators = ['smcAC', 'smcDC']  # Current correlates with force
        self.vibration_indicators = ['vib_table', 'vib_spindle']
        self.temperature_indicators = ['AE_table', 'AE_spindle']  # AE correlates with friction/temp
    
    def compute_cutting_force_ratio(self, signal_window: np.ndarray) -> float:
        """
        NOVEL: Cutting force ratio indicator
        Ratio of peak force to average force indicates tool condition
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Force ratio feature
        """
        force_channels = []
        for i, col in enumerate(self.sensor_columns):
            if col in self.force_indicators:
                force_channels.append(signal_window[:, i])
        
        if not force_channels:
            return 0.0
        
        # Combine force channels
        combined_force = np.mean(force_channels, axis=0)
        
        # Peak to average ratio (increases with wear)
        peak_force = np.max(np.abs(combined_force))
        avg_force = np.mean(np.abs(combined_force))
        
        force_ratio = peak_force / avg_force if avg_force > 0 else 0.0
        
        return force_ratio
    
    def compute_wear_rate_indicator(self, signal_window: np.ndarray) -> float:
        """
        NOVEL: Wear rate indicator based on signal energy trends
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Wear rate indicator
        """
        # Compute energy across all channels
        energy_per_sample = np.sum(signal_window ** 2, axis=1)
        
        # Fit linear trend to energy
        x = np.arange(len(energy_per_sample))
        slope, _, _, _, _ = stats.linregress(x, energy_per_sample)
        
        # Positive slope indicates increasing wear
        return slope
    
    def compute_temperature_proxy(self, signal_window: np.ndarray) -> float:
        """
        NOVEL: Temperature proxy from acoustic emission
        Higher AE energy indicates more friction and heat
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Temperature proxy feature
        """
        ae_channels = []
        for i, col in enumerate(self.sensor_columns):
            if col in self.temperature_indicators:
                ae_channels.append(signal_window[:, i])
        
        if not ae_channels:
            return 0.0
        
        # RMS of AE signals (proxy for friction/temperature)
        ae_combined = np.mean(ae_channels, axis=0)
        temperature_proxy = np.sqrt(np.mean(ae_combined ** 2))
        
        return temperature_proxy
    
    def compute_chatter_indicator(self, signal_window: np.ndarray) -> float:
        """
        NOVEL: Tool chatter indicator from vibration signals
        Chatter increases with tool wear
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Chatter indicator
        """
        vib_channels = []
        for i, col in enumerate(self.sensor_columns):
            if col in self.vibration_indicators:
                vib_channels.append(signal_window[:, i])
        
        if not vib_channels:
            return 0.0
        
        vib_combined = np.mean(vib_channels, axis=0)
        
        # Chatter frequency typically 200-2000 Hz
        # Use band-pass filter
        from scipy.signal import butter, filtfilt
        
        fs = 50000  # Sampling rate
        lowcut = 200
        highcut = 2000
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, vib_combined)
        
        # Chatter indicator: energy in chatter frequency band
        chatter_energy = np.sqrt(np.mean(filtered ** 2))
        
        return chatter_energy
    
    def compute_monotonicity_score(self, signal_window: np.ndarray) -> float:
        """
        NOVEL: Degradation monotonicity score
        Measures how consistently signal energy increases (physical wear constraint)
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Monotonicity score
        """
        # Total energy per time step
        energy = np.sum(signal_window ** 2, axis=1)
        
        # Count monotonic increases
        diffs = np.diff(energy)
        monotonic_increases = np.sum(diffs > 0)
        
        monotonicity = monotonic_increases / len(diffs) if len(diffs) > 0 else 0.5
        
        return monotonicity
    
    def extract_physics_features(self, signal_window: np.ndarray) -> np.ndarray:
        """
        Extract all physics-informed features
        
        Args:
            signal_window: (window_length, num_channels) array
            
        Returns:
            Physics-based feature vector
        """
        features = [
            self.compute_cutting_force_ratio(signal_window),
            self.compute_wear_rate_indicator(signal_window),
            self.compute_temperature_proxy(signal_window),
            self.compute_chatter_indicator(signal_window),
            self.compute_monotonicity_score(signal_window)
        ]
        
        return np.array(features)


class HybridFeatureExtractor:
    """
    Combines traditional and physics-informed features
    """
    
    def __init__(self, sensor_columns: List[str]):
        """
        Initialize hybrid feature extractor
        
        Args:
            sensor_columns: List of sensor channel names
        """
        self.traditional = FeatureExtractor(sensor_columns)
        self.physics = PhysicsInformedFeatures(sensor_columns)
        self.sensor_columns = sensor_columns
    
    def extract_features(
        self, 
        signal_window: np.ndarray,
        use_physics: bool = True
    ) -> np.ndarray:
        """
        Extract hybrid features
        
        Args:
            signal_window: (window_length, num_channels) array
            use_physics: Whether to include physics-informed features
            
        Returns:
            Complete feature vector
        """
        # Traditional features
        traditional_features = self.traditional.extract_all_features(signal_window)
        
        if use_physics:
            # Physics-informed features
            physics_features = self.physics.extract_physics_features(signal_window)
            
            # Concatenate
            all_features = np.concatenate([traditional_features, physics_features])
        else:
            all_features = traditional_features
        
        return all_features
    
    def fit_scaler(self, X: np.ndarray):
        """Fit feature scaler on training data"""
        self.traditional.scaler.fit(X)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        return self.traditional.scaler.transform(X)