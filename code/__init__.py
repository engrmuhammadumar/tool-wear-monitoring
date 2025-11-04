"""
Physics-Informed Adaptive Conformal Prediction for Tool RUL Estimation
PHM 2010 Dataset Implementation

Novel Contributions:
1. Physics-informed feature extraction with degradation constraints
2. Adaptive conformal prediction with time-varying uncertainty
3. Multi-scale temporal attention mechanism
4. Cross-condition generalization framework

Authors: [Your Names]
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import PHMDataLoader
from .feature_engineering import FeatureExtractor, PhysicsInformedFeatures
from .models import PhysicsInformedRULModel
from .conformal_prediction import AdaptiveConformalPredictor
from .utils import set_seed, load_config

__all__ = [
    'PHMDataLoader',
    'FeatureExtractor',
    'PhysicsInformedFeatures',
    'PhysicsInformedRULModel',
    'AdaptiveConformalPredictor',
    'set_seed',
    'load_config'
]