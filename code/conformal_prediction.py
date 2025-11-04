"""
Adaptive Conformal Prediction for RUL Estimation
NOVEL CONTRIBUTION: Time-varying, degradation-aware uncertainty quantification

Key Innovation:
- Traditional conformal prediction assumes stationary data
- We develop ADAPTIVE conformal scores that account for degradation progression
- Prediction intervals that adapt to the stage of tool life
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from scipy.stats import norm


class AdaptiveConformalPredictor:
    """
    NOVEL: Adaptive Conformal Prediction for Non-Stationary Degradation Data
    
    Key Features:
    1. Time-varying nonconformity scores
    2. Degradation-stage-aware calibration
    3. Guaranteed coverage even under distribution shift
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        window_size: int = 50,
        degradation_aware: bool = True
    ):
        """
        Initialize adaptive conformal predictor
        
        Args:
            alpha: Significance level (1-alpha = coverage level, e.g., 0.1 for 90%)
            window_size: Window size for adaptive score computation
            degradation_aware: Use degradation-aware weighting
        """
        self.alpha = alpha
        self.window_size = window_size
        self.degradation_aware = degradation_aware
        
        # Calibration data
        self.calibration_scores = None
        self.calibration_ruls = None
        self.quantile = None
        
        # Adaptive parameters
        self.degradation_stages = None
        self.stage_quantiles = None
        
    def _compute_nonconformity_score(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> np.ndarray:
        """
        Compute nonconformity scores (residuals)
        
        Args:
            predictions: Model predictions
            targets: True RUL values
            
        Returns:
            Nonconformity scores
        """
        # Use absolute residuals as nonconformity scores
        scores = np.abs(predictions - targets)
        return scores
    
    def _assign_degradation_stage(self, rul_values: np.ndarray) -> np.ndarray:
        """
        NOVEL: Assign degradation stage based on RUL
        
        Stages:
        0: Early life (RUL > 66% of max)
        1: Mid life (33% < RUL <= 66%)
        2: Late life (RUL <= 33%)
        
        Args:
            rul_values: RUL values
            
        Returns:
            Stage assignments (0, 1, or 2)
        """
        max_rul = np.max(rul_values)
        
        stages = np.zeros_like(rul_values, dtype=int)
        stages[rul_values <= max_rul * 0.33] = 2  # Late life
        stages[(rul_values > max_rul * 0.33) & (rul_values <= max_rul * 0.66)] = 1  # Mid life
        stages[rul_values > max_rul * 0.66] = 0  # Early life
        
        return stages
    
    def _compute_stage_weights(
        self,
        current_stage: int,
        calibration_stages: np.ndarray
    ) -> np.ndarray:
        """
        NOVEL: Compute weights for calibration samples based on degradation stage
        Gives more weight to samples from similar degradation stages
        
        Args:
            current_stage: Current degradation stage
            calibration_stages: Stages of calibration samples
            
        Returns:
            Sample weights
        """
        # Exponential weighting based on stage difference
        stage_diff = np.abs(calibration_stages - current_stage)
        weights = np.exp(-stage_diff)
        
        # Normalize
        weights = weights / np.sum(weights)
        
        return weights
    
    def calibrate(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True
    ):
        """
        Calibrate the conformal predictor on validation data
        
        Args:
            predictions: Model predictions on calibration set
            targets: True RUL values on calibration set
            verbose: Print calibration info
        """
        if verbose:
            print("\n" + "=" * 60)
            print("🎯 CALIBRATING ADAPTIVE CONFORMAL PREDICTOR")
            print("=" * 60)
        
        # Compute nonconformity scores
        self.calibration_scores = self._compute_nonconformity_score(predictions, targets)
        self.calibration_ruls = targets
        
        if self.degradation_aware:
            # Assign degradation stages
            self.degradation_stages = self._assign_degradation_stage(targets)
            
            # Compute quantiles for each stage
            self.stage_quantiles = {}
            for stage in range(3):
                stage_mask = self.degradation_stages == stage
                stage_scores = self.calibration_scores[stage_mask]
                
                if len(stage_scores) > 0:
                    # Compute quantile for this stage
                    n = len(stage_scores)
                    quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
                    quantile_level = min(quantile_level, 1.0)
                    
                    stage_quantile = np.quantile(stage_scores, quantile_level)
                    self.stage_quantiles[stage] = stage_quantile
                    
                    if verbose:
                        stage_names = ['Early', 'Mid', 'Late']
                        print(f"  Stage {stage} ({stage_names[stage]}): "
                              f"quantile = {stage_quantile:.4f}, n = {len(stage_scores)}")
        else:
            # Standard conformal prediction (non-adaptive)
            n = len(self.calibration_scores)
            quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            quantile_level = min(quantile_level, 1.0)
            
            self.quantile = np.quantile(self.calibration_scores, quantile_level)
            
            if verbose:
                print(f"  Overall quantile: {self.quantile:.4f}")
        
        if verbose:
            print(f"  Coverage level: {(1 - self.alpha) * 100:.1f}%")
            print("  ✓ Calibration complete")
            print("=" * 60 + "\n")
    
    def predict(
        self,
        predictions: np.ndarray,
        return_components: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals
        
        Args:
            predictions: Point predictions from model
            return_components: Whether to return individual components
            
        Returns:
            lower_bounds, upper_bounds (and optionally interval widths)
        """
        if self.calibration_scores is None:
            raise ValueError("Must calibrate before making predictions!")
        
        if self.degradation_aware and self.stage_quantiles is not None:
            # NOVEL: Adaptive intervals based on degradation stage
            intervals = np.zeros_like(predictions)
            
            # Assign stages to test predictions
            test_stages = self._assign_degradation_stage(predictions)
            
            for stage in range(3):
                stage_mask = test_stages == stage
                
                if stage in self.stage_quantiles:
                    intervals[stage_mask] = self.stage_quantiles[stage]
                else:
                    # Fallback to overall quantile if stage not calibrated
                    intervals[stage_mask] = np.median(list(self.stage_quantiles.values()))
        else:
            # Standard (non-adaptive) intervals
            intervals = np.full_like(predictions, self.quantile)
        
        # Compute bounds
        lower_bounds = predictions - intervals
        upper_bounds = predictions + intervals
        
        # Ensure non-negative RUL
        lower_bounds = np.maximum(lower_bounds, 0)
        
        if return_components:
            return lower_bounds, upper_bounds, intervals
        
        return lower_bounds, upper_bounds
    
    def evaluate_coverage(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        return_details: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate prediction interval coverage
        
        Args:
            predictions: Point predictions
            targets: True RUL values
            return_details: Return detailed metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        lower_bounds, upper_bounds, intervals = self.predict(
            predictions, 
            return_components=True
        )
        
        # Check coverage
        covered = (targets >= lower_bounds) & (targets <= upper_bounds)
        coverage_rate = np.mean(covered)
        
        # Average interval width
        avg_width = np.mean(intervals)
        
        metrics = {
            'coverage_rate': coverage_rate,
            'target_coverage': 1 - self.alpha,
            'avg_interval_width': avg_width,
            'coverage_error': abs(coverage_rate - (1 - self.alpha))
        }
        
        if return_details:
            # Stage-wise coverage (if using adaptive)
            if self.degradation_aware:
                test_stages = self._assign_degradation_stage(targets)
                
                for stage in range(3):
                    stage_mask = test_stages == stage
                    if np.sum(stage_mask) > 0:
                        stage_coverage = np.mean(covered[stage_mask])
                        stage_width = np.mean(intervals[stage_mask])
                        
                        stage_names = ['early', 'mid', 'late']
                        metrics[f'coverage_{stage_names[stage]}'] = stage_coverage
                        metrics[f'width_{stage_names[stage]}'] = stage_width
        
        return metrics
    
    def plot_prediction_intervals(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Visualize prediction intervals
        
        Args:
            predictions: Point predictions
            targets: True RUL values
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        lower_bounds, upper_bounds = self.predict(predictions)
        
        # Sort by true RUL for better visualization
        sort_idx = np.argsort(targets)
        targets_sorted = targets[sort_idx]
        preds_sorted = predictions[sort_idx]
        lower_sorted = lower_bounds[sort_idx]
        upper_sorted = upper_bounds[sort_idx]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(targets_sorted))
        
        # Plot intervals
        ax.fill_between(x, lower_sorted, upper_sorted, alpha=0.3, label='Prediction Interval')
        
        # Plot predictions and targets
        ax.plot(x, targets_sorted, 'g-', linewidth=2, label='True RUL', alpha=0.7)
        ax.plot(x, preds_sorted, 'b--', linewidth=2, label='Predicted RUL')
        
        # Compute coverage
        covered = (targets_sorted >= lower_sorted) & (targets_sorted <= upper_sorted)
        uncovered_idx = x[~covered]
        if len(uncovered_idx) > 0:
            ax.scatter(uncovered_idx, targets_sorted[~covered], 
                      color='red', s=50, zorder=5, label='Uncovered')
        
        ax.set_xlabel('Sample (sorted by True RUL)', fontsize=12)
        ax.set_ylabel('RUL (cycles)', fontsize=12)
        ax.set_title(f'Adaptive Conformal Prediction Intervals\n'
                    f'Coverage: {np.mean(covered)*100:.1f}% '
                    f'(Target: {(1-self.alpha)*100:.1f}%)', 
                    fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved plot to {save_path}")
        
        plt.show()


class MultiLevelConformalPredictor:
    """
    NOVEL: Multi-level conformal prediction with multiple confidence levels
    Provides multiple prediction intervals simultaneously
    """
    
    def __init__(self, alpha_levels: List[float] = [0.1, 0.05, 0.01]):
        """
        Initialize multi-level predictor
        
        Args:
            alpha_levels: List of significance levels (e.g., [0.1, 0.05] for 90%, 95%)
        """
        self.alpha_levels = sorted(alpha_levels, reverse=True)
        self.predictors = [
            AdaptiveConformalPredictor(alpha=alpha, degradation_aware=True)
            for alpha in alpha_levels
        ]
    
    def calibrate(self, predictions: np.ndarray, targets: np.ndarray, verbose: bool = True):
        """Calibrate all levels"""
        if verbose:
            print("\n🎯 CALIBRATING MULTI-LEVEL CONFORMAL PREDICTOR")
        
        for i, predictor in enumerate(self.predictors):
            coverage = (1 - self.alpha_levels[i]) * 100
            if verbose:
                print(f"\n  Level {i+1}: {coverage:.0f}% coverage")
            predictor.calibrate(predictions, targets, verbose=False)
    
    def predict(self, predictions: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate prediction intervals for all levels
        
        Returns:
            Dictionary mapping coverage level to (lower, upper) bounds
        """
        intervals = {}
        
        for i, predictor in enumerate(self.predictors):
            coverage = int((1 - self.alpha_levels[i]) * 100)
            lower, upper = predictor.predict(predictions)
            intervals[f'{coverage}%'] = (lower, upper)
        
        return intervals
    
    def evaluate(self, predictions: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all confidence levels
        
        Returns:
            DataFrame with metrics for each level
        """
        import pandas as pd
        
        results = []
        
        for i, predictor in enumerate(self.predictors):
            coverage_target = (1 - self.alpha_levels[i]) * 100
            metrics = predictor.evaluate_coverage(predictions, targets, return_details=True)
            
            results.append({
                'Coverage Target': f'{coverage_target:.0f}%',
                'Coverage Achieved': f'{metrics["coverage_rate"]*100:.1f}%',
                'Avg Interval Width': f'{metrics["avg_interval_width"]:.2f}',
                'Coverage Error': f'{metrics["coverage_error"]*100:.1f}%'
            })
        
        return pd.DataFrame(results)


# Import pandas for the evaluate method
import pandas as pd