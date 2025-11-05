"""
Utility functions for PHM RUL prediction project
"""

import os
import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} for reproducibility")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✓ Loaded configuration from {config_path}")
    return config


def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✓ Saved configuration to {save_path}")


def create_directories(base_path: str):
    """
    Create project directory structure
    
    Args:
        base_path: Base project path
    """
    directories = [
        'results',
        'figures',
        'checkpoints',
        'logs'
    ]
    
    for dir_name in directories:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✓ Created project directories")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get compute device (GPU or CPU)
    
    Args:
        prefer_gpu: Prefer GPU if available
        
    Returns:
        torch.device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU")
    
    return device


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if should stop training
        
        Args:
            val_loss: Validation loss
            model: Model to save if improved
            
        Returns:
            Whether to stop training
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('  Early stopping triggered!')
        else:
            if self.verbose:
                print(f'  Validation loss improved: {self.best_loss:.4f} → {val_loss:.4f}')
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        
        return self.early_stop
    
    def load_best_model(self, model: torch.nn.Module):
        """Load best model state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def compute_phm_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute PHM Challenge scoring function
    
    Score penalizes late predictions more heavily than early ones
    
    Args:
        predictions: Predicted RUL
        targets: True RUL
        
    Returns:
        PHM score (lower is better)
    """
    errors = predictions - targets
    
    # Asymmetric scoring
    score = 0
    for error in errors:
        if error < 0:  # Late prediction (more dangerous)
            score += np.exp(-error / 13) - 1
        else:  # Early prediction
            score += np.exp(error / 10) - 1
    
    return score / len(errors)


def plot_training_history(
    train_losses: list,
    val_losses: list,
    save_path: str = None
):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
    
    plt.show()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    title: str = "RUL Predictions",
    save_path: str = None
):
    """
    Plot predicted vs actual RUL
    
    Args:
        predictions: Predicted RUL values
        targets: True RUL values
        title: Plot title
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True RUL', fontsize=12)
    ax1.set_ylabel('Predicted RUL', fontsize=12)
    ax1.set_title('Predictions vs Actual', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = predictions - targets
    ax2.scatter(targets, residuals, alpha=0.5, s=20, c='orange')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('True RUL', fontsize=12)
    ax2.set_ylabel('Residual (Predicted - True)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved predictions plot to {save_path}")
    
    plt.show()


def plot_attention_weights(
    attention_weights: np.ndarray,
    save_path: str = None
):
    """
    Visualize multi-scale attention weights
    
    Args:
        attention_weights: Attention weights array (scales, seq_len, seq_len)
        save_path: Path to save figure
    """
    num_scales = attention_weights.shape[0]
    
    fig, axes = plt.subplots(1, num_scales, figsize=(5*num_scales, 4))
    
    if num_scales == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        sns.heatmap(
            attention_weights[i],
            cmap='YlOrRd',
            ax=ax,
            cbar=True,
            square=True
        )
        ax.set_title(f'Scale {i+1}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Key Position', fontsize=10)
        ax.set_ylabel('Query Position', fontsize=10)
    
    plt.suptitle('Multi-Scale Temporal Attention Weights', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved attention weights to {save_path}")
    
    plt.show()


def save_results(results: Dict, save_path: str):
    """
    Save results dictionary to file
    
    Args:
        results: Results dictionary
        save_path: Path to save
    """
    import json
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"✓ Saved results to {save_path}")


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("\n" + "=" * 60)
    print(f"📊 {title}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:.<40} {value:.4f}")
        else:
            print(f"  {key:.<40} {value}")
    
    print("=" * 60 + "\n")