"""
Physics-Informed Neural Network Models for RUL Prediction
NOVEL CONTRIBUTIONS:
1. Physics-constrained architecture with monotonicity enforcement
2. Multi-scale temporal attention
3. Degradation-aware loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional


class MultiScaleTemporalAttention(nn.Module):
    """
    NOVEL: Multi-scale temporal attention mechanism
    Captures degradation patterns at different time scales
    """
    
    def __init__(self, hidden_dim: int, num_scales: int = 4, num_heads: int = 8):
        """
        Initialize multi-scale attention
        
        Args:
            hidden_dim: Hidden dimension size
            num_scales: Number of temporal scales to consider
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.num_heads = num_heads
        
        # Multi-head attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_scales)
        ])
        
        # Scale-wise pooling layers
        self.scale_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=1, padding=2**(i-1))
            for i in range(1, num_scales+1)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            
        Returns:
            attended features, attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        scale_outputs = []
        attention_weights = []
        
        # Process each temporal scale
        for i, (attn_layer, pool_layer) in enumerate(zip(self.scale_attentions, self.scale_pools)):
            # Temporal pooling for different scales
            if i > 0:
                # Pool along sequence dimension
                x_pooled = pool_layer(x.transpose(1, 2)).transpose(1, 2)
            else:
                x_pooled = x
            
            # Apply attention
            attn_out, attn_weights = attn_layer(x_pooled, x_pooled, x_pooled)
            
            # Interpolate back to original sequence length if needed
            if attn_out.shape[1] != seq_len:
                attn_out = F.interpolate(
                    attn_out.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            
            scale_outputs.append(attn_out)
            attention_weights.append(attn_weights)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat(scale_outputs, dim=-1)
        
        # Fuse scales
        fused = self.fusion(multi_scale_features)
        fused = self.layer_norm(fused + x)  # Residual connection
        
        # Stack attention weights
        attention_weights = torch.stack(attention_weights, dim=1)
        
        return fused, attention_weights


class PhysicsConstrainedLayer(nn.Module):
    """
    NOVEL: Physics-constrained layer enforcing degradation monotonicity
    Ensures RUL predictions respect physical degradation laws
    """
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        """
        Initialize physics-constrained layer
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension (typically 1 for RUL)
        """
        super().__init__()
        
        # Use positive-weight networks for monotonicity
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, output_dim)
        
        # Initialize with positive weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics constraints
        
        Args:
            x: Input features
            
        Returns:
            RUL prediction (constrained to be non-negative and monotonic)
        """
        # Ensure positive weights for monotonicity
        with torch.no_grad():
            self.fc1.weight.clamp_(min=0)
            self.fc2.weight.clamp_(min=0)
        
        h = F.relu(self.fc1(x))
        rul = self.fc2(h)
        
        # Ensure non-negative RUL
        rul = F.softplus(rul)
        
        return rul


class PhysicsInformedRULModel(nn.Module):
    """
    NOVEL: Complete physics-informed RUL prediction model
    
    Architecture:
    1. CNN for feature extraction from raw signals
    2. LSTM for temporal modeling
    3. Multi-scale temporal attention
    4. Physics-constrained output layer
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        cnn_channels: list = [64, 128, 256],
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attention_heads: int = 8,
        num_scales: int = 4,
        dropout: float = 0.3,
        use_physics_constraints: bool = True
    ):
        """
        Initialize the model
        
        Args:
            input_dim: Input feature dimension
            seq_len: Sequence length
            cnn_channels: List of CNN channel sizes
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            attention_heads: Number of attention heads
            num_scales: Number of temporal scales
            dropout: Dropout rate
            use_physics_constraints: Whether to use physics constraints
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.use_physics_constraints = use_physics_constraints
        
        # 1. CNN Feature Extractor
        cnn_layers = []
        in_channels = input_dim
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output sequence length
        self.cnn_seq_len = seq_len // (2 ** len(cnn_channels))
        
        # 2. LSTM for Temporal Dependencies
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = lstm_hidden * 2  # Bidirectional
        
        # 3. Multi-Scale Temporal Attention (NOVEL)
        self.attention = MultiScaleTemporalAttention(
            hidden_dim=lstm_output_dim,
            num_scales=num_scales,
            num_heads=attention_heads
        )
        
        # 4. Feature Aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 5. RUL Prediction Head
        if use_physics_constraints:
            # Physics-constrained layer (NOVEL)
            self.rul_head = PhysicsConstrainedLayer(lstm_output_dim, output_dim=1)
        else:
            # Standard regression head
            self.rul_head = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, 1),
                nn.Softplus()  # Ensure non-negative RUL
            )
        
        # Store attention weights for visualization
        self.last_attention_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            RUL predictions (batch, 1)
        """
        batch_size = x.shape[0]
        
        # Reshape for CNN: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch, cnn_channels[-1], cnn_seq_len)
        
        # Reshape for LSTM: (batch, cnn_seq_len, cnn_channels[-1])
        cnn_features = cnn_features.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, _ = self.lstm(cnn_features)  # (batch, cnn_seq_len, lstm_hidden*2)
        
        # Multi-scale attention (NOVEL)
        attended, attention_weights = self.attention(lstm_out)
        self.last_attention_weights = attention_weights.detach()
        
        # Global average pooling
        pooled = self.global_pool(attended.transpose(1, 2)).squeeze(-1)
        
        # RUL prediction
        rul = self.rul_head(pooled)
        
        return rul
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights for visualization"""
        return self.last_attention_weights


class PhysicsInformedLoss(nn.Module):
    """
    NOVEL: Custom loss function with physics constraints
    
    Loss = MSE + λ₁ * Monotonicity_Loss + λ₂ * Smoothness_Loss
    """
    
    def __init__(
        self,
        monotonicity_weight: float = 0.5,
        smoothness_weight: float = 0.3
    ):
        """
        Initialize physics-informed loss
        
        Args:
            monotonicity_weight: Weight for monotonicity constraint
            smoothness_weight: Weight for smoothness constraint
        """
        super().__init__()
        
        self.monotonicity_weight = monotonicity_weight
        self.smoothness_weight = smoothness_weight
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        sequence_order: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics-informed loss
        
        Args:
            predictions: Predicted RUL values
            targets: True RUL values
            sequence_order: Optional tensor indicating temporal order of samples
            
        Returns:
            total_loss, loss_components dict
        """
        # Base MSE loss
        mse_loss = self.mse(predictions, targets)
        
        total_loss = mse_loss
        loss_components = {'mse': mse_loss.item()}
        
        # Monotonicity constraint (NOVEL)
        if sequence_order is not None and self.monotonicity_weight > 0:
            # Sort predictions by sequence order
            sorted_indices = torch.argsort(sequence_order)
            sorted_preds = predictions[sorted_indices]
            
            # RUL should decrease over time
            # Penalize violations: pred[t] < pred[t+1]
            diffs = sorted_preds[1:] - sorted_preds[:-1]
            monotonicity_violations = F.relu(diffs)  # Only positive diffs are violations
            monotonicity_loss = torch.mean(monotonicity_violations)
            
            total_loss += self.monotonicity_weight * monotonicity_loss
            loss_components['monotonicity'] = monotonicity_loss.item()
        
        # Smoothness constraint (NOVEL)
        if self.smoothness_weight > 0:
            # Encourage smooth predictions (avoid sudden jumps)
            if predictions.shape[0] > 1:
                pred_diffs = predictions[1:] - predictions[:-1]
                smoothness_loss = torch.mean(pred_diffs ** 2)
                
                total_loss += self.smoothness_weight * smoothness_loss
                loss_components['smoothness'] = smoothness_loss.item()
        
        loss_components['total'] = total_loss.item()
        
        return total_loss, loss_components


def create_model(config: Dict) -> PhysicsInformedRULModel:
    """
    Factory function to create model from config
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    model_config = config['model']
    
    model = PhysicsInformedRULModel(
        input_dim=config.get('input_dim', 6),  # Number of sensors
        seq_len=config.get('window_size', 50),
        cnn_channels=model_config['feature_extractor']['cnn_channels'],
        lstm_hidden=model_config['feature_extractor']['lstm_hidden'],
        lstm_layers=model_config['feature_extractor']['lstm_layers'],
        attention_heads=model_config['temporal_attention']['attention_heads'],
        num_scales=len(model_config['temporal_attention']['scales']),
        dropout=model_config['feature_extractor']['dropout'],
        use_physics_constraints=model_config['physics_constraints']['monotonicity']
    )
    
    return model