"""
Trajectory Prediction Neural Networks
====================================

Neural networks for predicting and generating handwriting trajectories
with temporal consistency and style conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging
from collections import deque

from .base_model import BaseNeuralNetwork

logger = logging.getLogger(__name__)


class TrajectoryPredictor(BaseNeuralNetwork):
    """
    Neural network for predicting handwriting trajectories.
    
    Uses LSTM/GRU to model temporal dependencies in handwriting motions
    and incorporates style conditioning.
    """
    
    def __init__(self, config: Dict[str, Any], input_dim: int, output_dim: int, style_dim: int = 0):
        """
        Initialize trajectory predictor.
        
        Args:
            config: Model configuration
            input_dim: Input dimension per timestep
            output_dim: Output dimension per timestep  
            style_dim: Style conditioning dimension
        """
        super().__init__(config, "TrajectoryPredictor")
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.style_dim = style_dim
        
        # RNN configuration
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 2)
        self.rnn_type = config.get('rnn_type', 'lstm')
        self.dropout = config.get('dropout', 0.1)
        self.bidirectional = config.get('bidirectional', False)
        
        # Sequence length
        self.max_seq_length = config.get('max_seq_length', 100)
        
        # Input projection (if style conditioning is used)
        if style_dim > 0:
            self.input_projection = nn.Linear(input_dim + style_dim, input_dim)
        else:
            self.input_projection = None
        
        # RNN layers
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                bidirectional=self.bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
        
        # Output projection
        rnn_output_dim = self.hidden_dim * (2 if self.bidirectional else 1)
        self.output_projection = nn.Sequential(
            nn.Linear(rnn_output_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
        # Attention mechanism (optional)
        self.use_attention = config.get('use_attention', False)
        if self.use_attention:
            self.attention = AttentionMechanism(rnn_output_dim, config.get('attention_dim', 64))
        
        self.to_device()
        
        logger.info(f"Initialized TrajectoryPredictor: {self.rnn_type.upper()} with {self.num_layers} layers")
    
    def forward(self, 
               sequences: torch.Tensor,
               style_features: Optional[torch.Tensor] = None,
               hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through trajectory predictor.
        
        Args:
            sequences: Input sequences [batch, seq_len, input_dim]
            style_features: Style conditioning features [batch, style_dim]
            hidden_state: Initial hidden state for RNN
            
        Returns:
            predictions: Output predictions [batch, seq_len, output_dim]
            final_hidden: Final hidden state
        """
        batch_size, seq_len, _ = sequences.shape
        
        # Apply style conditioning if provided
        if style_features is not None and self.input_projection is not None:
            # Broadcast style features to all timesteps
            style_expanded = style_features.unsqueeze(1).expand(-1, seq_len, -1)
            sequences_with_style = torch.cat([sequences, style_expanded], dim=-1)
            sequences = self.input_projection(sequences_with_style)
        
        # RNN forward pass
        rnn_output, final_hidden = self.rnn(sequences, hidden_state)
        
        # Apply attention if enabled
        if self.use_attention:
            rnn_output = self.attention(rnn_output)
        
        # Output projection
        predictions = self.output_projection(rnn_output)
        
        return predictions, final_hidden
    
    def predict_sequence(self,
                        initial_input: torch.Tensor,
                        sequence_length: int,
                        style_features: Optional[torch.Tensor] = None,
                        temperature: float = 1.0) -> torch.Tensor:
        """
        Generate trajectory sequence autoregressively.
        
        Args:
            initial_input: Initial input [batch, input_dim]
            sequence_length: Length of sequence to generate
            style_features: Style conditioning
            temperature: Sampling temperature
            
        Returns:
            generated_sequence: Generated trajectory [batch, seq_len, output_dim]
        """
        self.eval()
        batch_size = initial_input.shape[0]
        
        # Initialize outputs
        outputs = []
        hidden_state = None
        current_input = initial_input.unsqueeze(1)  # [batch, 1, input_dim]
        
        with torch.no_grad():
            for t in range(sequence_length):
                # Predict next step
                prediction, hidden_state = self.forward(
                    current_input, style_features, hidden_state
                )
                
                # Apply temperature sampling if needed
                if temperature != 1.0:
                    prediction = prediction / temperature
                
                outputs.append(prediction.squeeze(1))
                
                # Use prediction as next input (teacher forcing disabled)
                if t < sequence_length - 1:
                    current_input = prediction[:, -1:, :self.input_dim]
        
        return torch.stack(outputs, dim=1)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute trajectory prediction loss.
        
        Args:
            predictions: Predicted trajectories [batch, seq_len, output_dim]
            targets: Target trajectories [batch, seq_len, output_dim]
            mask: Sequence mask [batch, seq_len]
            
        Returns:
            loss: Prediction loss
        """
        # Basic MSE loss
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(loss)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss
    
    def compute_trajectory_metrics(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Compute trajectory-specific metrics.
        
        Args:
            predictions: Predicted trajectories
            targets: Target trajectories
            
        Returns:
            metrics: Dictionary of trajectory metrics
        """
        with torch.no_grad():
            # Position error
            position_error = F.mse_loss(predictions[..., :2], targets[..., :2])
            
            # Velocity error (if available)
            if predictions.shape[-1] >= 4:
                velocity_error = F.mse_loss(predictions[..., 2:4], targets[..., 2:4])
            else:
                velocity_error = 0.0
            
            # Smoothness metric (second derivative)
            pred_diff2 = torch.diff(predictions, n=2, dim=1)
            target_diff2 = torch.diff(targets, n=2, dim=1)
            smoothness_error = F.mse_loss(pred_diff2, target_diff2)
            
            # Path length similarity
            pred_path_length = torch.sum(torch.norm(torch.diff(predictions[..., :2], dim=1), dim=-1), dim=1)
            target_path_length = torch.sum(torch.norm(torch.diff(targets[..., :2], dim=1), dim=-1), dim=1)
            path_length_error = F.mse_loss(pred_path_length, target_path_length)
        
        return {
            'position_error': position_error.item(),
            'velocity_error': velocity_error.item() if isinstance(velocity_error, torch.Tensor) else velocity_error,
            'smoothness_error': smoothness_error.item(),
            'path_length_error': path_length_error.item()
        }


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for focusing on relevant parts of the trajectory.
    """
    
    def __init__(self, hidden_dim: int, attention_dim: int):
        """
        Initialize attention mechanism.
        
        Args:
            hidden_dim: Hidden dimension of RNN
            attention_dim: Attention projection dimension
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query_projection = nn.Linear(hidden_dim, attention_dim)
        self.key_projection = nn.Linear(hidden_dim, attention_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to hidden states.
        
        Args:
            hidden_states: RNN hidden states [batch, seq_len, hidden_dim]
            
        Returns:
            attended_output: Attention-weighted output [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to query, key, value
        queries = self.query_projection(hidden_states)  # [batch, seq_len, attention_dim]
        keys = self.key_projection(hidden_states)       # [batch, seq_len, attention_dim]
        values = self.value_projection(hidden_states)   # [batch, seq_len, hidden_dim]
        
        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2))  # [batch, seq_len, seq_len]
        attention_scores = attention_scores / (self.attention_dim ** 0.5)
        
        # Apply softmax
        attention_weights = self.softmax(attention_scores)
        
        # Apply attention to values
        attended_output = torch.bmm(attention_weights, values)  # [batch, seq_len, hidden_dim]
        
        return attended_output


class MultiModalTrajectoryPredictor(TrajectoryPredictor):
    """
    Multi-modal trajectory predictor that can handle different types of input
    (e.g., text, images, previous trajectories) to generate handwriting.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-modal trajectory predictor."""
        # Extract dimensions from config
        trajectory_dim = config.get('trajectory_dim', 4)  # x, y, vx, vy
        text_embedding_dim = config.get('text_embedding_dim', 256)
        image_feature_dim = config.get('image_feature_dim', 512)
        style_dim = config.get('style_dim', 64)
        
        # Combined input dimension
        total_input_dim = trajectory_dim + text_embedding_dim
        
        super().__init__(config, total_input_dim, trajectory_dim, style_dim)
        
        # Text encoder
        self.text_encoder = TextEncoder(config.get('text_encoder', {}))
        
        # Image encoder (optional)
        if config.get('use_image_conditioning', False):
            self.image_encoder = ImageEncoder(config.get('image_encoder', {}))
            total_input_dim += image_feature_dim
        else:
            self.image_encoder = None
        
        # Update input projection
        if self.style_dim > 0:
            self.input_projection = nn.Linear(total_input_dim + self.style_dim, self.input_dim)
        
        logger.info("Initialized MultiModalTrajectoryPredictor")
    
    def forward(self,
               trajectory_sequences: torch.Tensor,
               text_input: Optional[torch.Tensor] = None,
               image_input: Optional[torch.Tensor] = None,
               style_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with multi-modal inputs.
        
        Args:
            trajectory_sequences: Previous trajectory points [batch, seq_len, traj_dim]
            text_input: Text token sequences [batch, text_seq_len]
            image_input: Reference images [batch, channels, height, width]
            style_features: Style conditioning features [batch, style_dim]
            
        Returns:
            predictions: Trajectory predictions
            hidden_state: Final RNN hidden state
        """
        batch_size, seq_len, _ = trajectory_sequences.shape
        
        # Encode text
        if text_input is not None:
            text_features = self.text_encoder(text_input)
            # Broadcast to sequence length
            text_features = text_features.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            text_features = torch.zeros(batch_size, seq_len, self.text_encoder.output_dim, 
                                      device=trajectory_sequences.device)
        
        # Encode image
        if image_input is not None and self.image_encoder is not None:
            image_features = self.image_encoder(image_input)
            image_features = image_features.unsqueeze(1).expand(-1, seq_len, -1)
            combined_input = torch.cat([trajectory_sequences, text_features, image_features], dim=-1)
        else:
            combined_input = torch.cat([trajectory_sequences, text_features], dim=-1)
        
        # Standard forward pass
        return super().forward(combined_input, style_features)


class TextEncoder(nn.Module):
    """Simple text encoder for character/word-level text conditioning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize text encoder."""
        super().__init__()
        
        self.vocab_size = config.get('vocab_size', 256)  # ASCII characters
        self.embedding_dim = config.get('embedding_dim', 128)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.output_dim = config.get('output_dim', 256)
        
        # Character embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Text encoder RNN
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim * 2, self.output_dim)
    
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Encode text input.
        
        Args:
            text_tokens: Token indices [batch, text_len]
            
        Returns:
            text_features: Encoded text features [batch, output_dim]
        """
        # Embed tokens
        embedded = self.embedding(text_tokens)
        
        # RNN encoding
        rnn_output, (hidden, _) = self.rnn(embedded)
        
        # Use final hidden state (concatenate forward and backward)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        
        # Project to output dimension
        text_features = self.output_projection(final_hidden)
        
        return text_features


class ImageEncoder(nn.Module):
    """Simple image encoder for reference image conditioning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize image encoder."""
        super().__init__()
        
        self.output_dim = config.get('output_dim', 512)
        
        # Simple CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            
            nn.Flatten(),
            nn.Linear(128, self.output_dim)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode reference images.
        
        Args:
            images: Input images [batch, channels, height, width]
            
        Returns:
            image_features: Encoded image features [batch, output_dim]
        """
        return self.encoder(images)