"""
Style Encoder for Handwriting Style Analysis
============================================

Neural networks for encoding and manipulating handwriting styles,
enabling style transfer and personalized handwriting generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging

from .base_model import BaseNeuralNetwork, ConvolutionalEncoder

logger = logging.getLogger(__name__)


class StyleEncoder(BaseNeuralNetwork):
    """
    Encoder for extracting style features from handwriting samples.
    
    Can work with both trajectory data and handwriting images
    to extract style characteristics like slant, pressure, speed, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize style encoder.
        
        Args:
            config: Style encoder configuration
        """
        super().__init__(config, "StyleEncoder")
        
        self.input_type = config.get('input_type', 'trajectory')  # 'trajectory' or 'image'
        self.style_dim = config.get('style_dim', 64)
        
        if self.input_type == 'trajectory':
            self._build_trajectory_encoder(config)
        elif self.input_type == 'image':
            self._build_image_encoder(config)
        elif self.input_type == 'both':
            self._build_multimodal_encoder(config)
        else:
            raise ValueError(f"Unsupported input type: {self.input_type}")
        
        # Style feature projector
        self.style_projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.style_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get('dropout', 0.1)),
            nn.Linear(self.style_dim * 2, self.style_dim)
        )
        
        # Style classifier (optional, for supervised style learning)
        self.use_classification = config.get('use_style_classification', False)
        if self.use_classification:
            self.num_style_classes = config.get('num_style_classes', 10)
            self.style_classifier = nn.Linear(self.style_dim, self.num_style_classes)
        
        self.to_device()
        
        logger.info(f"Initialized StyleEncoder with {self.input_type} input, style_dim={self.style_dim}")
    
    def _build_trajectory_encoder(self, config: Dict[str, Any]) -> None:
        """Build encoder for trajectory input."""
        trajectory_dim = config.get('trajectory_dim', 4)  # x, y, vx, vy
        hidden_dim = config.get('hidden_dim', 128)
        
        # LSTM for temporal modeling
        self.trajectory_encoder = nn.LSTM(
            input_size=trajectory_dim,
            hidden_size=hidden_dim,
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1),
            bidirectional=True,
            batch_first=True
        )
        
        # Statistical feature extractor
        self.stat_features = StatisticalFeatureExtractor(trajectory_dim)
        
        self.feature_dim = hidden_dim * 2 + self.stat_features.output_dim
    
    def _build_image_encoder(self, config: Dict[str, Any]) -> None:
        """Build encoder for image input."""
        # CNN encoder for handwriting images
        self.image_encoder = ConvolutionalEncoder(
            config=config.get('conv_config', {}),
            input_channels=config.get('input_channels', 1)
        )
        
        self.feature_dim = self.image_encoder.output_dim
    
    def _build_multimodal_encoder(self, config: Dict[str, Any]) -> None:
        """Build encoder for both trajectory and image input."""
        self._build_trajectory_encoder(config)
        traj_feature_dim = self.feature_dim
        
        self._build_image_encoder(config)
        img_feature_dim = self.feature_dim
        
        # Fusion layer
        self.feature_dim = traj_feature_dim + img_feature_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim // 2, self.feature_dim // 2)
        )
        self.feature_dim = self.feature_dim // 2
    
    def forward(self, 
               trajectory_input: Optional[torch.Tensor] = None,
               image_input: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through style encoder.
        
        Args:
            trajectory_input: Trajectory data [batch, seq_len, traj_dim]
            image_input: Handwriting images [batch, channels, height, width]
            
        Returns:
            outputs: Dictionary containing style features and optional classifications
        """
        if self.input_type == 'trajectory':
            features = self._encode_trajectory(trajectory_input)
        elif self.input_type == 'image':
            features = self._encode_image(image_input)
        elif self.input_type == 'both':
            features = self._encode_multimodal(trajectory_input, image_input)
        
        # Project to style space
        style_features = self.style_projector(features)
        
        outputs = {'style_features': style_features}
        
        # Optional style classification
        if self.use_classification:
            style_logits = self.style_classifier(style_features)
            outputs['style_logits'] = style_logits
        
        return outputs
    
    def _encode_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Encode trajectory data."""
        batch_size = trajectory.shape[0]
        
        # LSTM encoding
        lstm_output, (hidden, _) = self.trajectory_encoder(trajectory)
        
        # Use final hidden state (bidirectional)
        lstm_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [batch, hidden_dim * 2]
        
        # Statistical features
        stat_features = self.stat_features(trajectory)  # [batch, stat_dim]
        
        # Combine features
        combined_features = torch.cat([lstm_features, stat_features], dim=-1)
        
        return combined_features
    
    def _encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image data."""
        return self.image_encoder(image)
    
    def _encode_multimodal(self, 
                          trajectory: Optional[torch.Tensor],
                          image: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode both trajectory and image data."""
        features = []
        
        if trajectory is not None:
            traj_features = self._encode_trajectory(trajectory)
            features.append(traj_features)
        
        if image is not None:
            img_features = self._encode_image(image)
            features.append(img_features)
        
        if not features:
            raise ValueError("At least one input (trajectory or image) must be provided")
        
        # Concatenate features
        combined = torch.cat(features, dim=-1)
        
        # Fusion
        fused_features = self.fusion_layer(combined)
        
        return fused_features
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute style encoding loss."""
        if self.use_classification:
            # Classification loss
            return F.cross_entropy(outputs, targets)
        else:
            # Reconstruction or contrastive loss
            return F.mse_loss(outputs, targets)
    
    def extract_style_features(self, 
                             trajectory: Optional[torch.Tensor] = None,
                             image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract style features from input.
        
        Args:
            trajectory: Trajectory data
            image: Image data
            
        Returns:
            style_features: Extracted style features [batch, style_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(trajectory, image)
            return outputs['style_features']


class StatisticalFeatureExtractor(nn.Module):
    """
    Extracts statistical features from trajectory data
    that are indicative of handwriting style.
    """
    
    def __init__(self, trajectory_dim: int):
        """
        Initialize statistical feature extractor.
        
        Args:
            trajectory_dim: Dimension of trajectory data
        """
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.output_dim = 20  # Number of statistical features
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract statistical features from trajectory.
        
        Args:
            trajectory: Trajectory data [batch, seq_len, traj_dim]
            
        Returns:
            features: Statistical features [batch, output_dim]
        """
        batch_size, seq_len, _ = trajectory.shape
        
        # Position coordinates (assuming first 2 dimensions are x, y)
        positions = trajectory[..., :2]  # [batch, seq_len, 2]
        
        # Velocities (if available or compute from positions)
        if trajectory.shape[-1] >= 4:
            velocities = trajectory[..., 2:4]
        else:
            velocities = torch.diff(positions, dim=1, prepend=positions[:, :1])
        
        # Speed
        speeds = torch.norm(velocities, dim=-1)  # [batch, seq_len]
        
        # Extract statistical features
        features = []
        
        # 1. Mean and std of positions
        features.extend([
            positions.mean(dim=1).flatten(1),  # [batch, 2]
            positions.std(dim=1).flatten(1)    # [batch, 2]
        ])
        
        # 2. Mean and std of velocities
        features.extend([
            velocities.mean(dim=1).flatten(1),  # [batch, 2]
            velocities.std(dim=1).flatten(1)    # [batch, 2]
        ])
        
        # 3. Speed statistics
        features.extend([
            speeds.mean(dim=1, keepdim=True),   # [batch, 1]
            speeds.std(dim=1, keepdim=True),    # [batch, 1]
            speeds.max(dim=1, keepdim=True)[0], # [batch, 1]
            speeds.min(dim=1, keepdim=True)[0]  # [batch, 1]
        ])
        
        # 4. Trajectory curvature
        curvature = self._compute_curvature(positions)
        features.extend([
            curvature.mean(dim=1, keepdim=True),  # [batch, 1]
            curvature.std(dim=1, keepdim=True)    # [batch, 1]
        ])
        
        # 5. Writing pressure statistics (if available)
        if trajectory.shape[-1] >= 5:
            pressure = trajectory[..., 4]  # [batch, seq_len]
            features.extend([
                pressure.mean(dim=1, keepdim=True),  # [batch, 1]
                pressure.std(dim=1, keepdim=True)    # [batch, 1]
            ])
        else:
            # Placeholder zeros
            features.extend([
                torch.zeros(batch_size, 1, device=trajectory.device),
                torch.zeros(batch_size, 1, device=trajectory.device)
            ])
        
        # 6. Slant angle
        slant = self._compute_slant(positions)
        features.append(slant.unsqueeze(1))  # [batch, 1]
        
        # 7. Aspect ratio
        aspect_ratio = self._compute_aspect_ratio(positions)
        features.append(aspect_ratio.unsqueeze(1))  # [batch, 1]
        
        # Concatenate all features
        all_features = torch.cat(features, dim=1)  # [batch, total_features]
        
        # Pad or truncate to output_dim
        if all_features.shape[1] < self.output_dim:
            padding = torch.zeros(batch_size, self.output_dim - all_features.shape[1], 
                                device=trajectory.device)
            all_features = torch.cat([all_features, padding], dim=1)
        elif all_features.shape[1] > self.output_dim:
            all_features = all_features[:, :self.output_dim]
        
        return all_features
    
    def _compute_curvature(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute trajectory curvature."""
        # First and second derivatives
        vel = torch.diff(positions, dim=1)  # [batch, seq_len-1, 2]
        acc = torch.diff(vel, dim=1)        # [batch, seq_len-2, 2]
        
        # Curvature formula: |v x a| / |v|^3
        # For 2D: curvature = |vx*ay - vy*ax| / (vx^2 + vy^2)^(3/2)
        vel_aligned = vel[:, :-1]  # Align with acceleration
        
        cross_product = vel_aligned[..., 0] * acc[..., 1] - vel_aligned[..., 1] * acc[..., 0]
        speed_cubed = (torch.norm(vel_aligned, dim=-1) + 1e-8) ** 3
        
        curvature = torch.abs(cross_product) / speed_cubed
        
        # Pad to original sequence length
        padding = torch.zeros(positions.shape[0], 2, device=positions.device)
        curvature = torch.cat([padding, curvature], dim=1)
        
        return curvature
    
    def _compute_slant(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute average slant angle of handwriting."""
        # Compute local direction vectors
        directions = torch.diff(positions, dim=1)  # [batch, seq_len-1, 2]
        
        # Compute angles with respect to horizontal
        angles = torch.atan2(directions[..., 1], directions[..., 0])
        
        # Average angle (slant)
        mean_slant = torch.mean(angles, dim=1)  # [batch]
        
        return mean_slant
    
    def _compute_aspect_ratio(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute aspect ratio of handwriting bounding box."""
        # Bounding box
        min_pos = torch.min(positions, dim=1)[0]  # [batch, 2]
        max_pos = torch.max(positions, dim=1)[0]  # [batch, 2]
        
        width = max_pos[..., 0] - min_pos[..., 0]
        height = max_pos[..., 1] - min_pos[..., 1]
        
        # Aspect ratio (width / height)
        aspect_ratio = width / (height + 1e-8)
        
        return aspect_ratio


class StyleDiscriminator(BaseNeuralNetwork):
    """
    Discriminator for adversarial style learning.
    
    Distinguishes between different handwriting styles to enable
    style transfer and generation.
    """
    
    def __init__(self, config: Dict[str, Any], style_dim: int, num_styles: int):
        """
        Initialize style discriminator.
        
        Args:
            config: Discriminator configuration
            style_dim: Style feature dimension
            num_styles: Number of style classes
        """
        super().__init__(config, "StyleDiscriminator")
        
        self.style_dim = style_dim
        self.num_styles = num_styles
        
        # Discriminator network
        hidden_layers = config.get('hidden_layers', [128, 64, 32])
        
        layers = []
        prev_dim = style_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.get('dropout', 0.3))
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_styles))
        
        self.discriminator = nn.Sequential(*layers)
        
        self.to_device()
        
        logger.info(f"Initialized StyleDiscriminator: style_dim={style_dim}, num_styles={num_styles}")
    
    def forward(self, style_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            style_features: Style features [batch, style_dim]
            
        Returns:
            logits: Style classification logits [batch, num_styles]
        """
        return self.discriminator(style_features)
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute discriminator loss."""
        return F.cross_entropy(predictions, targets)


class StyleGAN(nn.Module):
    """
    Complete Style GAN for handwriting style transfer.
    
    Combines style encoder, generator, and discriminator for
    end-to-end style transfer training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Style GAN."""
        super().__init__()
        
        self.style_dim = config.get('style_dim', 64)
        self.num_styles = config.get('num_styles', 10)
        
        # Style encoder
        self.style_encoder = StyleEncoder(config.get('encoder', {}))
        
        # Style discriminator
        self.style_discriminator = StyleDiscriminator(
            config.get('discriminator', {}),
            self.style_dim,
            self.num_styles
        )
        
        # Loss weights
        self.reconstruction_weight = config.get('reconstruction_weight', 1.0)
        self.style_weight = config.get('style_weight', 0.1)
        self.adversarial_weight = config.get('adversarial_weight', 0.01)
        
        logger.info("Initialized StyleGAN")
    
    def encode_style(self, input_data: torch.Tensor) -> torch.Tensor:
        """Encode style from input data."""
        outputs = self.style_encoder(trajectory_input=input_data)
        return outputs['style_features']
    
    def discriminate_style(self, style_features: torch.Tensor) -> torch.Tensor:
        """Discriminate style class."""
        return self.style_discriminator(style_features)
    
    def compute_style_transfer_loss(self,
                                  source_data: torch.Tensor,
                                  target_style: torch.Tensor,
                                  generated_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute complete style transfer loss.
        
        Args:
            source_data: Source handwriting data
            target_style: Target style features
            generated_data: Generated handwriting with target style
            
        Returns:
            losses: Dictionary of loss components
        """
        # Encode styles
        source_style = self.encode_style(source_data)
        generated_style = self.encode_style(generated_data)
        
        # Reconstruction loss (content preservation)
        reconstruction_loss = F.mse_loss(generated_data, source_data)
        
        # Style transfer loss (style matching)
        style_loss = F.mse_loss(generated_style, target_style)
        
        # Adversarial loss (style discrimination)
        style_logits = self.discriminate_style(generated_style)
        # Placeholder for adversarial target (would need real implementation)
        adversarial_loss = torch.tensor(0.0, device=source_data.device)
        
        # Total loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                     self.style_weight * style_loss +
                     self.adversarial_weight * adversarial_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'style_loss': style_loss,
            'adversarial_loss': adversarial_loss
        }