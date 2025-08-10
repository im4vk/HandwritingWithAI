"""
Base Neural Network Classes
===========================

Common base classes and utilities for all AI models in the system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import logging
from abc import ABC, abstractmethod
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseNeuralNetwork(nn.Module, ABC):
    """
    Base class for all neural networks in the system.
    
    Provides common functionality for training, saving, loading, and evaluation.
    """
    
    def __init__(self, config: Dict[str, Any], name: str = "BaseModel"):
        """
        Initialize base neural network.
        
        Args:
            config: Model configuration dictionary
            name: Name of the model
        """
        super().__init__()
        self.config = config
        self.name = name
        self.device = self._setup_device(config.get('device', 'auto'))
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
        self.best_loss = float('inf')
        self.epoch = 0
        
        # Model metadata
        self.created_at = datetime.now().isoformat()
        self.training_stats = {
            'total_epochs': 0,
            'best_epoch': 0,
            'training_time': 0.0
        }
        
        logger.info(f"Initialized {self.name} on device: {self.device}")
    
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup computation device based on configuration."""
        if device_config == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using Apple Metal Performance Shaders")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU")
        else:
            device = torch.device(device_config)
            logger.info(f"Using specified device: {device}")
        
        return device
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass
    
    @abstractmethod
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for the specific model type."""
        pass
    
    def setup_optimizer(self, optimizer_config: Dict[str, Any]) -> None:
        """Setup optimizer and learning rate scheduler."""
        optimizer_type = optimizer_config.get('type', 'adam')
        lr = optimizer_config.get('learning_rate', 0.001)
        
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.parameters(),
                lr=lr,
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
        
        # Setup scheduler if specified
        if 'scheduler' in optimizer_config:
            scheduler_config = optimizer_config['scheduler']
            scheduler_type = scheduler_config.get('type', 'step')
            
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 100),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=scheduler_config.get('T_max', 1000)
                )
        
        logger.info(f"Setup optimizer: {optimizer_type} with lr={lr}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device
            if isinstance(batch_data, (list, tuple)):
                inputs, targets = batch_data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                inputs = batch_data.to(self.device)
                targets = None
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.forward(inputs)
            
            # Compute loss
            if targets is not None:
                loss = self.compute_loss(outputs, targets)
            else:
                loss = self.compute_loss(outputs, inputs)  # For autoencoders
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        
        if self.scheduler:
            self.scheduler.step()
        
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""
        self.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch_data in dataloader:
                # Move data to device
                if isinstance(batch_data, (list, tuple)):
                    inputs, targets = batch_data
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    inputs = batch_data.to(self.device)
                    targets = None
                
                # Forward pass
                outputs = self.forward(inputs)
                
                # Compute loss
                if targets is not None:
                    loss = self.compute_loss(outputs, targets)
                else:
                    loss = self.compute_loss(outputs, inputs)
                
                total_loss += loss.item()
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str, additional_info: Dict[str, Any] = None) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_name': self.name,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
            'created_at': self.created_at,
            'saved_at': datetime.now().isoformat()
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str, load_optimizer: bool = True) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        if load_optimizer and self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore training info
        self.epoch = checkpoint.get('epoch', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.training_stats = checkpoint.get('training_stats', {})
        
        logger.info(f"Loaded checkpoint from {filepath} (epoch {self.epoch})")
        return checkpoint
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'config': self.config,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'training_stats': self.training_stats,
            'created_at': self.created_at
        }
    
    def to_device(self) -> None:
        """Move model to configured device."""
        self.to(self.device)
        logger.info(f"Moved {self.name} to {self.device}")


class MultiLayerPerceptron(BaseNeuralNetwork):
    """
    General-purpose multi-layer perceptron.
    
    Useful as a building block for more complex architectures.
    """
    
    def __init__(self, config: Dict[str, Any], input_dim: int, output_dim: int, name: str = "MLP"):
        """
        Initialize MLP.
        
        Args:
            config: Model configuration
            input_dim: Input dimension
            output_dim: Output dimension
            name: Model name
        """
        super().__init__(config, name)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network
        hidden_layers = config.get('hidden_layers', [128, 64])
        activation = config.get('activation', 'relu')
        dropout_rate = config.get('dropout_rate', 0.1)
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        self.to_device()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return nn.functional.mse_loss(outputs, targets)


class ConvolutionalEncoder(BaseNeuralNetwork):
    """
    Convolutional encoder for processing handwriting images.
    """
    
    def __init__(self, config: Dict[str, Any], input_channels: int = 1, name: str = "ConvEncoder"):
        """
        Initialize convolutional encoder.
        
        Args:
            config: Model configuration
            input_channels: Number of input channels
            name: Model name
        """
        super().__init__(config, name)
        
        self.input_channels = input_channels
        channels = config.get('channels', [32, 64, 128])
        kernel_sizes = config.get('kernel_sizes', [3, 3, 3])
        
        layers = []
        prev_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            layers.extend([
                nn.Conv2d(prev_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            prev_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.output_dim = channels[-1]
        
        self.to_device()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.encoder(x)
        pooled = self.global_pool(features)
        return pooled.squeeze(-1).squeeze(-1)
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss."""
        return nn.functional.mse_loss(outputs, targets)