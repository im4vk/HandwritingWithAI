"""
Generative Adversarial Imitation Learning (GAIL) Implementation
==============================================================

GAIL implementation specifically designed for learning human handwriting
trajectories and robot motion policies.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import logging
import time
from collections import deque
import random

from .base_model import BaseNeuralNetwork, MultiLayerPerceptron

logger = logging.getLogger(__name__)


class GAILPolicy(BaseNeuralNetwork):
    """
    ENHANCED Policy network for GAIL with LSTM memory and sophisticated encoding.
    
    For handwriting, this maps current robot state and writing context
    to robot joint commands and pen control with sequence memory.
    """
    
    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dim: int):
        """
        Initialize ENHANCED policy network with LSTM and larger capacity.
        
        Args:
            config: Policy configuration
            obs_dim: Observation dimension (robot state + context)
            action_dim: Action dimension (joint velocities + pen pressure)
        """
        super().__init__(config, "GAILPolicy")
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Enhanced network configuration
        hidden_layers = config.get('hidden_layers', [512, 256, 128])  # Larger layers
        activation = config.get('activation', 'relu')
        lstm_hidden_dim = config.get('lstm_hidden_dim', 256)
        lstm_num_layers = config.get('lstm_num_layers', 2)
        
        # Input encoding layer (better feature extraction)
        self.input_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_layers[0]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1))
        )
        
        # LSTM layer for sequence memory (crucial for handwriting)
        self.lstm = nn.LSTM(
            input_size=hidden_layers[1],
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=config.get('dropout_rate', 0.1) if lstm_num_layers > 1 else 0
        )
        
        # ENHANCED: Multi-Head Attention for temporal dependencies
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim,
            num_heads=8,
            dropout=config.get('dropout_rate', 0.1),
            batch_first=True
        )
        
        # ENHANCED: Letter-context attention for better letter understanding
        self.letter_attention = nn.MultiheadAttention(
            embed_dim=hidden_layers[1],
            num_heads=4,
            dropout=config.get('dropout_rate', 0.1),
            batch_first=True
        )
        
        # Enhanced post-processing with attention integration
        self.post_lstm_layers = nn.Sequential(
            nn.Linear(lstm_hidden_dim, hidden_layers[2]),
            nn.LayerNorm(hidden_layers[2]),
            nn.ReLU(),
            nn.Dropout(config.get('dropout_rate', 0.1)),
            nn.Linear(hidden_layers[2], hidden_layers[2] // 2),
            nn.ReLU()
        )
        
        # Separate heads for mean and std (better than single output)
        self.mean_head = nn.Sequential(
            nn.Linear(hidden_layers[2] // 2, action_dim),
            nn.Tanh()  # Bounded output
        )
        
        self.std_head = nn.Sequential(
            nn.Linear(hidden_layers[2] // 2, action_dim),
            nn.Softplus()  # Ensure positive std
        )
        
        # Action bounds
        self.action_bounds = config.get('action_bounds', [-1.0, 1.0])
        
        # LSTM hidden state
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.hidden_state = None
        
        self.to_device()
        
        logger.info(f"Initialized ENHANCED GAIL Policy: obs_dim={obs_dim}, action_dim={action_dim}")
        logger.info(f"  LSTM: {lstm_num_layers} layers, {lstm_hidden_dim} hidden units")
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, obs: torch.Tensor, reset_memory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass through LSTM policy network.
        
        Args:
            obs: Observations [batch_size, seq_len, obs_dim] or [batch_size, obs_dim]
            reset_memory: Whether to reset LSTM hidden state
            
        Returns:
            mean: Action means [batch_size, action_dim]
            std: Action standard deviations [batch_size, action_dim]
        """
        # Ensure input is on correct device
        obs = obs.to(self.device)
        
        batch_size = obs.size(0)
        
        # Handle single timestep input
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
            single_timestep = True
        else:
            single_timestep = False
        
        seq_len = obs.size(1)
        
        # Reset hidden state if requested or if batch size changed
        if reset_memory or self.hidden_state is None or self.hidden_state[0].size(1) != batch_size:
            self.reset_hidden_state(batch_size)
        
        # Input encoding
        encoded = self.input_encoder(obs.view(-1, self.obs_dim))  # [batch*seq, encoded_dim]
        encoded = encoded.view(batch_size, seq_len, -1)  # [batch, seq, encoded_dim]
        
        # ENHANCED: Letter-context attention before LSTM
        letter_attended, _ = self.letter_attention(encoded, encoded, encoded)
        encoded = encoded + letter_attended  # Residual connection
        
        # LSTM forward pass with memory
        lstm_out, self.hidden_state = self.lstm(encoded, self.hidden_state)
        
        # ENHANCED: Temporal attention on LSTM output
        temporal_attended, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        lstm_out = lstm_out + temporal_attended  # Residual connection
        
        # Use only the last timestep for action prediction
        last_output = lstm_out[:, -1, :]  # [batch_size, lstm_hidden_dim]
        
        # Post-LSTM processing with enhanced features
        processed = self.post_lstm_layers(last_output)
        
        # Separate mean and std heads
        mean = self.mean_head(processed)
        std = self.std_head(processed)
        
        # Apply action bounds to mean
        mean = mean * (self.action_bounds[1] - self.action_bounds[0]) / 2.0
        mean = mean + (self.action_bounds[1] + self.action_bounds[0]) / 2.0
        
        # Ensure minimum std for exploration
        std = std + 0.01
        
        return mean, std
    
    def reset_hidden_state(self, batch_size: int):
        """Reset LSTM hidden state."""
        self.hidden_state = (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(self.device),
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_dim).to(self.device)
        )
    
    def sample_action(self, obs: torch.Tensor, deterministic: bool = False, reset_memory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from enhanced LSTM policy.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action
            reset_memory: Whether to reset LSTM memory
            
        Returns:
            action: Sampled actions
            log_prob: Log probability of actions
        """
        mean, std = self.forward(obs, reset_memory=reset_memory)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action).sum(dim=-1)
        else:
            # Sample from normal distribution
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
            log_prob = normal.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_sequence_memory_state(self):
        """Get current LSTM hidden state for sequence generation."""
        return self.hidden_state
    
    def set_sequence_memory_state(self, hidden_state):
        """Set LSTM hidden state for sequence generation."""
        self.hidden_state = hidden_state
    
    def evaluate_action(self, obs: torch.Tensor, action: torch.Tensor, reset_memory: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions using enhanced LSTM policy.
        
        Args:
            obs: Observations
            action: Actions to evaluate
            reset_memory: Whether to reset LSTM memory
            
        Returns:
            log_prob: Log probability of actions
            entropy: Policy entropy
        """
        mean, std = self.forward(obs, reset_memory=reset_memory)
        
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(action).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)
        
        return log_prob, entropy
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute policy loss (not used directly in GAIL)."""
        return F.mse_loss(outputs, targets)


class GAILDiscriminator(BaseNeuralNetwork):
    """
    Discriminator network for GAIL that distinguishes between
    expert demonstrations and policy-generated trajectories.
    """
    
    def __init__(self, config: Dict[str, Any], state_action_dim: int):
        """
        Initialize discriminator network.
        
        Args:
            config: Discriminator configuration  
            state_action_dim: Dimension of state-action pairs
        """
        super().__init__(config, "GAILDiscriminator")
        
        self.state_action_dim = state_action_dim
        
        # Discriminator network configuration
        hidden_layers = config.get('hidden_layers', [128, 64, 32])
        activation = config.get('activation', 'relu')
        
        # Build discriminator network
        self.discriminator_net = MultiLayerPerceptron(
            config={
                'hidden_layers': hidden_layers,
                'activation': activation,
                'dropout_rate': config.get('dropout_rate', 0.3)
            },
            input_dim=state_action_dim,
            output_dim=1,  # Binary classification
            name="DiscriminatorMLP"
        )
        
        self.to_device()
        
        logger.info(f"Initialized GAIL Discriminator: input_dim={state_action_dim}")
    
    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.
        
        Args:
            state_action: State-action pairs [batch_size, state_action_dim]
            
        Returns:
            logits: Discriminator logits [batch_size, 1]
        """
        return self.discriminator_net(state_action)
    
    def predict_reward(self, state_action: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for GAIL training.
        
        Args:
            state_action: State-action pairs
            
        Returns:
            reward: Predicted rewards
        """
        with torch.no_grad():
            logits = self.forward(state_action)
            # Convert discriminator output to reward
            # Higher values = more expert-like = higher reward
            reward = -torch.log(torch.sigmoid(logits) + 1e-8)
        
        return reward.squeeze(-1)
    
    def compute_loss(self, expert_sa: torch.Tensor, policy_sa: torch.Tensor) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            expert_sa: Expert state-action pairs
            policy_sa: Policy state-action pairs
            
        Returns:
            loss: Discriminator loss
        """
        # Get discriminator outputs
        expert_logits = self.forward(expert_sa)
        policy_logits = self.forward(policy_sa)
        
        # Binary cross-entropy loss
        # Expert data should be classified as 1, policy data as 0
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )
        
        return expert_loss + policy_loss


class HandwritingGAIL:
    """
    Complete GAIL implementation for handwriting imitation learning.
    
    Combines policy and discriminator networks to learn from expert
    handwriting demonstrations.
    """
    
    def __init__(self, config: Dict[str, Any], obs_dim: int, action_dim: int):
        """
        Initialize GAIL system.
        
        Args:
            config: GAIL configuration
            obs_dim: Observation dimension
            action_dim: Action dimension
        """
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_action_dim = obs_dim + action_dim
        
        # Initialize networks
        policy_config = config.get('policy_network', {})
        discriminator_config = config.get('discriminator_network', {})
        
        self.policy = GAILPolicy(policy_config, obs_dim, action_dim)
        self.discriminator = GAILDiscriminator(discriminator_config, self.state_action_dim)
        
        # Setup optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.get('policy_lr', 3e-4)
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.get('discriminator_lr', 3e-4)
        )
        
        # Training parameters
        self.gamma = config.get('gamma', 0.99)
        self.lam = config.get('lambda', 0.95)
        self.clip_param = config.get('clip_param', 0.2)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        
        # Experience buffer
        self.buffer_size = config.get('buffer_size', 100000)
        self.batch_size = config.get('batch_size', 64)
        self.policy_buffer = deque(maxlen=self.buffer_size)
        self.expert_buffer = deque(maxlen=self.buffer_size)
        
        # Training statistics
        self.total_steps = 0
        self.policy_updates = 0
        self.discriminator_updates = 0
        
        logger.info("Initialized HandwritingGAIL system")
    
    def generate_letter_trajectory(self, letter: str, style_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate handwriting trajectory for a specific letter using learned AI policy.
        
        Args:
            letter: Letter to generate ('A', 'B', 'C', etc.)
            style_params: Optional style parameters (speed, size, slant, etc.)
            
        Returns:
            np.ndarray: Generated trajectory points [N, 3] (x, y, z coordinates)
        """
        if style_params is None:
            style_params = {}
        
        # Set up letter-specific context
        letter_context = self._create_letter_context(letter, style_params)
        
        # Generate trajectory using AI policy
        trajectory = self._generate_trajectory_with_policy(letter_context, style_params)
        
        return trajectory
    
    def generate_word_trajectory(self, word: str, style_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Generate handwriting trajectory for a word using learned AI policy.
        
        Args:
            word: Word to generate
            style_params: Optional style parameters
            
        Returns:
            np.ndarray: Generated trajectory points [N, 3]
        """
        if style_params is None:
            style_params = {
                'letter_spacing': 0.02,
                'word_spacing': 0.06,
                'base_size': 0.03,
                'start_position': [0.1, 0.15, 0.02]
            }
        
        full_trajectory = []
        current_x = style_params.get('start_position', [0.1, 0.15, 0.02])[0]
        
        for char in word.upper():
            if char == ' ':
                current_x += style_params.get('word_spacing', 0.06)
            elif char.isalpha():
                # Update style params for current position
                char_style = style_params.copy()
                char_style['start_position'] = [
                    current_x,
                    style_params.get('start_position', [0.1, 0.15, 0.02])[1],
                    style_params.get('start_position', [0.1, 0.15, 0.02])[2]
                ]
                
                # Generate letter using AI
                letter_traj = self.generate_letter_trajectory(char, char_style)
                full_trajectory.extend(letter_traj)
                
                # Update position for next letter
                if len(letter_traj) > 0:
                    current_x = letter_traj[-1][0] + style_params.get('letter_spacing', 0.02)
        
        return np.array(full_trajectory) if full_trajectory else np.array([[0.1, 0.15, 0.02]])
    
    def _create_letter_context(self, letter: str, style_params: Dict[str, Any]) -> torch.Tensor:
        """
        Create observation context for letter generation.
        
        This encodes the letter to generate and style parameters into the observation space
        that the AI policy was trained on.
        """
        # Letter encoding (one-hot for 26 letters)
        letter_idx = ord(letter) - ord('A') if letter.isalpha() else 0
        letter_encoding = np.zeros(26)
        if 0 <= letter_idx < 26:
            letter_encoding[letter_idx] = 1.0
        
        # Style parameters
        start_pos = style_params.get('start_position', [0.1, 0.15, 0.02])
        size = style_params.get('base_size', 0.03)
        speed = style_params.get('speed', 1.0)
        slant = style_params.get('slant', 0.0)
        
        # ENHANCED robot state (20 dimensions to total 46 with letter encoding)
        robot_state = np.array([
            *start_pos,  # Current pen position (3)
            0.0, 0.0, 0.0,  # Pen velocity (3)
            0.0, 0.0, 0.0, 1.0,  # Pen orientation quaternion (4)
            0.0,  # Pen pressure (1)
            size, speed, slant,  # Style parameters (3)
            0.0, 0.0,  # Progress indicators (2)
            0.0, 0.0,  # Letter complexity hints (2)  
            0.0, 0.0   # Additional neural context (2)
        ])
        
        # Combine letter encoding and robot state
        context = np.concatenate([letter_encoding, robot_state])
        
        return torch.FloatTensor(context).unsqueeze(0).to(self.policy.device)
    
    def _generate_trajectory_with_policy(self, initial_context: torch.Tensor, style_params: Dict[str, Any]) -> np.ndarray:
        """
        PURE NEURAL GENERATION: 100% AI-generated handwriting with NO hardcoded patterns.
        
        The neural network learns to generate letter shapes through training data,
        without any predefined mathematical letter patterns.
        """
        # Use ONLY pure neural generation - no hardcoded fallbacks
        return self._generate_pure_neural_trajectory(initial_context, style_params)
    
    def _apply_neural_enhancements(self, base_trajectory: np.ndarray, letter_context: torch.Tensor, style_params: Dict[str, Any]) -> np.ndarray:
        """
        Apply MINIMAL neural network enhancements while preserving letter structure.
        
        FIXED: Use very light touch to preserve recognizable letter shapes.
        """
        if len(base_trajectory) < 2:
            return base_trajectory
            
        enhanced_trajectory = base_trajectory.copy()
        
        # MINIMAL enhancement - just tiny variations for natural handwriting feel
        try:
            # Add very small random variations to simulate natural hand tremor
            for i in range(1, len(enhanced_trajectory)):  # Skip first point
                # Tiny natural variation (much smaller than before)
                natural_variation = np.random.normal(0, 0.0003, 3)  # Very small (0.3mm)
                enhanced_trajectory[i] += natural_variation
                
                # Ensure we don't drift too far from original path
                if i > 0:
                    # Keep points close to original trajectory
                    original_distance = np.linalg.norm(base_trajectory[i] - base_trajectory[i-1])
                    current_distance = np.linalg.norm(enhanced_trajectory[i] - enhanced_trajectory[i-1])
                    
                    # If distance changed too much, scale back
                    if abs(current_distance - original_distance) > original_distance * 0.1:  # Max 10% change
                        direction = (enhanced_trajectory[i] - enhanced_trajectory[i-1])
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            enhanced_trajectory[i] = enhanced_trajectory[i-1] + direction * original_distance
        
        except Exception as e:
            logger.debug(f"Neural enhancement failed: {e}, using base trajectory")
            return base_trajectory
        
        return enhanced_trajectory
    
    def _generate_pure_neural_trajectory(self, initial_context: torch.Tensor, style_params: Dict[str, Any]) -> np.ndarray:
        """
        PURE NEURAL trajectory generation - step-by-step neural network generation.
        
        This method uses ONLY the trained neural network to generate trajectories
        by sequential action prediction with LSTM memory.
        
        Args:
            initial_context: Letter and robot state context [batch, 42]
            style_params: Style parameters for generation
            
        Returns:
            Generated trajectory points as numpy array
        """
        logger.debug("🧠 Generating PURE NEURAL trajectory...")
        
        # Generation parameters
        max_steps = style_params.get('max_steps', 50)
        step_size = style_params.get('step_size', 0.001)
        start_pos = np.array(style_params.get('start_position', [0.1, 0.15, 0.02]))
        
        # Initialize trajectory
        trajectory_points = [start_pos.copy()]
        current_position = start_pos.copy()
        
        # Reset policy memory for new sequence
        self.policy.reset_hidden_state(1)
        
        # Initialize tracking for position updates
        if not hasattr(self, 'previous_position'):
            self.previous_position = np.zeros(3)
        self.previous_position = start_pos.copy()
        
        # Current observation state (handle both single timestep and batch)
        if len(initial_context.shape) == 1:
            current_obs = initial_context.clone()
        else:
            current_obs = initial_context[0].clone()  # Take first element if batched
        
        # Sequential neural generation
        with torch.no_grad():
            for step in range(max_steps):
                # Update position information in observation
                current_obs = self._update_observation_with_position(current_obs, current_position, step, max_steps)
                
                # Generate action using neural network
                action, log_prob = self.policy.sample_action(
                    current_obs.unsqueeze(0), 
                    deterministic=False,  # Use stochastic for natural variation
                    reset_memory=False    # Maintain LSTM memory
                )
                
                # Extract action components
                action_np = action[0].cpu().numpy()
                dx, dy, dz = action_np[:3]  # Movement deltas
                pressure = action_np[3] if len(action_np) > 3 else 0.5  # Pen pressure
                stop_flag = action_np[4] if len(action_np) > 4 else 0.0  # Stop writing
                
                # ENHANCED: Improved neural action scaling for better letter formation
                movement = np.array([dx, dy, dz]) * step_size
                
                # Adaptive scaling based on letter progress and type
                letter_scale = style_params.get('base_size', 0.03) / 0.03
                progress_factor = (step + 1) / max_steps  # Progress through letter
                
                # Scale movement to encourage letter-like shapes
                movement *= letter_scale
                
                # ENHANCED: Advanced movement processing for better letter quality
                movement = self._process_neural_movement_enhanced(
                    movement, step, max_steps, trajectory_points, style_params
                )
                
                # Update position with enhanced movement
                new_position = current_position + movement
                
                # Ensure reasonable bounds
                new_position[0] = np.clip(new_position[0], 0.05, 0.4)  # X bounds
                new_position[1] = np.clip(new_position[1], 0.1, 0.25)  # Y bounds  
                new_position[2] = np.clip(new_position[2], 0.01, 0.05) # Z bounds
                
                trajectory_points.append(new_position.copy())
                current_position = new_position
                
                # ENHANCED stopping logic for better letter completion
                if step >= 10:  # Minimum points for a recognizable letter
                    # Primary stopping: Neural network confidence
                    if stop_flag > 0.8:  # High confidence stop signal
                        logger.debug(f"🛑 Neural stop signal at step {step}")
                        break
                    
                    # Secondary stopping: Letter completion analysis
                    if step >= 15:  # Allow reasonable letter development
                        # Check if we've made meaningful progress
                        if len(trajectory_points) >= 5:
                            # Calculate trajectory span (letter should have some size)
                            points_array = np.array(trajectory_points)
                            x_span = points_array[:, 0].max() - points_array[:, 0].min()
                            y_span = points_array[:, 1].max() - points_array[:, 1].min()
                            
                            # Stop if letter has reasonable size OR network signals completion
                            if (x_span > 0.01 and y_span > 0.01) and stop_flag > 0.3:
                                logger.debug(f"📏 Letter completed with size {x_span:.3f}×{y_span:.3f} at step {step}")
                                break
                    
                    # Emergency stopping for very long sequences
                    if step > 40:
                        logger.debug(f"⏰ Maximum steps reached at {step}")
                        break
        
        trajectory = np.array(trajectory_points)
        logger.debug(f"✅ Pure neural trajectory generated: {len(trajectory)} points")
        
        return trajectory
    
    def _update_observation_with_position(self, obs: torch.Tensor, position: np.ndarray, step: int, max_steps: int) -> torch.Tensor:
        """
        Update observation vector with current position and progress information.
        
        This provides the neural network with spatial and temporal context.
        """
        updated_obs = obs.clone()
        
        # ENHANCED position information (update robot state portion of observation)
        # Observation structure: [26 letter encoding] + [20 enhanced robot state]
        pos_start_idx = 26  # After letter encoding
        
        # Update position (3 values)
        updated_obs[pos_start_idx:pos_start_idx+3] = torch.FloatTensor(position)
        
        # Update velocity information (3 values)
        if hasattr(self, 'previous_position') and len(self.previous_position) > 0:
            velocity = position - self.previous_position
            updated_obs[pos_start_idx+3:pos_start_idx+6] = torch.FloatTensor(velocity * 100)  # Scaled velocity
        
        # Update progress information (2 values starting at index 15)
        progress = step / max_steps
        updated_obs[pos_start_idx+15] = progress  # Current progress
        updated_obs[pos_start_idx+16] = 1.0 - progress  # Remaining progress
        
        # Update letter complexity hints (2 values starting at index 17)
        # These help the network understand letter complexity
        letter_complexity = step / 20.0  # Normalized step complexity
        updated_obs[pos_start_idx+17] = letter_complexity
        updated_obs[pos_start_idx+18] = min(1.0, progress * 2)  # Early vs late progress
        
        # Additional neural context (2 values starting at index 19)
        updated_obs[pos_start_idx+19] = step / 50.0  # Normalized step for extended range
        
        # Store current position for next iteration
        self.previous_position = position.copy()
        
        return updated_obs
    
    def load_synthetic_expert_data(self):
        """
        Load ENHANCED synthetic expert demonstrations with multiple styles and variations.
        This provides rich, diverse training data for neural network learning.
        """
        logger.info("Loading ENHANCED synthetic expert handwriting demonstrations...")
        
        # Generate diverse expert trajectories for each letter
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        handwriting_styles = ['neat', 'cursive', 'rushed', 'careful', 'artistic']
        expert_observations = []
        expert_actions = []
        
        total_demonstrations = 0
        
        for letter in letters:
            logger.info(f"Generating training data for letter '{letter}'...")
            
            for style in handwriting_styles:
                # Generate multiple variations per style (3-5 variations each)
                num_variations = 4 if style in ['neat', 'careful'] else 3
                
                for variation in range(num_variations):
                    # Create style-specific parameters
                    style_params = self._create_style_parameters(style, variation)
                    
                    # Add consistent step size for proper action scaling in training
                    style_params['step_size'] = 0.001  # Consistent across all training data
                    style_params['max_steps'] = 30  # Reasonable trajectory length
                    
                    # Generate synthetic expert demonstration for this letter+style+variation
                    obs_sequence, action_sequence = self._generate_enhanced_expert_letter(
                        letter, style, variation, style_params
                    )
                    
                    expert_observations.extend(obs_sequence)
                    expert_actions.extend(action_sequence)
                    total_demonstrations += 1
        
        # Add to expert buffer
        if expert_observations and expert_actions:
            self.add_expert_data(
                np.array(expert_observations), 
                np.array(expert_actions)
            )
            logger.info(f"Added {len(expert_observations)} expert demonstrations from {total_demonstrations} letter variations")
            logger.info(f"Training data: {len(letters)} letters × {len(handwriting_styles)} styles × 3-4 variations = {total_demonstrations} total examples")
            
            # ENHANCED training focused on letter shape accuracy
            logger.info("🚀 Starting enhanced training focused on letter shape learning...")
            self.train_on_expert_data(
                num_epochs=30,  # More epochs for better learning
                validation_interval=5,  # More frequent validation
                early_stopping_patience=8  # More patience for complex learning
            )
            logger.info("✅ Enhanced GAIL model training completed with shape focus")
    
    def _create_style_parameters(self, style: str, variation: int) -> Dict[str, Any]:
        """Create style-specific parameters for handwriting generation."""
        base_params = {
            'start_position': [0.1, 0.15, 0.02],
            'base_size': 0.03,
            'speed': 1.0
        }
        
        # Add style-specific variations
        if style == 'neat':
            base_params.update({
                'size_variation': 0.002 + variation * 0.001,  # Small size variation
                'speed_variation': 0.9 + variation * 0.05,    # Consistent speed
                'pressure_variation': 0.5 + variation * 0.1,  # Moderate pressure
                'smoothness': 0.95 - variation * 0.05,        # High smoothness
                'letter_spacing': 0.025 + variation * 0.005   # Regular spacing
            })
        elif style == 'cursive':
            base_params.update({
                'size_variation': 0.005 + variation * 0.002,  # Flowing variation
                'speed_variation': 1.2 + variation * 0.1,     # Faster writing
                'pressure_variation': 0.3 + variation * 0.1,  # Lighter pressure
                'smoothness': 0.85 + variation * 0.05,        # Flowing curves
                'letter_spacing': 0.015 + variation * 0.003,  # Closer spacing
                'cursive_flow': True
            })
        elif style == 'rushed':
            base_params.update({
                'size_variation': 0.008 + variation * 0.003,  # Irregular size
                'speed_variation': 1.5 + variation * 0.2,     # Much faster
                'pressure_variation': 0.7 + variation * 0.15, # Variable pressure
                'smoothness': 0.6 - variation * 0.1,          # Less smooth
                'letter_spacing': 0.02 + variation * 0.01,    # Irregular spacing
                'jerkiness': 0.3 + variation * 0.1
            })
        elif style == 'careful':
            base_params.update({
                'size_variation': 0.001 + variation * 0.0005, # Very consistent
                'speed_variation': 0.7 + variation * 0.05,    # Slower speed
                'pressure_variation': 0.6 + variation * 0.05, # Steady pressure
                'smoothness': 0.98 - variation * 0.02,        # Very smooth
                'letter_spacing': 0.03 + variation * 0.002,   # Careful spacing
                'precision': 0.95 + variation * 0.02
            })
        elif style == 'artistic':
            base_params.update({
                'size_variation': 0.01 + variation * 0.005,   # Creative variation
                'speed_variation': 1.1 + variation * 0.15,    # Expressive speed
                'pressure_variation': 0.4 + variation * 0.2,  # Dynamic pressure
                'smoothness': 0.8 + variation * 0.1,          # Artistic flow
                'letter_spacing': 0.025 + variation * 0.008,  # Creative spacing
                'flourish': True,
                'artistic_flair': 0.2 + variation * 0.1
            })
        
        return base_params
    
    def _generate_enhanced_expert_letter(self, letter: str, style: str, variation: int, style_params: Dict[str, Any]):
        """Generate enhanced expert demonstration with specific style and variation."""
        observations = []
        actions = []
        
        # Create letter context with style information
        enhanced_style_params = style_params.copy()
        enhanced_style_params.update({
            'start_position': [
                0.1 + np.random.normal(0, style_params.get('size_variation', 0.001)),
                0.15 + np.random.normal(0, style_params.get('size_variation', 0.001)),
                0.02
            ],
            'base_size': 0.03 + np.random.normal(0, style_params.get('size_variation', 0.001)),
            'speed': style_params.get('speed_variation', 1.0),
            'style': style,
            'variation': variation
        })
        
        context = self._create_letter_context(letter, enhanced_style_params)
        
        # Generate style-enhanced trajectory
        base_trajectory = self._generate_synthetic_letter_trajectory(letter, enhanced_style_params)
        enhanced_trajectory = self._apply_style_variations(base_trajectory, style_params)
        
        # Convert trajectory to observation-action pairs with style awareness
        for i in range(len(enhanced_trajectory) - 1):
            current_pos = enhanced_trajectory[i]
            next_pos = enhanced_trajectory[i + 1]
            
            # Update context with current position and style
            context_copy = context.clone()
            context_copy[0, 26:29] = torch.FloatTensor(current_pos)
            
            # IMPROVED: Better action space scaling for neural learning
            movement = next_pos - current_pos
            
            # Scale movement to action space that neural network can learn effectively
            # Use consistent step size for training
            step_size = enhanced_style_params.get('step_size', 0.001)
            scaled_movement = movement / step_size  # Normalize to action space
            
            # Ensure movements are in learnable range [-1, 1]
            scaled_movement = np.clip(scaled_movement, -1.0, 1.0)
            
            # Better pressure calculation
            pressure = self._calculate_style_pressure(style_params, i, len(enhanced_trajectory))
            pressure = np.clip(pressure, 0.0, 1.0)  # Normalize pressure
            
            # More gradual stopping
            stop_flag = 0.0
            if i >= len(enhanced_trajectory) - 3:  # Start signaling stop 3 steps before end
                stop_flag = (i - (len(enhanced_trajectory) - 4)) / 3.0
            stop_flag = np.clip(stop_flag, 0.0, 1.0)
            
            # Create well-scaled action
            action = np.array([
                scaled_movement[0], scaled_movement[1], scaled_movement[2],
                pressure, stop_flag
            ])
            
            # Add controlled noise for robustness
            if variation > 0:
                noise_scale = 0.02 * variation  # Small, controlled noise
                noise = np.random.normal(0, noise_scale, 5)
                action += noise
                action = np.clip(action, -1.0, 1.0)  # Keep in valid range
            
            observations.append(context_copy.cpu().numpy()[0])
            actions.append(action)
        
        return observations, actions
    
    def _apply_style_variations(self, trajectory: np.ndarray, style_params: Dict[str, Any]) -> np.ndarray:
        """Apply style-specific variations to base trajectory."""
        enhanced_trajectory = trajectory.copy()
        
        # Get style parameters
        smoothness = style_params.get('smoothness', 0.8)
        jerkiness = style_params.get('jerkiness', 0.0)
        pressure_var = style_params.get('pressure_variation', 0.5)
        
        # Apply smoothness/jerkiness
        for i in range(1, len(enhanced_trajectory) - 1):
            if smoothness > 0.9:  # Very smooth (careful/neat)
                # Smooth between neighbors
                prev_pos = enhanced_trajectory[i-1]
                next_pos = enhanced_trajectory[i+1]
                enhanced_trajectory[i] = 0.7 * enhanced_trajectory[i] + 0.15 * prev_pos + 0.15 * next_pos
            elif jerkiness > 0.2:  # Jerky (rushed)
                # Add small random jerks
                jerk = np.random.normal(0, jerkiness * 0.001, 3)
                enhanced_trajectory[i] += jerk
        
        # Apply cursive flow
        if style_params.get('cursive_flow', False):
            # Add flowing connections between points
            for i in range(1, len(enhanced_trajectory)):
                # Slight curve towards next point
                if i < len(enhanced_trajectory) - 1:
                    direction = enhanced_trajectory[i+1] - enhanced_trajectory[i-1]
                    curve = direction * 0.1 * np.random.uniform(0.5, 1.0)
                    enhanced_trajectory[i] += curve * 0.3
        
        # Apply artistic flourishes
        if style_params.get('flourish', False):
            # Add slight artistic curves
            artistic_flair = style_params.get('artistic_flair', 0.1)
            for i in range(len(enhanced_trajectory)):
                flourish = np.array([
                    np.sin(i * 0.5) * artistic_flair * 0.002,
                    np.cos(i * 0.3) * artistic_flair * 0.002,
                    0
                ])
                enhanced_trajectory[i] += flourish
        
        return enhanced_trajectory
    
    def _calculate_style_pressure(self, style_params: Dict[str, Any], step: int, total_steps: int) -> float:
        """Calculate style-appropriate pressure for this step."""
        base_pressure = style_params.get('pressure_variation', 0.5)
        pressure_var = style_params.get('pressure_variation', 0.1)
        
        # Pressure varies throughout the stroke
        progress = step / max(total_steps - 1, 1)
        
        # Different styles have different pressure patterns
        style = style_params.get('style', 'neat')
        if style == 'careful':
            # Steady, consistent pressure
            pressure = base_pressure + np.random.normal(0, 0.05)
        elif style == 'rushed':
            # Variable pressure with quick changes
            pressure = base_pressure + np.random.normal(0, pressure_var) + 0.2 * np.sin(progress * 8)
        elif style == 'artistic':
            # Dynamic pressure for expression
            pressure = base_pressure + 0.3 * np.sin(progress * 4) + np.random.normal(0, 0.1)
        elif style == 'cursive':
            # Lighter pressure for flowing strokes
            pressure = base_pressure * 0.8 + np.random.normal(0, 0.05)
        else:  # neat
            # Moderate, consistent pressure
            pressure = base_pressure + np.random.normal(0, 0.08)
        
        return np.clip(pressure, 0.1, 1.0)
    
    def _add_style_noise(self, action: np.ndarray, style_params: Dict[str, Any]) -> np.ndarray:
        """Add style-appropriate noise to actions."""
        style = style_params.get('style', 'neat')
        noise_scale = 0.001
        
        if style == 'rushed':
            noise_scale = 0.003  # More erratic
        elif style == 'careful':
            noise_scale = 0.0005  # Very precise
        elif style == 'artistic':
            noise_scale = 0.002  # Creative variation
        elif style == 'cursive':
            noise_scale = 0.0015  # Flowing variation
        
        # Add noise to movement components only (not pressure/stop)
        noise = np.random.normal(0, noise_scale, len(action))
        noise[3:] = 0  # Don't add noise to pressure/stop flags
        
        return action + noise
    
    def validate_letter_quality(self, test_letters: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive letter quality validation system.
        
        Measures neural network performance across multiple metrics:
        - Shape similarity to expert patterns
        - Trajectory smoothness
        - Letter recognizability
        - Consistency across variations
        
        Args:
            test_letters: Letters to test (default: ['A', 'O', 'C', 'X', 'H'])
            
        Returns:
            Dict with quality scores for each metric
        """
        if test_letters is None:
            test_letters = ['A', 'O', 'C', 'X', 'H', 'M', 'S', 'U', 'B']
        
        logger.info(f"🧪 Starting validation for letters: {test_letters}")
        
        total_scores = {
            'shape_similarity': 0.0,
            'smoothness': 0.0,
            'recognizability': 0.0,
            'consistency': 0.0,
            'overall_quality': 0.0
        }
        
        detailed_results = {}
        
        for letter in test_letters:
            logger.info(f"Testing letter '{letter}'...")
            
            # Generate multiple variations for consistency testing
            letter_scores = self._validate_single_letter(letter, num_variations=3)
            detailed_results[letter] = letter_scores
            
            # Accumulate scores
            for metric in total_scores:
                total_scores[metric] += letter_scores[metric]
        
        # Calculate averages
        num_letters = len(test_letters)
        for metric in total_scores:
            total_scores[metric] /= num_letters
        
        # Log results
        logger.info("🏆 Validation Results:")
        for metric, score in total_scores.items():
            logger.info(f"   {metric.replace('_', ' ').title()}: {score:.3f}")
        
        return {
            'summary': total_scores,
            'detailed': detailed_results
        }
    
    def _validate_single_letter(self, letter: str, num_variations: int = 3) -> Dict[str, float]:
        """Validate quality metrics for a single letter."""
        
        # Generate expert reference trajectory
        expert_style_params = {'start_position': [0.1, 0.15, 0.02], 'base_size': 0.03}
        expert_trajectory = self._generate_synthetic_letter_trajectory(letter, expert_style_params)
        
        # Generate neural network trajectories (multiple variations)
        neural_trajectories = []
        for variation in range(num_variations):
            style_params = {
                'start_position': [0.1 + variation * 0.001, 0.15, 0.02],
                'base_size': 0.03,
                'max_steps': 50,
                'step_size': 0.001
            }
            neural_traj = self._generate_neural_trajectory_for_validation(letter, style_params)
            neural_trajectories.append(neural_traj)
        
        # Calculate metrics
        shape_scores = []
        smoothness_scores = []
        recognizability_scores = []
        
        for neural_traj in neural_trajectories:
            # 1. Shape Similarity
            shape_score = self._calculate_shape_similarity(expert_trajectory, neural_traj)
            shape_scores.append(shape_score)
            
            # 2. Smoothness
            smoothness_score = self._calculate_trajectory_smoothness(neural_traj)
            smoothness_scores.append(smoothness_score)
            
            # 3. Recognizability
            recognizability_score = self._calculate_letter_recognizability(neural_traj, letter)
            recognizability_scores.append(recognizability_score)
        
        # 4. Consistency (variation across generations)
        consistency_score = self._calculate_consistency(neural_trajectories)
        
        return {
            'shape_similarity': np.mean(shape_scores),
            'smoothness': np.mean(smoothness_scores),
            'recognizability': np.mean(recognizability_scores),
            'consistency': consistency_score,
            'overall_quality': np.mean([
                np.mean(shape_scores),
                np.mean(smoothness_scores),
                np.mean(recognizability_scores),
                consistency_score
            ])
        }
    
    def _generate_neural_trajectory_for_validation(self, letter: str, style_params: Dict[str, Any]) -> np.ndarray:
        """Generate trajectory using current neural network for validation."""
        # For now, use the hybrid approach but this will be replaced with pure neural
        context = self._create_letter_context(letter, style_params)
        trajectory = self._generate_trajectory_with_policy(context, style_params)
        return trajectory
    
    def _calculate_shape_similarity(self, expert_traj: np.ndarray, neural_traj: np.ndarray) -> float:
        """
        Calculate shape similarity between expert and neural trajectories.
        Uses Dynamic Time Warping (DTW) for trajectory comparison.
        """
        if len(expert_traj) == 0 or len(neural_traj) == 0:
            return 0.0
        
        # Normalize trajectories to same coordinate system
        expert_norm = self._normalize_trajectory(expert_traj)
        neural_norm = self._normalize_trajectory(neural_traj)
        
        # Calculate DTW distance
        dtw_distance = self._calculate_dtw_distance(expert_norm, neural_norm)
        
        # Convert distance to similarity score (0-1, higher is better)
        max_possible_distance = np.sqrt(2)  # Normalized coordinate system
        similarity = max(0.0, 1.0 - (dtw_distance / max_possible_distance))
        
        return similarity
    
    def _normalize_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """Normalize trajectory to [0,1] coordinate system."""
        if len(trajectory) == 0:
            return trajectory
        
        traj_2d = trajectory[:, :2]  # Only x,y coordinates
        
        # Find bounding box
        min_coords = np.min(traj_2d, axis=0)
        max_coords = np.max(traj_2d, axis=0)
        
        # Avoid division by zero
        ranges = max_coords - min_coords
        ranges[ranges == 0] = 1.0
        
        # Normalize to [0,1]
        normalized = (traj_2d - min_coords) / ranges
        
        return normalized
    
    def _calculate_dtw_distance(self, traj1: np.ndarray, traj2: np.ndarray) -> float:
        """
        Calculate Dynamic Time Warping distance between two trajectories.
        Simplified DTW implementation for trajectory comparison.
        """
        n, m = len(traj1), len(traj2)
        
        # Create distance matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Euclidean distance between points
                point_distance = np.linalg.norm(traj1[i-1] - traj2[j-1])
                
                # DTW recurrence relation
                dtw_matrix[i, j] = point_distance + min(
                    dtw_matrix[i-1, j],      # Insertion
                    dtw_matrix[i, j-1],      # Deletion
                    dtw_matrix[i-1, j-1]     # Match
                )
        
        # Normalize by path length
        path_length = n + m
        return dtw_matrix[n, m] / path_length
    
    def _calculate_trajectory_smoothness(self, trajectory: np.ndarray) -> float:
        """
        Calculate trajectory smoothness based on acceleration changes.
        Higher score = smoother trajectory.
        """
        if len(trajectory) < 3:
            return 1.0  # Too short to measure
        
        # Calculate velocities
        velocities = np.diff(trajectory[:, :2], axis=0)
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate jerk (change in acceleration)
        jerks = np.diff(accelerations, axis=0)
        
        # Smoothness is inverse of average jerk magnitude
        if len(jerks) == 0:
            return 1.0
        
        avg_jerk = np.mean(np.linalg.norm(jerks, axis=1))
        
        # Convert to 0-1 score (lower jerk = higher smoothness)
        # Scale factor chosen empirically
        smoothness = np.exp(-avg_jerk * 1000)
        
        return np.clip(smoothness, 0.0, 1.0)
    
    def _calculate_letter_recognizability(self, trajectory: np.ndarray, expected_letter: str) -> float:
        """
        Calculate how recognizable the generated letter is.
        Uses geometric feature analysis.
        """
        if len(trajectory) < 3:
            return 0.0
        
        # Extract geometric features
        features = self._extract_letter_features(trajectory)
        
        # Compare with expected letter features
        expected_features = self._get_expected_letter_features(expected_letter)
        
        # Calculate feature similarity
        feature_score = self._compare_letter_features(features, expected_features)
        
        return feature_score
    
    def _extract_letter_features(self, trajectory: np.ndarray) -> Dict[str, float]:
        """Extract geometric features from trajectory."""
        traj_2d = trajectory[:, :2]
        
        # Bounding box features
        min_coords = np.min(traj_2d, axis=0)
        max_coords = np.max(traj_2d, axis=0)
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        aspect_ratio = width / max(height, 1e-6)
        
        # Path features
        total_length = np.sum(np.linalg.norm(np.diff(traj_2d, axis=0), axis=1))
        
        # Curvature features
        curvatures = self._calculate_curvature(traj_2d)
        avg_curvature = np.mean(np.abs(curvatures)) if len(curvatures) > 0 else 0
        
        # Symmetry features (simplified)
        center_x = (min_coords[0] + max_coords[0]) / 2
        left_points = np.sum(traj_2d[:, 0] < center_x)
        right_points = np.sum(traj_2d[:, 0] >= center_x)
        symmetry = 1.0 - abs(left_points - right_points) / len(traj_2d)
        
        return {
            'aspect_ratio': aspect_ratio,
            'total_length': total_length,
            'avg_curvature': avg_curvature,
            'symmetry': symmetry,
            'num_points': len(trajectory)
        }
    
    def _calculate_curvature(self, trajectory: np.ndarray) -> np.ndarray:
        """Calculate curvature at each point along the trajectory."""
        if len(trajectory) < 3:
            return np.array([])
        
        # First and second derivatives
        dx = np.gradient(trajectory[:, 0])
        dy = np.gradient(trajectory[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        
        # Curvature formula
        curvature = (dx * d2y - dy * d2x) / ((dx**2 + dy**2)**1.5 + 1e-6)
        
        return curvature
    
    def _get_expected_letter_features(self, letter: str) -> Dict[str, float]:
        """Get expected geometric features for a letter."""
        # Simplified expected features for common letters
        feature_templates = {
            'A': {'aspect_ratio': 0.7, 'avg_curvature': 0.1, 'symmetry': 0.8},
            'O': {'aspect_ratio': 1.0, 'avg_curvature': 0.8, 'symmetry': 0.9},
            'C': {'aspect_ratio': 0.8, 'avg_curvature': 0.6, 'symmetry': 0.5},
            'X': {'aspect_ratio': 0.8, 'avg_curvature': 0.0, 'symmetry': 0.9},
            'H': {'aspect_ratio': 0.6, 'avg_curvature': 0.0, 'symmetry': 0.9},
            'M': {'aspect_ratio': 0.8, 'avg_curvature': 0.2, 'symmetry': 0.8},
            'S': {'aspect_ratio': 0.6, 'avg_curvature': 0.7, 'symmetry': 0.3},
            'U': {'aspect_ratio': 0.7, 'avg_curvature': 0.5, 'symmetry': 0.8},
            'B': {'aspect_ratio': 0.6, 'avg_curvature': 0.4, 'symmetry': 0.6}
        }
        
        return feature_templates.get(letter, {'aspect_ratio': 0.7, 'avg_curvature': 0.3, 'symmetry': 0.7})
    
    def _compare_letter_features(self, features: Dict[str, float], expected: Dict[str, float]) -> float:
        """Compare extracted features with expected features."""
        scores = []
        
        for feature_name in ['aspect_ratio', 'avg_curvature', 'symmetry']:
            if feature_name in features and feature_name in expected:
                actual = features[feature_name]
                target = expected[feature_name]
                
                # Calculate similarity (closer to target = higher score)
                diff = abs(actual - target)
                similarity = max(0.0, 1.0 - diff)
                scores.append(similarity)
        
        return np.mean(scores) if scores else 0.0
    
    def _calculate_consistency(self, trajectories: List[np.ndarray]) -> float:
        """Calculate consistency across multiple generations of the same letter."""
        if len(trajectories) < 2:
            return 1.0
        
        # Compare each pair of trajectories
        similarities = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                similarity = self._calculate_shape_similarity(trajectories[i], trajectories[j])
                similarities.append(similarity)
        
        # Consistency is the average pairwise similarity
        return np.mean(similarities) if similarities else 0.0
        
    def _generate_synthetic_expert_letter(self, letter: str):
        """Generate synthetic expert demonstration for a specific letter."""
        observations = []
        actions = []
        
        # Create letter context
        style_params = {
            'start_position': [0.1, 0.15, 0.02],
            'base_size': 0.03,
            'speed': 1.0
        }
        context = self._create_letter_context(letter, style_params)
        
        # Generate realistic trajectory pattern based on letter
        trajectory = self._generate_synthetic_letter_trajectory(letter, style_params)
        
        # Convert trajectory to observation-action pairs
        for i in range(len(trajectory) - 1):
            current_pos = trajectory[i]
            next_pos = trajectory[i + 1]
            
            # Update context with current position
            context_copy = context.clone()
            context_copy[0, 26:29] = torch.FloatTensor(current_pos)
            
            # Calculate action (movement + pressure + stop)
            movement = next_pos - current_pos
            pressure = 0.5  # Moderate pressure
            stop_flag = 1.0 if i == len(trajectory) - 2 else 0.0
            
            action = np.array([*movement, pressure, stop_flag])
            
            observations.append(context_copy.cpu().numpy()[0])
            actions.append(action)
        
        return observations, actions
    
    def _generate_synthetic_letter_trajectory(self, letter: str, style_params: Dict[str, Any]):
        """Generate sophisticated, realistic trajectory for each letter."""
        start_pos = np.array(style_params['start_position'])
        size = style_params['base_size']
        
        # SOPHISTICATED LETTER PATTERNS - Based on actual handwriting mechanics
        if letter == 'A':
            # A: Left stroke, right stroke, crossbar
            return np.array([
                start_pos,                                    # Start bottom left
                start_pos + [size*0.5, size, 0],            # Go to apex
                start_pos + [size, 0, 0],                   # Down to bottom right
                start_pos + [size*0.25, size*0.4, 0],      # Crossbar left
                start_pos + [size*0.75, size*0.4, 0]       # Crossbar right
            ])
            
        elif letter == 'B':
            # B: Vertical line + two curves
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Top curve
            for i in range(4):
                t = i / 3
                x = start_pos[0] + size*0.6*t
                y = start_pos[1] + size*(0.8 + 0.2*np.sin(np.pi*t))
                points.append([x, y, start_pos[2]])
            # Middle horizontal
            points.append(start_pos + [0, size*0.5, 0])
            # Bottom curve  
            for i in range(4):
                t = i / 3
                x = start_pos[0] + size*0.7*t
                y = start_pos[1] + size*(0.3 - 0.3*np.sin(np.pi*t))
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'C':
            # C: Open circle arc
            points = []
            for i in range(12):
                angle = np.pi/6 + 4*np.pi/3 * i / 11  # Open arc
                x = start_pos[0] + size*0.5 + size*0.4*np.cos(angle)
                y = start_pos[1] + size*0.5 + size*0.4*np.sin(angle)
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'D':
            # D: Vertical line + arc
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Curved right side
            for i in range(8):
                t = i / 7
                angle = np.pi/2 - np.pi*t
                x = start_pos[0] + size*0.6*(1 + 0.6*np.cos(angle))
                y = start_pos[1] + size*(0.5 + 0.5*np.sin(angle))
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'E':
            # E: Vertical + three horizontals
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Top horizontal
            for i in range(3):
                points.append(start_pos + [size*i/2*0.8, size, 0])
            # Middle horizontal  
            for i in range(3):
                points.append(start_pos + [size*i/2*0.6, size*0.5, 0])
            # Bottom horizontal
            for i in range(3):
                points.append(start_pos + [size*i/2*0.8, 0, 0])
            return np.array(points)
            
        elif letter == 'F':
            # F: Like E but no bottom horizontal
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Top horizontal
            for i in range(3):
                points.append(start_pos + [size*i/2*0.8, size, 0])
            # Middle horizontal  
            for i in range(3):
                points.append(start_pos + [size*i/2*0.6, size*0.5, 0])
            return np.array(points)
            
        elif letter == 'G':
            # G: C with horizontal bar
            points = []
            # Arc like C
            for i in range(10):
                angle = np.pi/6 + 4*np.pi/3 * i / 9
                x = start_pos[0] + size*0.5 + size*0.4*np.cos(angle)
                y = start_pos[1] + size*0.5 + size*0.4*np.sin(angle)
                points.append([x, y, start_pos[2]])
            # Horizontal bar
            for i in range(3):
                points.append(start_pos + [size*0.5 + size*0.3*i/2, size*0.3, 0])
            return np.array(points)
            
        elif letter == 'H':
            # H: Two verticals + crossbar
            points = []
            # Left vertical
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Right vertical
            for i in range(6):
                points.append(start_pos + [size*0.8, size*i/5, 0])
            # Crossbar
            for i in range(4):
                points.append(start_pos + [size*0.2*i, size*0.5, 0])
            return np.array(points)
            
        elif letter == 'I':
            # I: Vertical line with serifs
            points = []
            # Top serif
            for i in range(3):
                points.append(start_pos + [size*0.3*(i-1), size, 0])
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*(1-i/5), 0])
            # Bottom serif
            for i in range(3):
                points.append(start_pos + [size*0.3*(i-1), 0, 0])
            return np.array(points)
            
        elif letter == 'J':
            # J: Vertical with curve at bottom
            points = []
            # Vertical stroke
            for i in range(5):
                points.append(start_pos + [size*0.5, size*(1-i/5), 0])
            # Bottom curve
            for i in range(4):
                t = i / 3
                angle = -np.pi/2 * t
                x = start_pos[0] + size*(0.5 - 0.4*np.sin(angle))
                y = start_pos[1] + size*0.3*np.cos(angle)
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'K':
            # K: Vertical + two diagonals
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Upper diagonal
            for i in range(4):
                points.append(start_pos + [size*0.6*i/3, size*(0.5 + 0.5*i/3), 0])
            # Lower diagonal  
            for i in range(4):
                points.append(start_pos + [size*0.6*i/3, size*(0.5 - 0.5*i/3), 0])
            return np.array(points)
            
        elif letter == 'L':
            # L: Vertical + horizontal
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Bottom horizontal
            for i in range(4):
                points.append(start_pos + [size*0.7*i/3, 0, 0])
            return np.array(points)
            
        elif letter == 'M':
            # M: Two verticals + middle peak
            return np.array([
                start_pos,                                    # Bottom left
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.4, size*0.6, 0],       # Middle valley
                start_pos + [size*0.6, size, 0],           # Middle peak
                start_pos + [size, 0, 0]                   # Bottom right
            ])
            
        elif letter == 'N':
            # N: Two verticals + diagonal
            return np.array([
                start_pos,                                    # Bottom left
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.8, 0, 0],              # Bottom right
                start_pos + [size*0.8, size, 0]            # Top right
            ])
            
        elif letter == 'O':
            # O: Complete circle
            points = []
            for i in range(16):
                angle = 2 * np.pi * i / 15
                x = start_pos[0] + size*0.5 + size*0.4*np.cos(angle)
                y = start_pos[1] + size*0.5 + size*0.4*np.sin(angle)
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'P':
            # P: Vertical + top curve
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Top horizontal and curve
            for i in range(6):
                t = i / 5
                if t <= 0.5:
                    x = start_pos[0] + size*0.6*t*2
                    y = start_pos[1] + size
                else:
                    angle = np.pi - np.pi*(t-0.5)*2
                    x = start_pos[0] + size*0.6*(1 + 0.3*np.cos(angle))
                    y = start_pos[1] + size*(0.75 + 0.25*np.sin(angle))
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'Q':
            # Q: O with tail
            points = []
            # Circle like O
            for i in range(12):
                angle = 2 * np.pi * i / 11
                x = start_pos[0] + size*0.5 + size*0.4*np.cos(angle)
                y = start_pos[1] + size*0.5 + size*0.4*np.sin(angle)
                points.append([x, y, start_pos[2]])
            # Tail
            for i in range(3):
                points.append(start_pos + [size*(0.6 + 0.3*i/2), size*0.2*(1-i/2), 0])
            return np.array(points)
            
        elif letter == 'R':
            # R: Like P with diagonal leg
            points = []
            # Vertical stroke
            for i in range(6):
                points.append(start_pos + [0, size*i/5, 0])
            # Top curve like P
            for i in range(4):
                t = i / 3
                x = start_pos[0] + size*0.5*t
                y = start_pos[1] + size*(0.75 + 0.25*np.sin(np.pi*t))
                points.append([x, y, start_pos[2]])
            # Diagonal leg
            for i in range(4):
                points.append(start_pos + [size*(0.3 + 0.5*i/3), size*(0.5 - 0.5*i/3), 0])
            return np.array(points)
            
        elif letter == 'S':
            # S: Double curve
            points = []
            for i in range(12):
                t = i / 11
                # Create S-curve using sine
                x = start_pos[0] + size*0.5*(1 + 0.6*np.sin(2*np.pi*t))
                y = start_pos[1] + size*t
                points.append([x, y, start_pos[2]])
            return np.array(points)
            
        elif letter == 'T':
            # T: Horizontal top + vertical center
            points = []
            # Top horizontal
            for i in range(6):
                points.append(start_pos + [size*0.8*i/5, size, 0])
            # Vertical center
            for i in range(6):
                points.append(start_pos + [size*0.4, size*(1-i/5), 0])
            return np.array(points)
            
        elif letter == 'U':
            # U: Curve at bottom
            points = []
            # Left vertical
            for i in range(4):
                points.append(start_pos + [0, size*(1-i/4), 0])
            # Bottom curve
            for i in range(6):
                angle = np.pi + np.pi * i / 5
                x = start_pos[0] + size*0.4*(1 + np.cos(angle))
                y = start_pos[1] + size*0.3*(1 + np.sin(angle))
                points.append([x, y, start_pos[2]])
            # Right vertical
            for i in range(4):
                points.append(start_pos + [size*0.8, size*0.3 + size*0.7*i/3, 0])
            return np.array(points)
            
        elif letter == 'V':
            # V: Two diagonal strokes meeting at bottom
            return np.array([
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.5, 0, 0],              # Bottom center
                start_pos + [size, size, 0]                # Top right
            ])
            
        elif letter == 'W':
            # W: Like two V's
            return np.array([
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.25, 0, 0],             # Bottom left
                start_pos + [size*0.5, size*0.6, 0],       # Middle peak
                start_pos + [size*0.75, 0, 0],             # Bottom right  
                start_pos + [size, size, 0]                # Top right
            ])
            
        elif letter == 'X':
            # X: Two diagonal lines crossing
            return np.array([
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.8, 0, 0],              # Bottom right
                start_pos + [size*0.8, size, 0],           # Top right
                start_pos + [0, 0, 0]                      # Bottom left
            ])
            
        elif letter == 'Y':
            # Y: Two diagonals meeting + vertical
            return np.array([
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.5, size*0.5, 0],       # Center
                start_pos + [size, size, 0],               # Top right  
                start_pos + [size*0.5, size*0.5, 0],       # Back to center
                start_pos + [size*0.5, 0, 0]               # Bottom center
            ])
            
        elif letter == 'Z':
            # Z: Top horizontal + diagonal + bottom horizontal
            return np.array([
                start_pos + [0, size, 0],                   # Top left
                start_pos + [size*0.8, size, 0],           # Top right
                start_pos + [0, 0, 0],                     # Bottom left
                start_pos + [size*0.8, 0, 0]               # Bottom right
            ])
            
        else:
            # Unknown letter: Make a distinct pattern
            return np.array([
                start_pos,
                start_pos + [size*0.5, size*0.5, 0],
                start_pos + [size, 0, 0],
                start_pos + [size*0.5, size, 0],
                start_pos
            ])
    
    def train_on_expert_data(self, num_epochs: int = 50, validation_interval: int = 5, early_stopping_patience: int = 10):
        """
        COMPREHENSIVE GAIL training with validation checkpoints and early stopping.
        
        Features:
        - Proper GAIL adversarial training loop
        - Validation checkpoints every N epochs
        - Early stopping based on validation quality
        - Loss tracking and monitoring
        - Model saving at best validation score
        
        Args:
            num_epochs: Maximum training epochs
            validation_interval: Validate every N epochs
            early_stopping_patience: Stop if no improvement for N validations
        """
        logger.info(f"🚀 Starting COMPREHENSIVE GAIL training...")
        logger.info(f"   📊 Expert samples: {len(self.expert_buffer)}")
        logger.info(f"   🎯 Max epochs: {num_epochs}")
        logger.info(f"   ✅ Validation every {validation_interval} epochs")
        logger.info(f"   ⏹️  Early stopping patience: {early_stopping_patience}")
        
        if len(self.expert_buffer) < self.batch_size:
            logger.warning("Not enough expert data for training")
            return
        
        # Training state
        training_history = {
            'epoch': [],
            'discriminator_loss': [],
            'policy_loss': [],
            'validation_quality': [],
            'learning_rate': []
        }
        
        best_validation_score = -1.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            epoch_d_loss = 0.0
            epoch_p_loss = 0.0
            num_batches = len(self.expert_buffer) // self.batch_size
            
            # Train discriminator multiple times per policy update (standard GAIL practice)
            for batch_idx in range(num_batches):
                # Sample expert and policy data
                expert_batch = self.sample_expert_batch(self.batch_size)
                
                # Generate policy rollouts
                policy_obs, policy_actions = self._generate_policy_rollouts(self.batch_size)
                policy_batch = torch.cat([
                    torch.FloatTensor(policy_obs).to(self.policy.device),
                    torch.FloatTensor(policy_actions).to(self.policy.device)
                ], dim=1)
                
                # Update discriminator 3 times for each policy update
                for _ in range(3):
                    d_loss = self.update_discriminator_step(expert_batch, policy_batch)
                    epoch_d_loss += d_loss
                
                # Update policy once
                p_loss = self.update_policy_step(policy_obs, policy_actions)
                epoch_p_loss += p_loss
            
            # Calculate average losses
            avg_d_loss = epoch_d_loss / max(num_batches * 3, 1)  # 3 D updates per batch
            avg_p_loss = epoch_p_loss / max(num_batches, 1)
            
            # Record training metrics
            training_history['epoch'].append(epoch)
            training_history['discriminator_loss'].append(avg_d_loss)
            training_history['policy_loss'].append(avg_p_loss)
            training_history['learning_rate'].append(self.policy_optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start_time
            
            # Validation checkpoint
            if epoch % validation_interval == 0 or epoch == num_epochs - 1:
                logger.info(f"🧪 Running validation at epoch {epoch}...")
                
                # Run validation on subset of letters
                validation_letters = ['A', 'O', 'C', 'H']
                validation_results = self.validate_letter_quality(validation_letters)
                validation_score = validation_results['summary']['overall_quality']
                
                training_history['validation_quality'].append(validation_score)
                
                # Check for improvement
                if validation_score > best_validation_score:
                    best_validation_score = validation_score
                    patience_counter = 0
                    
                    # Save best model state
                    best_model_state = {
                        'policy_state_dict': self.policy.state_dict(),
                        'discriminator_state_dict': self.discriminator.state_dict(),
                        'epoch': epoch,
                        'validation_score': validation_score
                    }
                    
                    logger.info(f"🏆 NEW BEST MODEL: validation score {validation_score:.3f}")
                else:
                    patience_counter += 1
                    logger.info(f"⏳ No improvement: patience {patience_counter}/{early_stopping_patience}")
                
                # Log progress
                logger.info(f"📊 Epoch {epoch:3d}/{num_epochs}: "
                          f"D_loss={avg_d_loss:.4f}, P_loss={avg_p_loss:.4f}, "
                          f"Val_quality={validation_score:.3f}, Time={epoch_time:.1f}s")
                
                # Early stopping check
                if patience_counter >= early_stopping_patience:
                    logger.info(f"🛑 Early stopping triggered after {epoch+1} epochs")
                    logger.info(f"   Best validation score: {best_validation_score:.3f}")
                    break
            else:
                # Log training progress
                if epoch % 2 == 0:
                    logger.info(f"📈 Epoch {epoch:3d}: D_loss={avg_d_loss:.4f}, P_loss={avg_p_loss:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            logger.info(f"🔄 Restoring best model from epoch {best_model_state['epoch']}")
            self.policy.load_state_dict(best_model_state['policy_state_dict'])
            self.discriminator.load_state_dict(best_model_state['discriminator_state_dict'])
        
        # Save training history
        self.training_history = training_history
        
        logger.info("🎉 COMPREHENSIVE GAIL training completed!")
        logger.info(f"   🏆 Best validation score: {best_validation_score:.3f}")
        logger.info(f"   📊 Total epochs: {len(training_history['epoch'])}")
        
        return training_history
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics and statistics."""
        if not hasattr(self, 'training_history'):
            return {"error": "No training history available"}
        
        history = self.training_history
        
        if len(history['epoch']) == 0:
            return {"error": "No training data recorded"}
        
        metrics = {
            'total_epochs': len(history['epoch']),
            'final_discriminator_loss': history['discriminator_loss'][-1] if history['discriminator_loss'] else 0,
            'final_policy_loss': history['policy_loss'][-1] if history['policy_loss'] else 0,
            'best_validation_score': max(history['validation_quality']) if history['validation_quality'] else 0,
            'average_discriminator_loss': np.mean(history['discriminator_loss']) if history['discriminator_loss'] else 0,
            'average_policy_loss': np.mean(history['policy_loss']) if history['policy_loss'] else 0,
            'loss_convergence': self._calculate_loss_convergence(),
            'training_stability': self._calculate_training_stability()
        }
        
        return metrics
    
    def _calculate_loss_convergence(self) -> float:
        """Calculate how well the losses have converged (lower variance = better convergence)."""
        if not hasattr(self, 'training_history') or len(self.training_history['discriminator_loss']) < 10:
            return 0.0
        
        # Use last 10 epochs to assess convergence
        recent_d_losses = self.training_history['discriminator_loss'][-10:]
        recent_p_losses = self.training_history['policy_loss'][-10:]
        
        d_variance = np.var(recent_d_losses)
        p_variance = np.var(recent_p_losses)
        
        # Lower variance = better convergence (invert and normalize)
        convergence_score = 1.0 / (1.0 + d_variance + p_variance)
        return convergence_score
    
    def _calculate_training_stability(self) -> float:
        """Calculate training stability (consistent improvement over time)."""
        if not hasattr(self, 'training_history') or len(self.training_history['validation_quality']) < 3:
            return 0.0
        
        val_scores = self.training_history['validation_quality']
        
        # Calculate how often validation improved
        improvements = 0
        for i in range(1, len(val_scores)):
            if val_scores[i] >= val_scores[i-1]:
                improvements += 1
        
        stability = improvements / max(len(val_scores) - 1, 1)
        return stability
    
    def plot_training_progress(self, save_path: str = None):
        """
        Plot training progress with loss curves and validation scores.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not hasattr(self, 'training_history'):
            logger.warning("No training history available for plotting")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.training_history
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('GAIL Training Progress', fontsize=16)
            
            # Plot discriminator loss
            if history['discriminator_loss']:
                ax1.plot(history['epoch'], history['discriminator_loss'], 'b-', label='Discriminator Loss')
                ax1.set_title('Discriminator Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.grid(True)
            
            # Plot policy loss
            if history['policy_loss']:
                ax2.plot(history['epoch'], history['policy_loss'], 'r-', label='Policy Loss')
                ax2.set_title('Policy Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.grid(True)
            
            # Plot validation quality
            if history['validation_quality']:
                val_epochs = [history['epoch'][i] for i in range(0, len(history['epoch']), 5)][:len(history['validation_quality'])]
                ax3.plot(val_epochs, history['validation_quality'], 'g-o', label='Validation Quality')
                ax3.set_title('Validation Quality')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Quality Score')
                ax3.grid(True)
            
            # Plot learning rate
            if history['learning_rate']:
                ax4.plot(history['epoch'], history['learning_rate'], 'm-', label='Learning Rate')
                ax4.set_title('Learning Rate')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Learning Rate')
                ax4.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting training progress: {e}")
    
    def save_training_checkpoint(self, checkpoint_path: str):
        """Save complete training checkpoint including history and model states."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'training_history': getattr(self, 'training_history', {}),
            'expert_buffer': self.expert_buffer,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Training checkpoint saved to {checkpoint_path}")
    
    def load_training_checkpoint(self, checkpoint_path: str):
        """Load complete training checkpoint."""
        # Fix PyTorch 2.6 compatibility issue
        checkpoint = torch.load(checkpoint_path, map_location=self.policy.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.expert_buffer = checkpoint.get('expert_buffer', [])
        
        logger.info(f"Training checkpoint loaded from {checkpoint_path}")
    
    def _generate_policy_rollouts(self, num_samples: int):
        """Generate policy rollouts for training."""
        observations = []
        actions = []
        
        # Generate random letter contexts for policy rollouts
        letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        for _ in range(num_samples):
            letter = np.random.choice(list(letters))
            style_params = {
                'start_position': [0.1 + np.random.normal(0, 0.01), 
                                 0.15 + np.random.normal(0, 0.01), 
                                 0.02],
                'base_size': 0.03 + np.random.normal(0, 0.005),
                'speed': 1.0 + np.random.normal(0, 0.1)
            }
            
            context = self._create_letter_context(letter, style_params)
            
            # Sample action from current policy
            with torch.no_grad():
                action_mean, action_std = self.policy(context)
                # Add exploration noise
                action = action_mean + action_std * torch.randn_like(action_mean)
                
            observations.append(context.cpu().numpy()[0])
            actions.append(action.cpu().numpy()[0])
        
        return np.array(observations), np.array(actions)
    
    def train_with_adversarial_quality(self, num_epochs: int = 25, validation_interval: int = 5, 
                                     early_stopping_patience: int = 10, discriminator_training_ratio: int = 2,
                                     quality_loss_weight: float = 0.3):
        """
        ADVANCED ADVERSARIAL TRAINING for enhanced letter quality.
        
        Combines GAIL training with quality-based adversarial loss to improve
        letter recognition and formation quality.
        """
        logger.info(f"🚀 Starting adversarial quality training for {num_epochs} epochs...")
        
        if not self.expert_observations or not self.expert_actions:
            logger.warning("Not enough expert data for adversarial training")
            return
        
        # Training metrics
        training_metrics = {
            'policy_losses': [],
            'discriminator_losses': [],
            'quality_scores': [],
            'validation_scores': []
        }
        
        best_quality_score = -float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_policy_loss = 0.0
            epoch_discriminator_loss = 0.0
            epoch_quality_score = 0.0
            num_batches = 0
            
            # Create batches from expert data
            batch_size = self.config.get('batch_size', 64)
            num_expert_samples = len(self.expert_observations)
            
            # Training loop
            for batch_start in range(0, num_expert_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_expert_samples)
                
                # Expert batch
                expert_obs_batch = torch.FloatTensor(
                    self.expert_observations[batch_start:batch_end]
                ).to(self.policy.device)
                expert_actions_batch = torch.FloatTensor(
                    self.expert_actions[batch_start:batch_end]  
                ).to(self.policy.device)
                
                # Generate policy rollouts
                policy_obs, policy_actions = self.generate_policy_rollouts(batch_end - batch_start)
                policy_obs_batch = torch.FloatTensor(policy_obs).to(self.policy.device)
                policy_actions_batch = torch.FloatTensor(policy_actions).to(self.policy.device)
                
                # ENHANCED: Train discriminator with quality assessment
                for _ in range(discriminator_training_ratio):
                    d_metrics = self.update_discriminator_step(
                        torch.cat([expert_obs_batch, expert_actions_batch], dim=1),
                        torch.cat([policy_obs_batch, policy_actions_batch], dim=1)
                    )
                    epoch_discriminator_loss += d_metrics['d_loss']
                
                # ENHANCED: Train policy with adversarial quality loss
                policy_loss, quality_score = self.update_policy_with_quality(
                    policy_obs_batch, policy_actions_batch, quality_loss_weight
                )
                epoch_policy_loss += policy_loss
                epoch_quality_score += quality_score
                
                num_batches += 1
            
            # Average metrics for this epoch
            avg_policy_loss = epoch_policy_loss / num_batches if num_batches > 0 else 0
            avg_discriminator_loss = epoch_discriminator_loss / (num_batches * discriminator_training_ratio) if num_batches > 0 else 0
            avg_quality_score = epoch_quality_score / num_batches if num_batches > 0 else 0
            
            # Record training metrics
            training_metrics['policy_losses'].append(avg_policy_loss)
            training_metrics['discriminator_losses'].append(avg_discriminator_loss)
            training_metrics['quality_scores'].append(avg_quality_score)
            
            # Validation and early stopping
            if (epoch + 1) % validation_interval == 0:
                validation_score = self.validate_letter_quality(['A', 'O', 'H'])
                training_metrics['validation_scores'].append(validation_score['overall_quality'])
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}:")
                logger.info(f"  Policy Loss: {avg_policy_loss:.4f}")
                logger.info(f"  Discriminator Loss: {avg_discriminator_loss:.4f}")
                logger.info(f"  Quality Score: {avg_quality_score:.4f}")
                logger.info(f"  Validation Score: {validation_score['overall_quality']:.4f}")
                
                # Early stopping based on validation quality
                if validation_score['overall_quality'] > best_quality_score:
                    best_quality_score = validation_score['overall_quality']
                    patience_counter = 0
                    logger.info("  🎯 New best validation score!")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"  ⏹️  Early stopping after {epoch+1} epochs")
                        break
        
        logger.info("🏆 Adversarial quality training completed!")
        logger.info(f"   Best validation score: {best_quality_score:.4f}")
        
        return training_metrics
    
    def update_policy_with_quality(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor, 
                                 quality_loss_weight: float):
        """Enhanced policy update with quality-based adversarial loss."""
        self.policy.train()
        
        # Standard GAIL policy loss (fool the discriminator)
        combined_batch = torch.cat([obs_batch, actions_batch], dim=1)
        discriminator_pred = self.discriminator(combined_batch)
        
        # Policy wants discriminator to think its actions are expert (label=1)
        target_labels = torch.ones_like(discriminator_pred)
        adversarial_loss = F.binary_cross_entropy_with_logits(discriminator_pred, target_labels)
        
        # ENHANCED: Quality-based loss for better letter formation
        quality_loss = self.calculate_trajectory_quality_loss(obs_batch, actions_batch)
        
        # Combined loss
        total_loss = adversarial_loss + quality_loss_weight * quality_loss
        
        # Update policy
        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Calculate quality score for monitoring
        quality_score = 1.0 / (1.0 + quality_loss.item())  # Higher is better
        
        return total_loss.item(), quality_score
    
    def calculate_trajectory_quality_loss(self, obs_batch: torch.Tensor, actions_batch: torch.Tensor):
        """Calculate quality loss for better letter formation (no hardcoded shapes)."""
        # Extract movement actions (dx, dy)
        movements = actions_batch[:, :2]  # [batch_size, 2]
        
        # Quality metrics (all learnable, no hardcoded shapes)
        
        # 1. Smoothness loss: Penalize erratic movements
        movement_magnitudes = torch.norm(movements, dim=1)
        smoothness_variance = torch.var(movement_magnitudes)
        smoothness_loss = smoothness_variance
        
        # 2. Consistency loss: Similar letters should have similar movement patterns
        if movements.size(0) > 1:
            # Calculate pairwise movement consistency
            movement_diffs = torch.cdist(movements, movements, p=2)
            consistency_loss = torch.mean(movement_diffs)
        else:
            consistency_loss = torch.tensor(0.0, device=movements.device)
        
        # 3. Trajectory coherence: Movements should form connected paths
        # Penalize very small movements (might indicate stuck behavior)
        small_movement_penalty = torch.mean(torch.exp(-movement_magnitudes * 1000))
        
        # 4. Direction diversity: Letters should have varied directions
        if movements.size(0) > 2:
            movement_angles = torch.atan2(movements[:, 1], movements[:, 0])
            angle_variance = torch.var(movement_angles)
            direction_diversity = 1.0 / (1.0 + angle_variance)  # Inverse for loss
        else:
            direction_diversity = torch.tensor(0.0, device=movements.device)
        
        # Combine quality metrics
        total_quality_loss = (
            smoothness_loss * 0.3 +
            consistency_loss * 0.25 + 
            small_movement_penalty * 0.25 +
            direction_diversity * 0.2
        )
        
        return total_quality_loss
    
    def _process_neural_movement_enhanced(self, movement: np.ndarray, step: int, max_steps: int, 
                                        trajectory_points: list, style_params: Dict[str, Any]) -> np.ndarray:
        """Enhanced processing of neural network movement for better letter formation."""
        step_size = style_params.get('step_size', 0.001)
        
        # 1. ENHANCED: Magnitude optimization for letter coherence
        movement_magnitude = np.linalg.norm(movement[:2])
        
        if movement_magnitude < step_size * 0.08:
            # Boost very small movements for continuous writing flow
            direction = movement[:2] / (movement_magnitude + 1e-8)
            movement[:2] = direction * step_size * 0.4
        elif movement_magnitude > step_size * 3.5:
            # Limit excessive movements for letter coherence
            movement[:2] = movement[:2] / movement_magnitude * step_size * 3.0
        
        # 2. ENHANCED: Trajectory continuity and smoothing
        if len(trajectory_points) >= 2:
            # Calculate movement direction consistency
            prev_movement = trajectory_points[-1] - trajectory_points[-2]
            prev_direction = prev_movement[:2] / (np.linalg.norm(prev_movement[:2]) + 1e-8)
            
            current_direction = movement[:2] / (np.linalg.norm(movement[:2]) + 1e-8)
            
            # Apply gentle directional smoothing for letter-like curves
            smoothing_factor = min(0.3, step / max_steps * 0.5)  # Progressive smoothing
            smoothed_direction = (1 - smoothing_factor) * current_direction + smoothing_factor * prev_direction
            smoothed_direction /= (np.linalg.norm(smoothed_direction) + 1e-8)
            
            movement[:2] = smoothed_direction * np.linalg.norm(movement[:2])
        
        # 3. ENHANCED: Progressive letter formation scaling
        progress = step / max_steps
        
        if progress < 0.3:
            # Early phase: establish letter foundation
            movement *= 1.1
        elif progress < 0.7:
            # Middle phase: main letter body formation
            movement *= 1.0
        else:
            # Late phase: letter completion and refinement
            movement *= 0.85
        
        # 4. ENHANCED: Letter quality feedback loop
        if len(trajectory_points) >= 5:
            quality_score = self._assess_current_letter_quality(trajectory_points)
            
            # Adjust movement based on current letter quality
            if quality_score < 0.3:
                # Poor quality: encourage larger, more deliberate movements
                movement *= 1.2
            elif quality_score > 0.8:
                # Good quality: use more refined movements
                movement *= 0.9
        
        return movement
    
    def _assess_current_letter_quality(self, trajectory_points: list) -> float:
        """Assess current trajectory quality for real-time feedback."""
        if len(trajectory_points) < 3:
            return 0.0
        
        points = np.array(trajectory_points)
        
        # 1. Spatial distribution (letter should cover reasonable area)
        x_span = points[:, 0].max() - points[:, 0].min()
        y_span = points[:, 1].max() - points[:, 1].min()
        spatial_score = min(1.0, (x_span + y_span) / 0.035)  # Normalized to expected letter size
        
        # 2. Movement consistency (avoid erratic jumps)
        if len(points) >= 3:
            movements = np.diff(points, axis=0)
            movement_magnitudes = np.linalg.norm(movements[:, :2], axis=1)
            consistency_score = 1.0 / (1.0 + np.std(movement_magnitudes))
        else:
            consistency_score = 1.0
        
        # 3. Trajectory smoothness
        if len(points) >= 4:
            second_derivatives = np.diff(points, n=2, axis=0)
            smoothness_score = 1.0 / (1.0 + np.mean(np.linalg.norm(second_derivatives, axis=1)))
        else:
            smoothness_score = 1.0
        
        # 4. Letter-like characteristics (complexity and structure)
        if len(points) >= 5:
            # Check for reasonable complexity (not too simple, not too chaotic)
            total_distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
            direct_distance = np.linalg.norm(points[-1] - points[0])
            complexity_ratio = total_distance / max(direct_distance, 0.001)
            complexity_score = min(1.0, max(0.0, (complexity_ratio - 1.0) / 3.0))  # Good complexity 1-4 ratio
        else:
            complexity_score = 0.5
        
        # Combine scores with weights
        overall_quality = (
            spatial_score * 0.3 +
            consistency_score * 0.25 +
            smoothness_score * 0.25 +
            complexity_score * 0.2
        )
        
        return max(0.0, min(1.0, overall_quality))
    
    def _validate_character_formation(self, trajectory_points: list, target_letter: str) -> dict:
        """Validate character formation against expected letter characteristics."""
        if len(trajectory_points) < 3:
            return {'is_valid': False, 'confidence': 0.0, 'issues': ['insufficient_points']}
        
        points = np.array(trajectory_points)
        validation_result = {
            'is_valid': True,
            'confidence': 0.0,
            'issues': [],
            'metrics': {}
        }
        
        # 1. Size validation
        x_span = points[:, 0].max() - points[:, 0].min()
        y_span = points[:, 1].max() - points[:, 1].min()
        size_score = self._validate_letter_size(x_span, y_span, target_letter)
        validation_result['metrics']['size_score'] = size_score
        
        if size_score < 0.3:
            validation_result['issues'].append('inadequate_size')
        
        # 2. Shape characteristics validation
        shape_score = self._validate_letter_shape_characteristics(points, target_letter)
        validation_result['metrics']['shape_score'] = shape_score
        
        if shape_score < 0.4:
            validation_result['issues'].append('poor_shape_characteristics')
        
        # 3. Stroke pattern validation
        stroke_score = self._validate_stroke_patterns(points, target_letter)
        validation_result['metrics']['stroke_score'] = stroke_score
        
        if stroke_score < 0.35:
            validation_result['issues'].append('incorrect_stroke_pattern')
        
        # 4. Letter-specific validation
        letter_specific_score = self._validate_letter_specific_features(points, target_letter)
        validation_result['metrics']['letter_specific_score'] = letter_specific_score
        
        if letter_specific_score < 0.3:
            validation_result['issues'].append('missing_letter_features')
        
        # Calculate overall confidence
        confidence_weights = [0.25, 0.3, 0.25, 0.2]  # Weights for each score
        scores = [size_score, shape_score, stroke_score, letter_specific_score]
        overall_confidence = sum(w * s for w, s in zip(confidence_weights, scores))
        
        validation_result['confidence'] = max(0.0, min(1.0, overall_confidence))
        validation_result['is_valid'] = overall_confidence > 0.5 and len(validation_result['issues']) == 0
        
        return validation_result
    
    def _validate_letter_size(self, x_span: float, y_span: float, target_letter: str) -> float:
        """Validate letter size against expected dimensions."""
        # Expected letter size ranges
        expected_size = 0.025  # Base expected size
        min_size = expected_size * 0.4  # 40% of expected
        max_size = expected_size * 2.5  # 250% of expected
        
        # Letter-specific size adjustments
        wide_letters = {'M', 'W'}
        narrow_letters = {'I', 'L'}
        
        if target_letter in wide_letters:
            expected_width = expected_size * 1.3
        elif target_letter in narrow_letters:
            expected_width = expected_size * 0.6
        else:
            expected_width = expected_size
        
        # Size validation
        max_dimension = max(x_span, y_span)
        
        if max_dimension < min_size:
            return 0.1  # Too small
        elif max_dimension > max_size:
            return 0.2  # Too large
        else:
            # Calculate how close to expected size
            size_ratio = min(max_dimension / expected_size, expected_size / max_dimension)
            return max(0.3, min(1.0, size_ratio))
    
    def _validate_letter_shape_characteristics(self, points: np.ndarray, target_letter: str) -> float:
        """Validate shape characteristics for the target letter."""
        # Define letter categories
        curved_letters = {'C', 'G', 'O', 'Q', 'S', 'U'}
        angular_letters = {'A', 'E', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'T', 'V', 'W', 'X', 'Y', 'Z'}
        closed_letters = {'A', 'B', 'D', 'O', 'P', 'Q', 'R'}
        
        score = 0.5  # Base score
        
        # 1. Curvature analysis
        curvature_score = self._analyze_trajectory_curvature(points)
        
        if target_letter in curved_letters:
            # Curved letters should have high curvature
            score += 0.3 * min(1.0, curvature_score / 0.7)
        elif target_letter in angular_letters:
            # Angular letters should have low curvature
            score += 0.3 * min(1.0, (1.0 - curvature_score) / 0.7)
        
        # 2. Closure analysis for closed letters
        if target_letter in closed_letters:
            closure_score = self._analyze_trajectory_closure(points)
            score += 0.2 * closure_score
        
        return max(0.0, min(1.0, score))
    
    def _validate_stroke_patterns(self, points: np.ndarray, target_letter: str) -> float:
        """Validate stroke patterns for the target letter."""
        # Analyze movement patterns
        movements = np.diff(points, axis=0)
        movement_magnitudes = np.linalg.norm(movements[:, :2], axis=1)
        
        # Detect potential stroke breaks (very small movements)
        stroke_breaks = np.where(movement_magnitudes < 0.0005)[0]
        estimated_strokes = len(stroke_breaks) + 1
        
        # Expected stroke counts for letters
        multi_stroke_letters = {
            'A': 2, 'B': 2, 'D': 2, 'H': 3, 'P': 2, 'R': 2,
            'F': 2, 'E': 2, 'T': 2
        }
        
        expected_strokes = multi_stroke_letters.get(target_letter, 1)
        
        # Score based on stroke count match
        if estimated_strokes == expected_strokes:
            return 0.8
        elif abs(estimated_strokes - expected_strokes) == 1:
            return 0.6
        else:
            return 0.3
    
    def _validate_letter_specific_features(self, points: np.ndarray, target_letter: str) -> float:
        """Validate letter-specific features."""
        x_span = points[:, 0].max() - points[:, 0].min()
        y_span = points[:, 1].max() - points[:, 1].min()
        
        score = 0.5  # Base score
        
        # Aspect ratio validation
        if x_span > 0 and y_span > 0:
            aspect_ratio = x_span / y_span
            
            # Letter-specific aspect ratio expectations
            if target_letter in {'I', 'L'}:
                # Tall, narrow letters
                expected_ratio = 0.3
                ratio_score = 1.0 / (1.0 + abs(aspect_ratio - expected_ratio))
                score += 0.3 * ratio_score
            elif target_letter in {'M', 'W'}:
                # Wide letters
                expected_ratio = 1.5
                ratio_score = 1.0 / (1.0 + abs(aspect_ratio - expected_ratio))
                score += 0.3 * ratio_score
            elif target_letter in {'O', 'Q'}:
                # Roughly square letters
                expected_ratio = 1.0
                ratio_score = 1.0 / (1.0 + abs(aspect_ratio - expected_ratio))
                score += 0.3 * ratio_score
        
        # Symmetry validation for symmetric letters
        symmetric_letters = {'A', 'H', 'I', 'M', 'O', 'T', 'U', 'V', 'W', 'X', 'Y'}
        if target_letter in symmetric_letters:
            symmetry_score = self._analyze_trajectory_symmetry(points)
            score += 0.2 * symmetry_score
        
        return max(0.0, min(1.0, score))
    
    def _analyze_trajectory_curvature(self, points: np.ndarray) -> float:
        """Analyze the curvature of the trajectory."""
        if len(points) < 3:
            return 0.0
        
        # Calculate curvature using second derivatives
        first_derivatives = np.diff(points, axis=0)
        second_derivatives = np.diff(first_derivatives, axis=0)
        
        # Curvature magnitude
        curvature_magnitudes = np.linalg.norm(second_derivatives, axis=1)
        average_curvature = np.mean(curvature_magnitudes)
        
        # Normalize to 0-1 range
        return min(1.0, average_curvature * 1000)  # Scale factor for typical handwriting
    
    def _analyze_trajectory_closure(self, points: np.ndarray) -> float:
        """Analyze how well the trajectory closes (for closed letters)."""
        if len(points) < 4:
            return 0.0
        
        # Distance between start and end points
        start_end_distance = np.linalg.norm(points[-1] - points[0])
        
        # Total trajectory length
        total_distance = np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        
        # Closure score (lower start-end distance relative to total distance is better)
        if total_distance > 0:
            closure_ratio = start_end_distance / total_distance
            return max(0.0, 1.0 - closure_ratio * 5)  # Scale factor
        
        return 0.0
    
    def _analyze_trajectory_symmetry(self, points: np.ndarray) -> float:
        """Analyze the symmetry of the trajectory."""
        if len(points) < 4:
            return 0.0
        
        # Find the center of the trajectory
        center_x = (points[:, 0].min() + points[:, 0].max()) / 2
        center_y = (points[:, 1].min() + points[:, 1].max()) / 2
        
        # Analyze horizontal symmetry
        left_points = points[points[:, 0] < center_x]
        right_points = points[points[:, 0] > center_x]
        
        if len(left_points) > 0 and len(right_points) > 0:
            # Mirror right points and calculate distance to left points
            mirrored_right = right_points.copy()
            mirrored_right[:, 0] = 2 * center_x - mirrored_right[:, 0]
            
            # Simple symmetry measure (this is a basic approximation)
            if len(left_points) == len(mirrored_right):
                symmetry_distances = np.linalg.norm(left_points - mirrored_right, axis=1)
                average_asymmetry = np.mean(symmetry_distances)
                symmetry_score = max(0.0, 1.0 - average_asymmetry * 100)  # Scale factor
                return symmetry_score
        
        return 0.5  # Default moderate symmetry score
    
    def update_discriminator_step(self, expert_batch: torch.Tensor, policy_batch: torch.Tensor):
        """Single discriminator update step."""
        self.discriminator.train()
        
        # Expert data should be labeled as 1 (real)
        expert_labels = torch.ones(expert_batch.size(0), 1).to(self.discriminator.device)
        expert_pred = self.discriminator(expert_batch)
        expert_loss = F.binary_cross_entropy_with_logits(expert_pred, expert_labels)
        
        # Policy data should be labeled as 0 (fake)
        policy_labels = torch.zeros(policy_batch.size(0), 1).to(self.discriminator.device)
        policy_pred = self.discriminator(policy_batch)
        policy_loss = F.binary_cross_entropy_with_logits(policy_pred, policy_labels)
        
        # Total discriminator loss
        d_loss = expert_loss + policy_loss
        
        # Update discriminator
        self.discriminator_optimizer.zero_grad()
        d_loss.backward()
        self.discriminator_optimizer.step()
        
        return d_loss.item()
    
    def update_policy_step(self, observations: np.ndarray, actions: np.ndarray):
        """Single policy update step."""
        self.policy.train()
        
        obs_tensor = torch.FloatTensor(observations).to(self.policy.device)
        action_tensor = torch.FloatTensor(actions).to(self.policy.device)
        
        # Get policy predictions
        action_mean, action_std = self.policy(obs_tensor)
        
        # Calculate log probabilities
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(action_tensor).sum(dim=1)
        
        # Get discriminator rewards (higher for more realistic)
        state_action = torch.cat([obs_tensor, action_tensor], dim=1)
        with torch.no_grad():
            rewards = torch.sigmoid(self.discriminator(state_action)).squeeze()
        
        # Policy gradient loss (maximize reward)
        policy_loss = -(log_probs * rewards).mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
    
    def add_expert_data(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """
        Add expert demonstration data to buffer.
        
        Args:
            observations: Expert observations [num_steps, obs_dim]
            actions: Expert actions [num_steps, action_dim]
        """
        for obs, action in zip(observations, actions):
            state_action = np.concatenate([obs, action])
            self.expert_buffer.append(state_action)
        
        logger.info(f"Added {len(observations)} expert demonstrations to buffer")
    
    def add_policy_data(self, observations: np.ndarray, actions: np.ndarray) -> None:
        """
        Add policy-generated data to buffer.
        
        Args:
            observations: Policy observations [num_steps, obs_dim] 
            actions: Policy actions [num_steps, action_dim]
        """
        for obs, action in zip(observations, actions):
            state_action = np.concatenate([obs, action])
            self.policy_buffer.append(state_action)
        
        self.total_steps += len(observations)
    
    def sample_expert_batch(self, batch_size: int) -> torch.Tensor:
        """Sample batch from expert buffer."""
        if len(self.expert_buffer) < batch_size:
            batch_size = len(self.expert_buffer)
        
        batch = random.sample(self.expert_buffer, batch_size)
        return torch.FloatTensor(batch).to(self.policy.device)
    
    def sample_policy_batch(self, batch_size: int) -> torch.Tensor:
        """Sample batch from policy buffer."""
        if len(self.policy_buffer) < batch_size:
            batch_size = len(self.policy_buffer)
        
        batch = random.sample(self.policy_buffer, batch_size)
        return torch.FloatTensor(batch).to(self.policy.device)
    
    def update_discriminator(self, num_updates: int = 1) -> float:
        """
        Update discriminator network.
        
        Args:
            num_updates: Number of update steps
            
        Returns:
            avg_loss: Average discriminator loss
        """
        if len(self.expert_buffer) < self.batch_size or len(self.policy_buffer) < self.batch_size:
            return 0.0
        
        total_loss = 0.0
        
        for _ in range(num_updates):
            # Sample batches
            expert_batch = self.sample_expert_batch(self.batch_size)
            policy_batch = self.sample_policy_batch(self.batch_size)
            
            # Update discriminator
            self.discriminator_optimizer.zero_grad()
            loss = self.discriminator.compute_loss(expert_batch, policy_batch)
            loss.backward()
            self.discriminator_optimizer.step()
            
            total_loss += loss.item()
            self.discriminator_updates += 1
        
        return total_loss / num_updates
    
    def get_rewards(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Get rewards from discriminator for policy training.
        
        Args:
            observations: Policy observations
            actions: Policy actions
            
        Returns:
            rewards: Estimated rewards
        """
        state_actions = []
        for obs, action in zip(observations, actions):
            state_actions.append(np.concatenate([obs, action]))
        
        state_action_tensor = torch.FloatTensor(state_actions).to(self.policy.device)
        rewards = self.discriminator.predict_reward(state_action_tensor)
        
        return rewards.cpu().numpy()
    
    def update_policy(self, trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """
        Update policy using PPO with discriminator rewards.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            training_stats: Dictionary of training statistics
        """
        # Combine all trajectory data
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        
        for traj in trajectories:
            obs = traj['observations']
            actions = traj['actions']
            
            # Get rewards from discriminator
            rewards = self.get_rewards(obs, actions)
            dones = traj.get('dones', np.zeros(len(obs), dtype=bool))
            
            all_obs.extend(obs)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_dones.extend(dones)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(all_obs).to(self.policy.device)
        actions_tensor = torch.FloatTensor(all_actions).to(self.policy.device)
        rewards_tensor = torch.FloatTensor(all_rewards).to(self.policy.device)
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards_tensor, all_dones)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old policy predictions
        with torch.no_grad():
            old_log_probs, _ = self.policy.evaluate_action(obs_tensor, actions_tensor)
        
        # PPO updates
        policy_losses = []
        entropy_losses = []
        
        for _ in range(self.config.get('policy_epochs', 4)):
            # Get current policy predictions
            log_probs, entropy = self.policy.evaluate_action(obs_tensor, actions_tensor)
            
            # Compute policy loss
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy loss
            entropy_loss = -entropy.mean()
            
            # Total loss
            total_loss = policy_loss + self.entropy_coef * entropy_loss
            
            # Update policy
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.policy_optimizer.step()
            
            policy_losses.append(policy_loss.item())
            entropy_losses.append(entropy_loss.item())
        
        self.policy_updates += 1
        
        return {
            'policy_loss': np.mean(policy_losses),
            'entropy_loss': np.mean(entropy_losses),
            'mean_reward': rewards_tensor.mean().item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def _compute_gae(self, rewards: torch.Tensor, dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        running_advantage = 0
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_advantage = 0
                running_return = 0
            
            running_return = rewards[t] + self.gamma * running_return
            running_advantage = rewards[t] + self.gamma * self.lam * running_advantage
            
            returns[t] = running_return
            advantages[t] = running_advantage
        
        return advantages, returns
    
    def save_models(self, filepath_prefix: str) -> None:
        """Save both policy and discriminator models."""
        self.policy.save_checkpoint(f"{filepath_prefix}_policy.pth")
        self.discriminator.save_checkpoint(f"{filepath_prefix}_discriminator.pth")
        
        logger.info(f"Saved GAIL models with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix: str) -> None:
        """Load both policy and discriminator models."""
        self.policy.load_checkpoint(f"{filepath_prefix}_policy.pth")
        self.discriminator.load_checkpoint(f"{filepath_prefix}_discriminator.pth")
        
        logger.info(f"Loaded GAIL models with prefix: {filepath_prefix}")
    
    def get_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from policy for given observation.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
        """
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.policy.device)
        
        with torch.no_grad():
            action, _ = self.policy.sample_action(obs_tensor, deterministic)
        
        return action.cpu().numpy()[0]
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'total_steps': self.total_steps,
            'policy_updates': self.policy_updates,
            'discriminator_updates': self.discriminator_updates,
            'expert_buffer_size': len(self.expert_buffer),
            'policy_buffer_size': len(self.policy_buffer)
        }