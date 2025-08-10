"""
Dataset Loader for Handwriting Data
===================================

Utilities for loading various handwriting dataset formats including
IAM-DB, IRONOFF, UJIpenchars, and custom formats.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional, Union, Iterator
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import xml.etree.ElementTree as ET
import h5py
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """
    Metadata for handwriting datasets.
    
    Attributes:
        name: Dataset name
        version: Dataset version
        description: Dataset description
        num_samples: Total number of samples
        num_writers: Number of writers
        languages: List of languages
        characters: List of characters/symbols
        sampling_rate: Sampling rate in Hz
        data_format: Data format description
        splits: Available data splits
        features: Available features per sample
    """
    name: str
    version: str
    description: str
    num_samples: int
    num_writers: int
    languages: List[str]
    characters: List[str]
    sampling_rate: float
    data_format: str
    splits: List[str]
    features: List[str]


@dataclass 
class HandwritingSample:
    """
    Single handwriting sample.
    
    Attributes:
        trajectory: Trajectory points [n_points, features]
        text: Ground truth text
        writer_id: Writer identifier
        character_labels: Character-level labels
        timestamps: Time stamps
        pressure: Pen pressure values
        pen_states: Pen up/down states
        metadata: Additional sample metadata
    """
    trajectory: np.ndarray
    text: str
    writer_id: str
    character_labels: Optional[List[str]] = None
    timestamps: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    pen_states: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetLoader:
    """
    Loader for various handwriting datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset loader.
        
        Args:
            config: Loader configuration
        """
        self.config = config
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_dir = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset-specific configurations
        self.dataset_configs = {
            'iam_db': self._get_iam_config(),
            'ironoff': self._get_ironoff_config(),
            'ujipenchars': self._get_ujipenchars_config(),
            'custom': self._get_custom_config()
        }
        
        logger.info("Initialized DatasetLoader")
    
    def _get_iam_config(self) -> Dict[str, Any]:
        """Get IAM-DB dataset configuration."""
        return {
            'data_format': 'xml',
            'features': ['x', 'y', 'time', 'pen_state'],
            'sampling_rate': 200.0,  # Hz
            'coordinate_scale': 1.0,
            'time_scale': 1.0
        }
    
    def _get_ironoff_config(self) -> Dict[str, Any]:
        """Get IRONOFF dataset configuration."""
        return {
            'data_format': 'txt',
            'features': ['x', 'y', 'time', 'pen_state', 'azimuth', 'altitude', 'pressure'],
            'sampling_rate': 200.0,
            'coordinate_scale': 0.01,  # Convert to meters
            'time_scale': 0.01  # Convert to seconds
        }
    
    def _get_ujipenchars_config(self) -> Dict[str, Any]:
        """Get UJIpenchars dataset configuration."""
        return {
            'data_format': 'txt',
            'features': ['x', 'y', 'time', 'pen_state'],
            'sampling_rate': 100.0,
            'coordinate_scale': 0.001,
            'time_scale': 0.001
        }
    
    def _get_custom_config(self) -> Dict[str, Any]:
        """Get custom dataset configuration."""
        return {
            'data_format': 'flexible',
            'features': ['x', 'y'],
            'sampling_rate': 100.0,
            'coordinate_scale': 1.0,
            'time_scale': 1.0
        }
    
    def load_dataset(self, dataset_path: Union[str, Path], 
                    dataset_type: str = 'custom',
                    split: Optional[str] = None) -> List[HandwritingSample]:
        """
        Load handwriting dataset.
        
        Args:
            dataset_path: Path to dataset directory or file
            dataset_type: Type of dataset ('iam_db', 'ironoff', 'ujipenchars', 'custom')
            split: Dataset split to load ('train', 'val', 'test')
            
        Returns:
            samples: List of handwriting samples
        """
        dataset_path = Path(dataset_path)
        
        # Check cache
        cache_key = f"{dataset_type}_{dataset_path.name}_{split}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if self.cache_enabled and cache_file.exists():
            logger.info(f"Loading cached dataset: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Load based on dataset type
        if dataset_type == 'iam_db':
            samples = self._load_iam_db(dataset_path, split)
        elif dataset_type == 'ironoff':
            samples = self._load_ironoff(dataset_path, split)
        elif dataset_type == 'ujipenchars':
            samples = self._load_ujipenchars(dataset_path, split)
        elif dataset_type == 'custom':
            samples = self._load_custom_dataset(dataset_path, split)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Cache loaded dataset
        if self.cache_enabled:
            with open(cache_file, 'wb') as f:
                pickle.dump(samples, f)
            logger.info(f"Cached dataset: {cache_file}")
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_type}")
        return samples
    
    def _load_iam_db(self, dataset_path: Path, split: Optional[str] = None) -> List[HandwritingSample]:
        """Load IAM-DB dataset."""
        samples = []
        config = self.dataset_configs['iam_db']
        
        # Find XML files
        if split:
            xml_files = list(dataset_path.glob(f"*{split}*.xml"))
        else:
            xml_files = list(dataset_path.glob("*.xml"))
        
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                
                # Extract handwriting data
                for stroke_set in root.findall('.//StrokeSet'):
                    text = stroke_set.get('text', '')
                    writer_id = stroke_set.get('writer', 'unknown')
                    
                    # Collect all strokes
                    trajectory_points = []
                    timestamps = []
                    pen_states = []
                    
                    for stroke in stroke_set.findall('Stroke'):
                        for point in stroke.findall('Point'):
                            x = float(point.get('x', 0)) * config['coordinate_scale']
                            y = float(point.get('y', 0)) * config['coordinate_scale']
                            t = float(point.get('time', 0)) * config['time_scale']
                            
                            trajectory_points.append([x, y])
                            timestamps.append(t)
                            pen_states.append(1.0)  # Pen down during stroke
                        
                        # Add pen lift between strokes
                        if len(trajectory_points) > 0:
                            trajectory_points.append(trajectory_points[-1])
                            timestamps.append(timestamps[-1] + 0.01)
                            pen_states.append(0.0)  # Pen up
                    
                    if trajectory_points:
                        sample = HandwritingSample(
                            trajectory=np.array(trajectory_points),
                            text=text,
                            writer_id=writer_id,
                            timestamps=np.array(timestamps),
                            pen_states=np.array(pen_states),
                            metadata={'source_file': xml_file.name, 'dataset': 'iam_db'}
                        )
                        samples.append(sample)
            
            except Exception as e:
                logger.warning(f"Error loading {xml_file}: {e}")
        
        return samples
    
    def _load_ironoff(self, dataset_path: Path, split: Optional[str] = None) -> List[HandwritingSample]:
        """Load IRONOFF dataset."""
        samples = []
        config = self.dataset_configs['ironoff']
        
        # Find text files
        if split:
            txt_files = list(dataset_path.glob(f"*{split}*.txt"))
        else:
            txt_files = list(dataset_path.glob("*.txt"))
        
        for txt_file in txt_files:
            try:
                # Load data
                data = np.loadtxt(txt_file, skiprows=1)  # Skip header
                
                if len(data) == 0:
                    continue
                
                # Extract features
                x = data[:, 0] * config['coordinate_scale']
                y = data[:, 1] * config['coordinate_scale']
                time = data[:, 2] * config['time_scale']
                pen_state = data[:, 3]
                
                # Additional features if available
                pressure = data[:, 6] if data.shape[1] > 6 else np.ones(len(data))
                
                trajectory = np.column_stack([x, y])
                
                # Extract text from filename or metadata
                text = self._extract_text_from_filename(txt_file.stem)
                writer_id = self._extract_writer_id(txt_file.stem)
                
                sample = HandwritingSample(
                    trajectory=trajectory,
                    text=text,
                    writer_id=writer_id,
                    timestamps=time,
                    pressure=pressure,
                    pen_states=pen_state,
                    metadata={'source_file': txt_file.name, 'dataset': 'ironoff'}
                )
                samples.append(sample)
            
            except Exception as e:
                logger.warning(f"Error loading {txt_file}: {e}")
        
        return samples
    
    def _load_ujipenchars(self, dataset_path: Path, split: Optional[str] = None) -> List[HandwritingSample]:
        """Load UJIpenchars dataset."""
        samples = []
        config = self.dataset_configs['ujipenchars']
        
        # Find data files
        if split:
            data_files = list(dataset_path.glob(f"*{split}*"))
        else:
            data_files = list(dataset_path.glob("*.txt"))
        
        for data_file in data_files:
            try:
                # Load trajectory data
                data = np.loadtxt(data_file)
                
                if len(data) == 0:
                    continue
                
                # Extract features
                x = data[:, 0] * config['coordinate_scale']
                y = data[:, 1] * config['coordinate_scale']
                time = data[:, 2] * config['time_scale'] if data.shape[1] > 2 else np.arange(len(data)) / config['sampling_rate']
                pen_state = data[:, 3] if data.shape[1] > 3 else np.ones(len(data))
                
                trajectory = np.column_stack([x, y])
                
                # Extract character label from filename
                character = self._extract_character_from_filename(data_file.stem)
                writer_id = self._extract_writer_id(data_file.stem)
                
                sample = HandwritingSample(
                    trajectory=trajectory,
                    text=character,
                    writer_id=writer_id,
                    timestamps=time,
                    pen_states=pen_state,
                    metadata={'source_file': data_file.name, 'dataset': 'ujipenchars'}
                )
                samples.append(sample)
            
            except Exception as e:
                logger.warning(f"Error loading {data_file}: {e}")
        
        return samples
    
    def _load_custom_dataset(self, dataset_path: Path, split: Optional[str] = None) -> List[HandwritingSample]:
        """Load custom dataset format."""
        samples = []
        
        # Try different file formats
        if dataset_path.is_file():
            # Single file
            samples.extend(self._load_single_file(dataset_path))
        else:
            # Directory with multiple files
            for file_path in dataset_path.iterdir():
                if file_path.is_file():
                    try:
                        samples.extend(self._load_single_file(file_path))
                    except Exception as e:
                        logger.warning(f"Error loading {file_path}: {e}")
        
        return samples
    
    def _load_single_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load single file in various formats."""
        samples = []
        
        if file_path.suffix == '.json':
            samples.extend(self._load_json_file(file_path))
        elif file_path.suffix == '.csv':
            samples.extend(self._load_csv_file(file_path))
        elif file_path.suffix == '.h5' or file_path.suffix == '.hdf5':
            samples.extend(self._load_hdf5_file(file_path))
        elif file_path.suffix == '.npz':
            samples.extend(self._load_npz_file(file_path))
        elif file_path.suffix == '.txt':
            samples.extend(self._load_text_file(file_path))
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
        
        return samples
    
    def _load_json_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load JSON format handwriting data."""
        samples = []
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # List of samples
            for item in data:
                sample = self._parse_json_sample(item, file_path.name)
                if sample:
                    samples.append(sample)
        elif isinstance(data, dict):
            # Single sample or samples in dictionary
            if 'samples' in data:
                for item in data['samples']:
                    sample = self._parse_json_sample(item, file_path.name)
                    if sample:
                        samples.append(sample)
            else:
                sample = self._parse_json_sample(data, file_path.name)
                if sample:
                    samples.append(sample)
        
        return samples
    
    def _parse_json_sample(self, data: Dict[str, Any], source_file: str) -> Optional[HandwritingSample]:
        """Parse single JSON sample."""
        try:
            # Required fields
            if 'trajectory' not in data:
                return None
            
            trajectory = np.array(data['trajectory'])
            text = data.get('text', '')
            writer_id = data.get('writer_id', 'unknown')
            
            # Optional fields
            timestamps = np.array(data['timestamps']) if 'timestamps' in data else None
            pressure = np.array(data['pressure']) if 'pressure' in data else None
            pen_states = np.array(data['pen_states']) if 'pen_states' in data else None
            character_labels = data.get('character_labels')
            
            metadata = data.get('metadata', {})
            metadata['source_file'] = source_file
            metadata['dataset'] = 'custom'
            
            return HandwritingSample(
                trajectory=trajectory,
                text=text,
                writer_id=writer_id,
                character_labels=character_labels,
                timestamps=timestamps,
                pressure=pressure,
                pen_states=pen_states,
                metadata=metadata
            )
        
        except Exception as e:
            logger.warning(f"Error parsing JSON sample: {e}")
            return None
    
    def _load_csv_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load CSV format handwriting data."""
        samples = []
        
        try:
            df = pd.read_csv(file_path)
            
            # Group by sample_id or writer_id
            if 'sample_id' in df.columns:
                grouped = df.groupby('sample_id')
            elif 'writer_id' in df.columns:
                grouped = df.groupby('writer_id')
            else:
                # Treat entire file as single sample
                grouped = [('single', df)]
            
            for group_id, group_df in grouped:
                # Extract trajectory
                if 'x' in group_df.columns and 'y' in group_df.columns:
                    trajectory = group_df[['x', 'y']].values
                else:
                    continue
                
                # Extract other fields
                text = group_df['text'].iloc[0] if 'text' in group_df.columns else ''
                writer_id = str(group_id)
                
                timestamps = group_df['time'].values if 'time' in group_df.columns else None
                pressure = group_df['pressure'].values if 'pressure' in group_df.columns else None
                pen_states = group_df['pen_state'].values if 'pen_state' in group_df.columns else None
                
                sample = HandwritingSample(
                    trajectory=trajectory,
                    text=text,
                    writer_id=writer_id,
                    timestamps=timestamps,
                    pressure=pressure,
                    pen_states=pen_states,
                    metadata={'source_file': file_path.name, 'dataset': 'custom'}
                )
                samples.append(sample)
        
        except Exception as e:
            logger.warning(f"Error loading CSV file {file_path}: {e}")
        
        return samples
    
    def _load_hdf5_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load HDF5 format handwriting data."""
        samples = []
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Iterate through groups (samples)
                for sample_id in f.keys():
                    group = f[sample_id]
                    
                    # Extract trajectory
                    if 'trajectory' in group:
                        trajectory = group['trajectory'][:]
                    else:
                        continue
                    
                    # Extract metadata
                    text = group.attrs.get('text', '').decode() if 'text' in group.attrs else ''
                    writer_id = group.attrs.get('writer_id', '').decode() if 'writer_id' in group.attrs else sample_id
                    
                    # Extract optional data
                    timestamps = group['timestamps'][:] if 'timestamps' in group else None
                    pressure = group['pressure'][:] if 'pressure' in group else None
                    pen_states = group['pen_states'][:] if 'pen_states' in group else None
                    
                    sample = HandwritingSample(
                        trajectory=trajectory,
                        text=text,
                        writer_id=writer_id,
                        timestamps=timestamps,
                        pressure=pressure,
                        pen_states=pen_states,
                        metadata={'source_file': file_path.name, 'dataset': 'custom'}
                    )
                    samples.append(sample)
        
        except Exception as e:
            logger.warning(f"Error loading HDF5 file {file_path}: {e}")
        
        return samples
    
    def _load_npz_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load NumPy compressed format."""
        samples = []
        
        try:
            data = np.load(file_path, allow_pickle=True)
            
            # Check if it's a single sample or multiple samples
            if 'trajectories' in data:
                # Multiple samples
                trajectories = data['trajectories']
                texts = data.get('texts', [''] * len(trajectories))
                writer_ids = data.get('writer_ids', [f'writer_{i}' for i in range(len(trajectories))])
                
                for i, traj in enumerate(trajectories):
                    sample = HandwritingSample(
                        trajectory=traj,
                        text=texts[i] if i < len(texts) else '',
                        writer_id=writer_ids[i] if i < len(writer_ids) else f'writer_{i}',
                        metadata={'source_file': file_path.name, 'dataset': 'custom'}
                    )
                    samples.append(sample)
            
            elif 'trajectory' in data:
                # Single sample
                sample = HandwritingSample(
                    trajectory=data['trajectory'],
                    text=str(data.get('text', '')),
                    writer_id=str(data.get('writer_id', 'unknown')),
                    timestamps=data.get('timestamps'),
                    pressure=data.get('pressure'),
                    pen_states=data.get('pen_states'),
                    metadata={'source_file': file_path.name, 'dataset': 'custom'}
                )
                samples.append(sample)
        
        except Exception as e:
            logger.warning(f"Error loading NPZ file {file_path}: {e}")
        
        return samples
    
    def _load_text_file(self, file_path: Path) -> List[HandwritingSample]:
        """Load plain text format."""
        samples = []
        
        try:
            data = np.loadtxt(file_path)
            
            if len(data.shape) == 2 and data.shape[1] >= 2:
                # Assume first two columns are x, y coordinates
                trajectory = data[:, :2]
                
                # Extract additional features if available
                timestamps = data[:, 2] if data.shape[1] > 2 else None
                pen_states = data[:, 3] if data.shape[1] > 3 else None
                pressure = data[:, 4] if data.shape[1] > 4 else None
                
                # Extract metadata from filename
                text = self._extract_text_from_filename(file_path.stem)
                writer_id = self._extract_writer_id(file_path.stem)
                
                sample = HandwritingSample(
                    trajectory=trajectory,
                    text=text,
                    writer_id=writer_id,
                    timestamps=timestamps,
                    pressure=pressure,
                    pen_states=pen_states,
                    metadata={'source_file': file_path.name, 'dataset': 'custom'}
                )
                samples.append(sample)
        
        except Exception as e:
            logger.warning(f"Error loading text file {file_path}: {e}")
        
        return samples
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """Extract text content from filename."""
        # Common patterns for extracting text from filenames
        patterns = [
            r'_text_(.+?)_',
            r'_(.+?)_writer',
            r'text_(.+?)$',
            r'^(.+?)_'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1).replace('_', ' ')
        
        # Return filename if no pattern matches
        return filename.replace('_', ' ')
    
    def _extract_writer_id(self, filename: str) -> str:
        """Extract writer ID from filename."""
        import re
        
        # Look for writer ID patterns
        patterns = [
            r'writer_(\d+)',
            r'w(\d+)',
            r'user_(\d+)',
            r'subject_(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return f"writer_{match.group(1)}"
        
        return "unknown"
    
    def _extract_character_from_filename(self, filename: str) -> str:
        """Extract character from filename."""
        import re
        
        # Look for single character patterns
        patterns = [
            r'char_(.)',
            r'_(.)[_.]',
            r'^(.)[_.]'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)
        
        # Return first character of filename
        return filename[0] if filename else 'X'
    
    def get_dataset_metadata(self, dataset_path: Union[str, Path], 
                           dataset_type: str = 'custom') -> DatasetMetadata:
        """
        Get metadata for dataset.
        
        Args:
            dataset_path: Path to dataset
            dataset_type: Type of dataset
            
        Returns:
            metadata: Dataset metadata
        """
        # Load a sample of the dataset to analyze
        samples = self.load_dataset(dataset_path, dataset_type)
        
        if not samples:
            return DatasetMetadata(
                name=dataset_type,
                version="unknown",
                description="Empty dataset",
                num_samples=0,
                num_writers=0,
                languages=["unknown"],
                characters=[],
                sampling_rate=0.0,
                data_format="unknown",
                splits=[],
                features=[]
            )
        
        # Analyze samples
        writers = set(sample.writer_id for sample in samples)
        characters = set()
        for sample in samples:
            if sample.text:
                characters.update(sample.text)
        
        # Estimate sampling rate
        sampling_rates = []
        for sample in samples:
            if sample.timestamps is not None and len(sample.timestamps) > 1:
                dt = np.mean(np.diff(sample.timestamps))
                if dt > 0:
                    sampling_rates.append(1.0 / dt)
        
        avg_sampling_rate = np.mean(sampling_rates) if sampling_rates else 100.0
        
        # Determine features
        features = ['x', 'y']
        if any(sample.timestamps is not None for sample in samples):
            features.append('time')
        if any(sample.pressure is not None for sample in samples):
            features.append('pressure')
        if any(sample.pen_states is not None for sample in samples):
            features.append('pen_state')
        
        return DatasetMetadata(
            name=dataset_type,
            version="1.0",
            description=f"Handwriting dataset with {len(samples)} samples",
            num_samples=len(samples),
            num_writers=len(writers),
            languages=["unknown"],  # Would need language detection
            characters=sorted(list(characters)),
            sampling_rate=avg_sampling_rate,
            data_format=self.dataset_configs[dataset_type]['data_format'],
            splits=["all"],
            features=features
        )


class HandwritingDataset(Dataset):
    """
    PyTorch dataset for handwriting data.
    """
    
    def __init__(self, samples: List[HandwritingSample], 
                 transform: Optional[callable] = None,
                 target_transform: Optional[callable] = None):
        """
        Initialize handwriting dataset.
        
        Args:
            samples: List of handwriting samples
            transform: Transform function for input data
            target_transform: Transform function for targets
        """
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get dataset item.
        
        Args:
            idx: Sample index
            
        Returns:
            trajectory: Trajectory tensor
            text: Text label
        """
        sample = self.samples[idx]
        
        # Convert trajectory to tensor
        trajectory = torch.FloatTensor(sample.trajectory)
        
        # Apply transforms
        if self.transform:
            trajectory = self.transform(trajectory)
        
        text = sample.text
        if self.target_transform:
            text = self.target_transform(text)
        
        return trajectory, text
    
    def get_sample(self, idx: int) -> HandwritingSample:
        """Get full sample with all metadata."""
        return self.samples[idx]
    
    def filter_by_writer(self, writer_ids: List[str]) -> 'HandwritingDataset':
        """Filter dataset by writer IDs."""
        filtered_samples = [s for s in self.samples if s.writer_id in writer_ids]
        return HandwritingDataset(filtered_samples, self.transform, self.target_transform)
    
    def filter_by_text_length(self, min_length: int = 1, max_length: int = 100) -> 'HandwritingDataset':
        """Filter dataset by text length."""
        filtered_samples = [
            s for s in self.samples 
            if min_length <= len(s.text) <= max_length
        ]
        return HandwritingDataset(filtered_samples, self.transform, self.target_transform)
    
    def split_by_writer(self, train_ratio: float = 0.8) -> Tuple['HandwritingDataset', 'HandwritingDataset']:
        """Split dataset by writers."""
        writers = list(set(s.writer_id for s in self.samples))
        np.random.shuffle(writers)
        
        n_train_writers = int(len(writers) * train_ratio)
        train_writers = set(writers[:n_train_writers])
        
        train_samples = [s for s in self.samples if s.writer_id in train_writers]
        val_samples = [s for s in self.samples if s.writer_id not in train_writers]
        
        return (
            HandwritingDataset(train_samples, self.transform, self.target_transform),
            HandwritingDataset(val_samples, self.transform, self.target_transform)
        )