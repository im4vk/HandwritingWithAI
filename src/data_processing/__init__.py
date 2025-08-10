"""
Data Processing Package
======================

Utilities for processing handwriting datasets, feature extraction,
data augmentation, and format conversion.

Classes:
--------
- DatasetLoader: Loading various handwriting dataset formats
- DataPreprocessor: Data preprocessing and normalization utilities  
- FeatureExtractor: Feature extraction from handwriting data
- DataAugmenter: Data augmentation techniques for handwriting
- FormatConverter: Converting between different data formats
- HandwritingDataset: PyTorch dataset class for handwriting data
- DataUtils: General utilities for data processing
"""

from .dataset_loader import DatasetLoader, HandwritingDataset, DatasetMetadata
from .preprocessing import HandwritingPreprocessor
from .feature_extraction import FeatureExtractor, HandwritingFeatures, TemporalFeatures
from .data_augmentation import DataAugmenter, AugmentationMethod, AugmentationParams
from .format_converters import FormatConverter, SupportedFormat, ConversionOptions
from .utils import DataUtils, TrajectoryStats, DataValidation

__all__ = [
    'DatasetLoader',
    'HandwritingDataset',
    'DatasetMetadata',
    'HandwritingPreprocessor',
    'FeatureExtractor',
    'HandwritingFeatures',
    'TemporalFeatures',
    'DataAugmenter',
    'AugmentationMethod',
    'AugmentationParams',
    'FormatConverter',
    'SupportedFormat',
    'ConversionOptions',
    'DataUtils',
    'TrajectoryStats',
    'DataValidation'
]