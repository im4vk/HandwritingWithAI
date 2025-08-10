"""
Format Converters for Handwriting Data
======================================

Convert between different handwriting data formats.
"""

import numpy as np
import pandas as pd
import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

from .dataset_loader import HandwritingSample

logger = logging.getLogger(__name__)


class SupportedFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    NPZ = "npz"
    HDF5 = "hdf5"
    INKML = "inkml"


@dataclass
class ConversionOptions:
    """Options for format conversion."""
    include_pressure: bool = True
    include_timestamps: bool = True
    include_pen_states: bool = True
    coordinate_precision: int = 6
    time_precision: int = 3
    compress_output: bool = True


class FormatConverter:
    """Convert between different handwriting data formats."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize format converter."""
        self.config = config
        self.default_options = ConversionOptions(**config.get('conversion_options', {}))
        
        logger.info("Initialized FormatConverter")
    
    def convert_samples(self, samples: List[HandwritingSample],
                       output_format: SupportedFormat,
                       output_path: str,
                       options: Optional[ConversionOptions] = None) -> None:
        """
        Convert samples to specified format.
        
        Args:
            samples: Input samples
            output_format: Target format
            output_path: Output file path
            options: Conversion options
        """
        if options is None:
            options = self.default_options
        
        if output_format == SupportedFormat.JSON:
            self._convert_to_json(samples, output_path, options)
        elif output_format == SupportedFormat.CSV:
            self._convert_to_csv(samples, output_path, options)
        elif output_format == SupportedFormat.XML:
            self._convert_to_xml(samples, output_path, options)
        elif output_format == SupportedFormat.NPZ:
            self._convert_to_npz(samples, output_path, options)
        elif output_format == SupportedFormat.HDF5:
            self._convert_to_hdf5(samples, output_path, options)
        elif output_format == SupportedFormat.INKML:
            self._convert_to_inkml(samples, output_path, options)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Converted {len(samples)} samples to {output_format.value} format")
    
    def _convert_to_json(self, samples: List[HandwritingSample], 
                        output_path: str, options: ConversionOptions) -> None:
        """Convert to JSON format."""
        json_data = []
        
        for sample in samples:
            sample_data = {
                'trajectory': sample.trajectory.round(options.coordinate_precision).tolist(),
                'text': sample.text,
                'writer_id': sample.writer_id
            }
            
            if options.include_timestamps and sample.timestamps is not None:
                sample_data['timestamps'] = sample.timestamps.round(options.time_precision).tolist()
            
            if options.include_pressure and sample.pressure is not None:
                sample_data['pressure'] = sample.pressure.round(options.coordinate_precision).tolist()
            
            if options.include_pen_states and sample.pen_states is not None:
                sample_data['pen_states'] = sample.pen_states.tolist()
            
            if sample.character_labels:
                sample_data['character_labels'] = sample.character_labels
            
            if sample.metadata:
                sample_data['metadata'] = sample.metadata
            
            json_data.append(sample_data)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2 if not options.compress_output else None)
    
    def _convert_to_csv(self, samples: List[HandwritingSample],
                       output_path: str, options: ConversionOptions) -> None:
        """Convert to CSV format."""
        rows = []
        
        for sample_idx, sample in enumerate(samples):
            trajectory = sample.trajectory
            n_points = len(trajectory)
            
            for point_idx in range(n_points):
                row = {
                    'sample_id': sample_idx,
                    'point_id': point_idx,
                    'x': round(trajectory[point_idx, 0], options.coordinate_precision),
                    'y': round(trajectory[point_idx, 1], options.coordinate_precision),
                    'writer_id': sample.writer_id,
                    'text': sample.text
                }
                
                if options.include_timestamps and sample.timestamps is not None:
                    row['time'] = round(sample.timestamps[point_idx], options.time_precision)
                
                if options.include_pressure and sample.pressure is not None:
                    row['pressure'] = round(sample.pressure[point_idx], options.coordinate_precision)
                
                if options.include_pen_states and sample.pen_states is not None:
                    row['pen_state'] = sample.pen_states[point_idx]
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def _convert_to_xml(self, samples: List[HandwritingSample],
                       output_path: str, options: ConversionOptions) -> None:
        """Convert to XML format."""
        root = ET.Element('HandwritingData')
        
        for sample_idx, sample in enumerate(samples):
            sample_elem = ET.SubElement(root, 'Sample')
            sample_elem.set('id', str(sample_idx))
            sample_elem.set('writer_id', sample.writer_id)
            sample_elem.set('text', sample.text)
            
            # Add trajectory
            trajectory_elem = ET.SubElement(sample_elem, 'Trajectory')
            
            for point_idx, point in enumerate(sample.trajectory):
                point_elem = ET.SubElement(trajectory_elem, 'Point')
                point_elem.set('id', str(point_idx))
                point_elem.set('x', str(round(point[0], options.coordinate_precision)))
                point_elem.set('y', str(round(point[1], options.coordinate_precision)))
                
                if options.include_timestamps and sample.timestamps is not None:
                    point_elem.set('time', str(round(sample.timestamps[point_idx], options.time_precision)))
                
                if options.include_pressure and sample.pressure is not None:
                    point_elem.set('pressure', str(round(sample.pressure[point_idx], options.coordinate_precision)))
                
                if options.include_pen_states and sample.pen_states is not None:
                    point_elem.set('pen_state', str(sample.pen_states[point_idx]))
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def _convert_to_npz(self, samples: List[HandwritingSample],
                       output_path: str, options: ConversionOptions) -> None:
        """Convert to NumPy compressed format."""
        data = {
            'trajectories': [sample.trajectory for sample in samples],
            'texts': [sample.text for sample in samples],
            'writer_ids': [sample.writer_id for sample in samples]
        }
        
        if options.include_timestamps:
            timestamps = []
            for sample in samples:
                if sample.timestamps is not None:
                    timestamps.append(sample.timestamps)
                else:
                    timestamps.append(np.arange(len(sample.trajectory), dtype=float))
            data['timestamps'] = timestamps
        
        if options.include_pressure:
            pressures = []
            for sample in samples:
                if sample.pressure is not None:
                    pressures.append(sample.pressure)
                else:
                    pressures.append(np.ones(len(sample.trajectory)))
            data['pressures'] = pressures
        
        if options.include_pen_states:
            pen_states = []
            for sample in samples:
                if sample.pen_states is not None:
                    pen_states.append(sample.pen_states)
                else:
                    pen_states.append(np.ones(len(sample.trajectory)))
            data['pen_states'] = pen_states
        
        if options.compress_output:
            np.savez_compressed(output_path, **data)
        else:
            np.savez(output_path, **data)
    
    def _convert_to_hdf5(self, samples: List[HandwritingSample],
                        output_path: str, options: ConversionOptions) -> None:
        """Convert to HDF5 format."""
        import h5py
        
        with h5py.File(output_path, 'w') as f:
            for sample_idx, sample in enumerate(samples):
                group = f.create_group(f'sample_{sample_idx}')
                
                # Store trajectory
                group.create_dataset('trajectory', data=sample.trajectory,
                                   compression='gzip' if options.compress_output else None)
                
                # Store metadata as attributes
                group.attrs['text'] = sample.text.encode('utf-8')
                group.attrs['writer_id'] = sample.writer_id.encode('utf-8')
                
                # Store optional data
                if options.include_timestamps and sample.timestamps is not None:
                    group.create_dataset('timestamps', data=sample.timestamps,
                                       compression='gzip' if options.compress_output else None)
                
                if options.include_pressure and sample.pressure is not None:
                    group.create_dataset('pressure', data=sample.pressure,
                                       compression='gzip' if options.compress_output else None)
                
                if options.include_pen_states and sample.pen_states is not None:
                    group.create_dataset('pen_states', data=sample.pen_states,
                                       compression='gzip' if options.compress_output else None)
    
    def _convert_to_inkml(self, samples: List[HandwritingSample],
                         output_path: str, options: ConversionOptions) -> None:
        """Convert to InkML format."""
        # Create InkML document
        root = ET.Element('ink')
        root.set('xmlns', 'http://www.w3.org/2003/InkML')
        
        for sample_idx, sample in enumerate(samples):
            # Create trace group
            trace_group = ET.SubElement(root, 'traceGroup')
            trace_group.set('xml:id', f'sample_{sample_idx}')
            
            # Add annotation for text
            annotation = ET.SubElement(trace_group, 'annotation')
            annotation.set('type', 'truth')
            annotation.text = sample.text
            
            # Add writer annotation
            writer_annotation = ET.SubElement(trace_group, 'annotation')
            writer_annotation.set('type', 'writer')
            writer_annotation.text = sample.writer_id
            
            # Create trace for trajectory
            trace = ET.SubElement(trace_group, 'trace')
            
            # Build trace content
            trace_points = []
            for point_idx, point in enumerate(sample.trajectory):
                x = round(point[0], options.coordinate_precision)
                y = round(point[1], options.coordinate_precision)
                
                if options.include_timestamps and sample.timestamps is not None:
                    t = round(sample.timestamps[point_idx], options.time_precision)
                    trace_points.append(f'{x} {y} {t}')
                else:
                    trace_points.append(f'{x} {y}')
            
            trace.text = ', '.join(trace_points)
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    def batch_convert(self, input_dir: str, output_dir: str,
                     input_format: SupportedFormat,
                     output_format: SupportedFormat,
                     options: Optional[ConversionOptions] = None) -> None:
        """
        Batch convert files from one format to another.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            input_format: Source format
            output_format: Target format
            options: Conversion options
        """
        from .dataset_loader import DatasetLoader
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load samples using dataset loader
        loader = DatasetLoader({})
        
        # Convert each file
        converted_count = 0
        for file_path in input_path.iterdir():
            if file_path.is_file():
                try:
                    # Determine dataset type from input format
                    if input_format == SupportedFormat.JSON:
                        dataset_type = 'custom'
                    elif input_format == SupportedFormat.CSV:
                        dataset_type = 'custom'
                    else:
                        dataset_type = 'custom'
                    
                    # Load samples
                    samples = loader.load_dataset(file_path, dataset_type)
                    
                    if samples:
                        # Generate output filename
                        output_filename = file_path.stem + f'.{output_format.value}'
                        output_file_path = output_path / output_filename
                        
                        # Convert
                        self.convert_samples(samples, output_format, str(output_file_path), options)
                        converted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error converting {file_path}: {e}")
        
        logger.info(f"Converted {converted_count} files from {input_format.value} to {output_format.value}")
    
    def validate_conversion(self, original_samples: List[HandwritingSample],
                          converted_file: str,
                          output_format: SupportedFormat) -> Dict[str, Any]:
        """
        Validate conversion by comparing original and converted data.
        
        Args:
            original_samples: Original samples
            converted_file: Path to converted file
            output_format: Format of converted file
            
        Returns:
            validation_results: Validation metrics
        """
        # Load converted samples back
        from .dataset_loader import DatasetLoader
        loader = DatasetLoader({})
        
        try:
            converted_samples = loader.load_dataset(converted_file, 'custom')
        except Exception as e:
            return {'valid': False, 'error': str(e)}
        
        # Compare samples
        results = {
            'valid': True,
            'num_original': len(original_samples),
            'num_converted': len(converted_samples),
            'trajectory_errors': [],
            'text_errors': [],
            'metadata_errors': []
        }
        
        if len(original_samples) != len(converted_samples):
            results['valid'] = False
            results['count_mismatch'] = True
            return results
        
        # Compare individual samples
        for i, (orig, conv) in enumerate(zip(original_samples, converted_samples)):
            # Compare trajectories
            if not np.allclose(orig.trajectory, conv.trajectory, atol=1e-3):
                results['trajectory_errors'].append(i)
            
            # Compare text
            if orig.text != conv.text:
                results['text_errors'].append(i)
        
        # Overall validation
        results['trajectory_accuracy'] = 1.0 - len(results['trajectory_errors']) / len(original_samples)
        results['text_accuracy'] = 1.0 - len(results['text_errors']) / len(original_samples)
        results['valid'] = (results['trajectory_accuracy'] > 0.95 and 
                           results['text_accuracy'] > 0.95)
        
        return results