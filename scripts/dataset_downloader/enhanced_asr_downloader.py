#!/usr/bin/env python3
"""
Enhanced ASR Dataset Downloader for Hugging Face Datasets

This script downloads and organizes Common Voice, FLEURS, and AI-Shell datasets
from Hugging Face for multiple languages with advanced caching and resume capabilities.
It extracts transcription information and organizes data by language for ASR training
with proper train/validation/test splits.

Supported datasets:
- Common Voice: Mozilla's multilingual speech corpus with demographic metadata
- FLEURS: Google's Few-shot Learning Evaluation of Universal Representations of Speech
- AI-Shell: SpeechColab's Chinese Mandarin speech recognition corpus

Supported languages: Korean (ko), English (en), Chinese (zh), French (fr), Spanish (es)

Features:
- Smart caching to avoid re-downloading existing data
- Multiple output formats (JSON, TSV, TXT, JSONL manifests)
- Comprehensive statistics generation for each language
- ASR framework compatibility (NeMo, ESPnet, etc.)
- Configurable data storage location (default: ~/work/data)

Author: Enhanced ASR Dataset Processing Tool
"""

import os
import sys
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
import time
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict
import librosa
import soundfile as sf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetCache:
    """Manages dataset download cache and metadata for efficient data management"""
    
    def __init__(self, cache_dir: str):
        """Initialize cache manager with specified directory
        
        Args:
            cache_dir: Directory path for storing cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata from file
        
        Returns:
            Dictionary containing cache metadata or empty dict if file doesn't exist
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata to file for persistence"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def get_cache_key(self, dataset_name: str, language: str, config: str = None) -> str:
        """Generate unique cache key for dataset/language combination
        
        Args:
            dataset_name: Name of the dataset (e.g., 'common_voice', 'fleurs')
            language: Language code (e.g., 'ko', 'en')
            config: Optional dataset configuration
            
        Returns:
            Unique string key for caching
        """
        key_parts = [dataset_name, language]
        if config:
            key_parts.append(config)
        return "_".join(key_parts)
    
    def is_cached(self, dataset_name: str, language: str, config: str = None) -> bool:
        """Check if dataset/language combination is already cached
        
        Args:
            dataset_name: Name of the dataset
            language: Language code
            config: Optional dataset configuration
            
        Returns:
            True if data is cached, False otherwise
        """
        cache_key = self.get_cache_key(dataset_name, language, config)
        return cache_key in self.metadata
    
    def mark_cached(self, dataset_name: str, language: str, config: str, 
                   samples_count: int, duration_hours: float):
        """Mark dataset/language combination as cached with metadata
        
        Args:
            dataset_name: Name of the dataset
            language: Language code
            config: Dataset configuration used
            samples_count: Number of samples processed
            duration_hours: Total duration in hours
        """
        cache_key = self.get_cache_key(dataset_name, language, config)
        self.metadata[cache_key] = {
            'dataset': dataset_name,
            'language': language,
            'config': config,
            'cached_at': time.time(),
            'samples_count': samples_count,
            'duration_hours': duration_hours
        }
        self._save_metadata()
    
    def get_cached_info(self, dataset_name: str, language: str, config: str = None) -> Dict:
        """Get cached dataset information if available
        
        Args:
            dataset_name: Name of the dataset
            language: Language code
            config: Optional dataset configuration
            
        Returns:
            Dictionary with cache information or empty dict if not cached
        """
        cache_key = self.get_cache_key(dataset_name, language, config)
        return self.metadata.get(cache_key, {})

class EnhancedASRDownloader:
    """Enhanced ASR Dataset Downloader with caching and multi-language support
    
    This class provides comprehensive functionality for downloading, processing, and organizing
    speech datasets from Hugging Face for ASR training. It supports multiple datasets and
    languages with intelligent caching and resume capabilities.
    """
    
    # Language mapping for different dataset naming conventions
    LANGUAGE_MAPPING = {
        'common_voice': {
            'ko': 'ko', 'en': 'en', 'zh': 'zh-CN', 
            'fr': 'fr', 'es': 'es'
        },
        'fleurs': {
            'ko': 'ko_kr', 'en': 'en_us', 'zh': 'cmn_hans_cn',
            'fr': 'fr_fr', 'es': 'es_419'
        },
        'aishell': {
            'zh': 'zh'  # AI-Shell is Chinese only
        }
    }
    
    # Default configuration for each dataset
    DATASET_CONFIGS = {
        'common_voice': {
            'dataset_id': 'mozilla-foundation/common_voice_17_0',
            'audio_column': 'audio',
            'text_column': 'sentence',
            'splits': ['train', 'validation', 'test']
        },
        'fleurs': {
            'dataset_id': 'google/fleurs',
            'audio_column': 'audio',
            'text_column': 'transcription',
            'splits': ['train', 'validation', 'test']
        },
        'aishell': {
            'dataset_id': 'speechcolab/aishell_1',
            'audio_column': 'audio',
            'text_column': 'transcription',
            'splits': ['train', 'validation', 'test']
        }
    }
    
    def __init__(self, data_dir: str = "~/work/data", cache_dir: str = None, 
                 hf_token: str = None):
        """Initialize the enhanced ASR dataset downloader
        
        Args:
            data_dir: Directory for storing processed datasets (default: ~/work/data)
            cache_dir: Directory for Hugging Face cache (default: data_dir/hf_cache)
            hf_token: Hugging Face authentication token for private datasets
        """
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir is None:
            cache_dir = str(self.data_dir / "hf_cache")
        self.cache_dir = Path(cache_dir)
        
        self.hf_token = hf_token
        self.dataset_cache = DatasetCache(cache_dir)
        
        # Initialize statistics storage
        self.all_statistics = {}
        self.language_summaries = {}
        
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def check_existing_data(self, dataset_name: str, language: str) -> Optional[Path]:
        """Check if processed data already exists for dataset/language"""
        output_dir = self.data_dir / "asr_datasets" / f"{dataset_name}_{language}"
        
        if output_dir.exists():
            # Check for key files to ensure data is complete
            lang_dir = output_dir / language
            required_files = ['train.json', 'statistics.json']
            
            if lang_dir.exists() and all((lang_dir / f).exists() for f in required_files):
                logger.info(f"Found existing processed data for {dataset_name}/{language} at {output_dir}")
                return output_dir
        
        return None
    
    def load_existing_statistics(self, output_dir: Path, language: str) -> Dict:
        """Load statistics from existing processed data"""
        stats_file = output_dir / language / "statistics.json"
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def download_dataset(self, dataset_name: str, language: str, 
                        force_download: bool = False) -> Optional[DatasetDict]:
        """Download dataset with caching support"""
        
        if dataset_name not in self.DATASET_CONFIGS:
            logger.error(f"Unsupported dataset: {dataset_name}")
            return None
        
        # Check for existing processed data first
        existing_dir = self.check_existing_data(dataset_name, language)
        if existing_dir and not force_download:
            # Load existing statistics
            stats = self.load_existing_statistics(existing_dir, language)
            if stats:
                self.all_statistics[f"{dataset_name}_{language}"] = stats
                logger.info(f"Loaded existing statistics for {dataset_name}/{language}")
            return None  # Signal that data already exists
        
        config = self.DATASET_CONFIGS[dataset_name]
        dataset_id = config['dataset_id']
        
        # Get language code for this dataset
        lang_mapping = self.LANGUAGE_MAPPING.get(dataset_name, {})
        dataset_lang = lang_mapping.get(language)
        
        if not dataset_lang:
            logger.warning(f"Language {language} not supported for {dataset_name}")
            return None
        
        # Check cache
        if self.dataset_cache.is_cached(dataset_name, language, dataset_lang) and not force_download:
            logger.info(f"Dataset {dataset_name}/{language} found in cache, skipping download")
            return None
        
        try:
            logger.info(f"Downloading {dataset_name} dataset for {language} ({dataset_lang})")
            
            # Download parameters
            download_params = {
                'cache_dir': str(self.cache_dir),
                'trust_remote_code': True
            }
            
            if self.hf_token:
                download_params['token'] = self.hf_token
            
            # Dataset-specific download logic
            if dataset_name == 'common_voice':
                dataset = load_dataset(dataset_id, dataset_lang, **download_params)
            elif dataset_name == 'fleurs':
                dataset = load_dataset(dataset_id, dataset_lang, **download_params)
            elif dataset_name == 'aishell':
                dataset = load_dataset(dataset_id, **download_params)
            else:
                logger.error(f"Unknown dataset: {dataset_name}")
                return None
            
            logger.info(f"Successfully downloaded {dataset_name} for {language}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download {dataset_name} for {language}: {str(e)}")
            return None
    
    def process_audio_sample(self, sample: Dict, audio_column: str, text_column: str) -> Optional[Dict]:
        """Process a single audio sample"""
        try:
            audio_data = sample[audio_column]
            text = sample.get(text_column, '').strip()
            
            if not text:
                return None
            
            # Extract audio information
            audio_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            # Calculate duration
            duration = len(audio_array) / sampling_rate
            
            # Create processed sample
            processed_sample = {
                'audio_filepath': '',  # Will be set during export
                'text': text,
                'duration': duration,
                'sampling_rate': sampling_rate,
                'num_samples': len(audio_array)
            }
            
            # Add metadata if available
            for key in ['speaker_id', 'gender', 'age', 'accent', 'locale', 'segment_id']:
                if key in sample:
                    processed_sample[key] = sample[key]
            
            return processed_sample
            
        except Exception as e:
            logger.warning(f"Failed to process audio sample: {str(e)}")
            return None
    
    def process_dataset(self, dataset: DatasetDict, dataset_name: str, language: str) -> Dict[str, List[Dict]]:
        """Process dataset and organize by splits"""
        if not dataset:
            return {}
        
        config = self.DATASET_CONFIGS[dataset_name]
        audio_column = config['audio_column']
        text_column = config['text_column']
        
        processed_data = {}
        
        for split_name in dataset.keys():
            logger.info(f"Processing {dataset_name} {language}/{split_name}...")
            
            split_data = dataset[split_name]
            processed_samples = []
            
            # Process samples with progress bar
            for sample in tqdm(split_data, desc=f"Processing {split_name}"):
                processed_sample = self.process_audio_sample(sample, audio_column, text_column)
                if processed_sample:
                    processed_samples.append(processed_sample)
            
            processed_data[split_name] = processed_samples
            logger.info(f"Processed {len(processed_samples)} samples from {dataset_name} {language}/{split_name}")
        
        return processed_data
    
    def organize_for_asr(self, processed_data: Dict[str, List[Dict]], language: str) -> Dict[str, List[Dict]]:
        """Organize data into train/dev/test splits for ASR training"""
        # Standard split mapping
        split_mapping = {
            'train': 'train',
            'validation': 'dev',
            'dev': 'dev',
            'test': 'test'
        }
        
        organized_data = {}
        
        for original_split, samples in processed_data.items():
            asr_split = split_mapping.get(original_split, original_split)
            
            if asr_split not in organized_data:
                organized_data[asr_split] = []
            
            organized_data[asr_split].extend(samples)
        
        # Ensure we have all required splits
        required_splits = ['train', 'dev', 'test']
        for split in required_splits:
            if split not in organized_data:
                organized_data[split] = []
        
        return organized_data
    
    def calculate_statistics(self, organized_data: Dict[str, List[Dict]], 
                           dataset_name: str, language: str) -> Dict:
        """Calculate comprehensive statistics for the dataset"""
        
        all_samples = []
        for samples in organized_data.values():
            all_samples.extend(samples)
        
        if not all_samples:
            return {}
        
        # Basic statistics
        total_samples = len(all_samples)
        total_duration = sum(sample['duration'] for sample in all_samples)
        
        # Text statistics
        texts = [sample['text'] for sample in all_samples]
        text_lengths = [len(text) for text in texts]
        
        # Character vocabulary
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Duration statistics
        durations = [sample['duration'] for sample in all_samples]
        
        # Speaker statistics (if available)
        speakers = set()
        for sample in all_samples:
            if 'speaker_id' in sample:
                speakers.add(sample['speaker_id'])
        
        # Split statistics
        split_stats = {}
        for split, samples in organized_data.items():
            if samples:
                split_stats[split] = {
                    'count': int(len(samples)),
                    'duration': float(sum(s['duration'] for s in samples)),
                    'avg_duration': float(np.mean([s['duration'] for s in samples])),
                    'avg_text_length': float(np.mean([len(s['text']) for s in samples]))
                }
        
        # Text length distribution
        text_length_dist = {
            '0-50': sum(1 for length in text_lengths if 0 <= length <= 50),
            '50-100': sum(1 for length in text_lengths if 50 < length <= 100),
            '100-200': sum(1 for length in text_lengths if 100 < length <= 200),
            '200+': sum(1 for length in text_lengths if length > 200)
        }
        
        # Duration distribution
        duration_dist = {
            '0-5s': sum(1 for d in durations if 0 <= d <= 5),
            '5-10s': sum(1 for d in durations if 5 < d <= 10),
            '10-20s': sum(1 for d in durations if 10 < d <= 20),
            '20+s': sum(1 for d in durations if d > 20)
        }
        
        statistics = {
            'dataset': dataset_name,
            'language': language,
            'total_samples': int(total_samples),
            'total_duration_hours': float(total_duration / 3600),
            'avg_duration_seconds': float(np.mean(durations)),
            'min_duration_seconds': float(np.min(durations)),
            'max_duration_seconds': float(np.max(durations)),
            'vocabulary_size': int(len(all_chars)),
            'unique_speakers': int(len(speakers)),
            'avg_text_length': float(np.mean(text_lengths)),
            'min_text_length': int(np.min(text_lengths)),
            'max_text_length': int(np.max(text_lengths)),
            'split_statistics': split_stats,
            'text_length_distribution': text_length_dist,
            'duration_distribution': duration_dist,
            'character_vocabulary': sorted(list(all_chars))
        }
        
        return statistics
    
    def export_data(self, organized_data: Dict[str, List[Dict]], 
                   statistics: Dict, language: str, output_dir: Path):
        """Export organized data in multiple formats"""
        
        language_dir = output_dir / language
        language_dir.mkdir(parents=True, exist_ok=True)
        
        # Export each split in multiple formats
        for split_name, samples in organized_data.items():
            if not samples:
                continue
            
            # JSON format (detailed)
            json_file = language_dir / f"{split_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=2, ensure_ascii=False)
            
            # TSV format (tabular)
            tsv_file = language_dir / f"{split_name}.tsv"
            df = pd.DataFrame(samples)
            df.to_csv(tsv_file, sep='\t', index=False)
            
            # Text format (transcriptions only)
            txt_file = language_dir / f"{split_name}_text.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(sample['text'] + '\n')
            
            # JSONL manifest format (for NeMo/ESPnet)
            jsonl_file = language_dir / f"{split_name}_manifest.jsonl"
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    manifest_entry = {
                        'audio_filepath': sample.get('audio_filepath', ''),
                        'text': sample['text'],
                        'duration': sample['duration']
                    }
                    f.write(json.dumps(manifest_entry, ensure_ascii=False) + '\n')
        
        # Export statistics
        stats_file = language_dir / "statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {language} data in multiple formats to {language_dir}")
    
    def generate_overall_report(self, output_dir: Path):
        """Generate comprehensive report across all processed datasets"""
        
        if not self.all_statistics:
            logger.warning("No statistics available for report generation")
            return
        
        # Create summary report
        report_lines = ["# ASR Dataset Processing Report\n"]
        report_lines.append(f"**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall summary
        total_samples = sum(stats.get('total_samples', 0) for stats in self.all_statistics.values())
        total_duration = sum(stats.get('total_duration_hours', 0) for stats in self.all_statistics.values())
        languages = set(stats.get('language') for stats in self.all_statistics.values())
        
        report_lines.extend([
            "## Overall Summary\n",
            f"- **Total Languages:** {len(languages)}",
            f"- **Total Samples:** {total_samples:,}",
            f"- **Total Duration:** {total_duration:.1f} hours",
            f"- **Languages Processed:** {', '.join(sorted(languages))}\n",
            "## Language-Specific Statistics\n"
        ])
        
        # Language-specific details
        for key, stats in sorted(self.all_statistics.items()):
            lang = stats.get('language', 'unknown')
            dataset = stats.get('dataset', 'unknown')
            
            report_lines.extend([
                f"### {lang.upper()} ({dataset})\n",
                f"- **Total Samples:** {stats.get('total_samples', 0):,}",
                f"- **Duration:** {stats.get('total_duration_hours', 0):.1f} hours",
                f"- **Average Duration:** {stats.get('avg_duration_seconds', 0):.1f} seconds",
                f"- **Vocabulary Size:** {stats.get('vocabulary_size', 0):,} characters",
                f"- **Average Text Length:** {stats.get('avg_text_length', 0):.1f} characters",
                f"- **Unique Speakers:** {stats.get('unique_speakers', 0):,}\n"
            ])
            
            # Split breakdown
            split_stats = stats.get('split_statistics', {})
            if split_stats:
                report_lines.append("#### Split Distribution")
                for split, split_info in split_stats.items():
                    report_lines.append(f"- **{split.capitalize()}:** {split_info.get('count', 0):,} samples")
                report_lines.append("")
        
        # Save report
        report_file = output_dir / "dataset_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Create CSV summary
        summary_data = []
        for key, stats in self.all_statistics.items():
            summary_data.append({
                'Language': stats.get('language', ''),
                'Dataset': stats.get('dataset', ''),
                'Total_Samples': stats.get('total_samples', 0),
                'Duration_Hours': stats.get('total_duration_hours', 0),
                'Avg_Duration_Sec': stats.get('avg_duration_seconds', 0),
                'Vocabulary_Size': stats.get('vocabulary_size', 0),
                'Avg_Text_Length': stats.get('avg_text_length', 0),
                'Unique_Speakers': stats.get('unique_speakers', 0)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "language_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Save overall statistics JSON
        overall_stats_file = output_dir / "overall_statistics.json"
        with open(overall_stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_statistics, f, indent=2, ensure_ascii=False)
        
        logger.info("Comprehensive statistics and report generated successfully")
    
    def process_single_dataset(self, dataset_name: str, language: str, 
                             force_download: bool = False) -> bool:
        """Process a single dataset for a single language"""
        
        logger.info(f"Processing {dataset_name} for {language}")
        
        # Check existing data first
        existing_dir = self.check_existing_data(dataset_name, language)
        if existing_dir and not force_download:
            # Load existing statistics
            stats = self.load_existing_statistics(existing_dir, language)
            if stats:
                self.all_statistics[f"{dataset_name}_{language}"] = stats
                logger.info(f"Using existing data for {dataset_name}/{language}")
                return True
        
        # Download dataset
        dataset = self.download_dataset(dataset_name, language, force_download)
        
        if dataset is None and not force_download:
            # Data might already exist, check if we loaded stats
            return f"{dataset_name}_{language}" in self.all_statistics
        
        if dataset is None:
            return False
        
        # Process dataset
        processed_data = self.process_dataset(dataset, dataset_name, language)
        if not processed_data:
            logger.error(f"No data processed for {dataset_name}/{language}")
            return False
        
        # Organize for ASR training
        organized_data = self.organize_for_asr(processed_data, language)
        
        # Calculate statistics
        statistics = self.calculate_statistics(organized_data, dataset_name, language)
        self.all_statistics[f"{dataset_name}_{language}"] = statistics
        
        # Export data
        output_dir = self.data_dir / "asr_datasets" / f"{dataset_name}_{language}"
        self.export_data(organized_data, statistics, language, output_dir)
        
        # Mark as cached
        total_samples = statistics.get('total_samples', 0)
        total_duration = statistics.get('total_duration_hours', 0)
        lang_mapping = self.LANGUAGE_MAPPING.get(dataset_name, {})
        dataset_lang = lang_mapping.get(language, language)
        
        self.dataset_cache.mark_cached(dataset_name, language, dataset_lang, 
                                     total_samples, total_duration)
        
        logger.info(f"Successfully processed {dataset_name} for {language}")
        return True
    
    def download_and_process(self, datasets: List[str], languages: List[str], 
                           force_download: bool = False):
        """Main method to download and process multiple datasets and languages"""
        
        logger.info("Starting ASR dataset download and processing...")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Languages: {languages}")
        
        # Ensure FLEURS is processed first
        dataset_order = []
        if 'fleurs' in datasets:
            dataset_order.append('fleurs')
        for dataset in datasets:
            if dataset not in dataset_order:
                dataset_order.append(dataset)
        
        logger.info(f"Processing order: {dataset_order}")
        
        success_count = 0
        total_count = 0
        
        for dataset_name in dataset_order:
            for language in languages:
                total_count += 1
                
                # Skip unsupported combinations
                if dataset_name == 'aishell' and language != 'zh':
                    logger.info(f"Skipping {dataset_name} for {language} (not supported)")
                    continue
                
                try:
                    success = self.process_single_dataset(dataset_name, language, force_download)
                    if success:
                        success_count += 1
                    else:
                        logger.error(f"Failed to process {dataset_name} for {language}")
                
                except Exception as e:
                    logger.error(f"Error processing {dataset_name}/{language}: {str(e)}")
        
        # Generate overall report
        if self.all_statistics:
            output_dir = self.data_dir / "asr_datasets"
            self.generate_overall_report(output_dir)
        
        # Print summary
        print("\n" + "="*80)
        print("ASR DATASET PROCESSING SUMMARY")
        print("="*80)
        
        if self.all_statistics:
            total_samples = sum(stats.get('total_samples', 0) for stats in self.all_statistics.values())
            total_duration = sum(stats.get('total_duration_hours', 0) for stats in self.all_statistics.values())
            processed_languages = set(stats.get('language') for stats in self.all_statistics.values())
            
            print(f"\nOverall Statistics:")
            print(f"  Total Languages: {len(processed_languages)}")
            print(f"  Total Samples: {total_samples:,}")
            print(f"  Total Duration: {total_duration:.1f} hours")
            
            for key, stats in sorted(self.all_statistics.items()):
                lang = stats.get('language', '')
                dataset = stats.get('dataset', '')
                samples = stats.get('total_samples', 0)
                duration = stats.get('total_duration_hours', 0)
                
                split_stats = stats.get('split_statistics', {})
                train_count = split_stats.get('train', {}).get('count', 0)
                dev_count = split_stats.get('dev', {}).get('count', 0)
                test_count = split_stats.get('test', {}).get('count', 0)
                
                print(f"\n{lang.upper()} ({dataset}):")
                print(f"  Total samples: {samples:,}")
                print(f"  Duration: {duration:.1f} hours")
                if train_count or dev_count or test_count:
                    print(f"  Train: {train_count:,}")
                    print(f"  Dev: {dev_count:,}")
                    print(f"  Test: {test_count:,}")
                print(f"  Speakers: {stats.get('unique_speakers', 0):,}")
                print(f"  Vocabulary: {stats.get('vocabulary_size', 0):,} characters")
                print(f"  Avg text length: {stats.get('avg_text_length', 0):.1f} chars")
        
        print(f"\nProcessing complete! Results saved to: {self.data_dir / 'asr_datasets'}")
        print(f"Cache saved to: {self.cache_dir}")

def main():
    """Main entry point for the Enhanced ASR Dataset Downloader
    
    This function sets up command line argument parsing and initiates the download
    and processing of ASR datasets from Hugging Face Hub.
    """
    parser = argparse.ArgumentParser(
        description="Enhanced ASR Dataset Downloader for Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all default datasets for all default languages
  python enhanced_asr_downloader.py
  
  # Download specific datasets for specific languages
  python enhanced_asr_downloader.py --datasets common_voice fleurs --languages ko en
  
  # Use custom data directory
  python enhanced_asr_downloader.py --data-dir /path/to/data
  
  # Force re-download (ignore cache)
  python enhanced_asr_downloader.py --force-download
        """
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        default=['ko', 'en', 'zh', 'fr', 'es'],
        help='Languages to download (default: ko en zh fr es)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['common_voice', 'fleurs', 'aishell'],
        default=['fleurs', 'common_voice', 'aishell'],
        help='Datasets to download (default: fleurs, common_voice, aishell)'
    )
    
    parser.add_argument(
        '--data-dir',
        default='~/work/data',
        help='Output directory for processed data (default: ~/work/data)'
    )
    
    parser.add_argument(
        '--cache-dir',
        help='Cache directory for Hugging Face datasets (default: data-dir/hf_cache)'
    )
    
    parser.add_argument(
        '--hf-token',
        help='Hugging Face authentication token'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force re-download even if data exists'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = EnhancedASRDownloader(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token
    )
    
    # Start processing
    downloader.download_and_process(
        datasets=args.datasets,
        languages=args.languages,
        force_download=args.force_download
    )

if __name__ == "__main__":
    main() 