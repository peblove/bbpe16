# Enhanced ASR Dataset Downloader

This tool downloads and organizes speech datasets from Hugging Face for Automatic Speech Recognition (ASR) training. It supports Common Voice, FLEURS, and AI-Shell datasets across multiple languages with advanced caching and resume capabilities.

## 🚀 Quick Start

### Basic Usage (All datasets, all languages)
```bash
cd scripts/dataset_downloader
python3 enhanced_asr_downloader.py
```

### Custom Language Selection
```bash
python3 enhanced_asr_downloader.py --languages ko en
```

### Custom Dataset Selection
```bash
python3 enhanced_asr_downloader.py --datasets common_voice fleurs
```

### Custom Data Directory
```bash
python3 enhanced_asr_downloader.py --data-dir /path/to/my/data
```

## 📋 Features

- **🌍 Multi-language Support**: Korean (ko), English (en), Chinese (zh), French (fr), Spanish (es)
- **📊 Multi-dataset Support**: Common Voice, FLEURS, AI-Shell
- **💾 Smart Caching**: Avoids re-downloading existing data
- **📈 Comprehensive Statistics**: Detailed analysis and reporting for each language
- **🔄 ASR-Ready Formats**: JSON, TSV, TXT, and JSONL manifest formats
- **🎯 Proper Data Splits**: Training, validation, and test sets organized for ASR training
- **🔧 Easy Integration**: Compatible with popular ASR frameworks (NeMo, ESPnet, etc.)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Sufficient disk space (datasets can be large: 1-20 GB per language)

### Install Dependencies
```bash
cd scripts/dataset_downloader
pip install -r requirements.txt
```

## 🗂️ Supported Datasets

| Dataset | Source | Languages | Content | Splits |
|---------|--------|-----------|---------|--------|
| **Common Voice** | Mozilla Foundation | ko, en, zh, fr, es | Crowdsourced speech recordings | train, validation, test |
| **FLEURS** | Google | ko, en, zh, fr, es | Few-shot Learning Evaluation corpus | train, validation, test |
| **AI-Shell** | SpeechColab | zh only | Mandarin speech recognition corpus | train, validation, test |

## 🎛️ Command Line Options

```bash
python3 enhanced_asr_downloader.py [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--languages` | Space-separated language codes | `ko en zh fr es` |
| `--datasets` | Dataset names to download | `fleurs common_voice aishell` |
| `--data-dir` | Output directory for processed data | `~/work/data` |
| `--cache-dir` | Cache directory for Hugging Face datasets | `{data-dir}/hf_cache` |
| `--hf-token` | Hugging Face authentication token | None |
| `--force-download` | Force re-download even if data exists | False |
| `--help` | Show help message | - |

## 📁 Output Structure

```
~/work/data/asr_datasets/
├── overall_statistics.json          # Overall processing statistics
├── language_summary.csv             # Summary table of all languages
├── dataset_report.md                # Detailed markdown report
├── common_voice_ko/                 # Common Voice Korean data
│   └── ko/
│       ├── train.json               # Training data (detailed metadata)
│       ├── train.tsv                # Training data (tabular format)
│       ├── train_text.txt           # Training transcriptions only
│       ├── train_manifest.jsonl     # Training manifest for ASR frameworks
│       ├── dev.json                 # Validation data
│       ├── dev.tsv                  # Validation data (tabular)
│       ├── dev_text.txt             # Validation transcriptions only
│       ├── dev_manifest.jsonl       # Validation manifest
│       ├── test.json                # Test data
│       ├── test.tsv                 # Test data (tabular)
│       ├── test_text.txt            # Test transcriptions only
│       ├── test_manifest.jsonl      # Test manifest
│       └── statistics.json          # Korean-specific statistics
├── fleurs_ko/                       # FLEURS Korean data (same structure)
├── fleurs_en/                       # FLEURS English data
├── common_voice_en/                 # Common Voice English data
└── aishell_zh/                      # AI-Shell Chinese data
```

## 📊 File Formats

### JSON Files (Detailed metadata)
```json
[
  {
    "text": "안녕하세요",
    "audio_filepath": "",
    "duration": 2.5,
    "sampling_rate": 16000,
    "speaker_id": "client_12345",
    "gender": "female",
    "age": "twenties"
  }
]
```

### TSV Files (Tabular format)
```tsv
text	audio_filepath	duration	speaker_id	gender
안녕하세요	cv_ko_train_000001	2.5	client_12345	female
```

### Manifest Files (ASR framework ready)
```jsonl
{"audio_filepath": "", "text": "안녕하세요", "duration": 2.5}
{"audio_filepath": "", "text": "좋은 아침입니다", "duration": 3.1}
```

### Text Files (Pure transcriptions)
```
안녕하세요
좋은 아침입니다
감사합니다
```

## 📈 Statistics and Reports

The tool generates comprehensive statistics including:

- **📊 Sample counts** by language and split
- **⏱️ Duration analysis** (total hours, average duration)
- **📝 Text statistics** (length distribution, vocabulary size)
- **🎤 Speaker information** (unique speakers, demographics)
- **📋 Dataset distribution** (samples per dataset)
- **✅ Quality metrics** (completeness scores)

### Key Output Files

1. **`dataset_report.md`** - Human-readable markdown report
2. **`overall_statistics.json`** - Machine-readable complete statistics
3. **`language_summary.csv`** - Spreadsheet-compatible summary
4. **`{dataset}_{language}/{language}/statistics.json`** - Per-language detailed statistics

## 🔧 Integration with ASR Frameworks

### NeMo (NVIDIA)
```python
from omegaconf import OmegaConf
from nemo.collections.asr.data.audio_to_text import AudioToCharDataset

# Use the manifest files directly
train_manifest = "~/work/data/asr_datasets/fleurs_ko/ko/train_manifest.jsonl"
config = OmegaConf.create({
    'manifest_filepath': train_manifest,
    'sample_rate': 16000,
    'labels': list("abcdefghijklmnopqrstuvwxyz'"),
})
dataset = AudioToCharDataset(config)
```

### ESPnet
```bash
# Use the manifest files for training
espnet2_train.py \
    --train_data_path_and_name_and_type ~/work/data/asr_datasets/fleurs_ko/ko/train_manifest.jsonl:speech:jsonl \
    --valid_data_path_and_name_and_type ~/work/data/asr_datasets/fleurs_ko/ko/dev_manifest.jsonl:speech:jsonl
```

### Custom Training
```python
import json
import pandas as pd

# Load data for custom training
with open('~/work/data/asr_datasets/fleurs_ko/ko/train.json', 'r') as f:
    train_data = json.load(f)

# Or use pandas for easier manipulation
train_df = pd.read_csv('~/work/data/asr_datasets/fleurs_ko/ko/train.tsv', sep='\t')
```

## 💡 Usage Examples

### Download Korean and English only
```bash
python3 enhanced_asr_downloader.py --languages ko en
```

### Download FLEURS only for all languages
```bash
python3 enhanced_asr_downloader.py --datasets fleurs
```

### Use custom data directory
```bash
python3 enhanced_asr_downloader.py --data-dir /mnt/storage/asr_data
```

### Force re-download (ignore cache)
```bash
python3 enhanced_asr_downloader.py --force-download
```

### Use Hugging Face token for authentication
```bash
python3 enhanced_asr_downloader.py --hf-token hf_your_token_here
```

## 🚨 Troubleshooting

### Common Issues

1. **Network Issues**: Datasets are large; ensure stable internet connection
2. **Disk Space**: Each language can require several GB of storage
3. **Memory Usage**: Processing large datasets may require sufficient RAM (2-8 GB)
4. **Authentication**: Some datasets may require Hugging Face login

### Error Solutions

```bash
# Install missing dependencies
pip install -r requirements.txt

# Check Python version
python3 --version  # Should be 3.8+

# Update Hugging Face libraries
pip install --upgrade datasets huggingface_hub

# Clear cache if corrupted
rm -rf ~/work/data/hf_cache
```

## ⚡ Performance Notes

- **Download Time**: 10 minutes to several hours depending on languages and datasets
- **Storage Requirements**: 1-20 GB per language depending on dataset size
- **Processing Time**: 5-30 minutes for organization and statistics generation
- **Memory Usage**: 2-8 GB RAM during processing

## 📄 License and Attribution

This tool downloads data from:
- **Common Voice**: Mozilla Public License 2.0
- **FLEURS**: Creative Commons Attribution 4.0
- **AI-Shell**: Apache License 2.0

Please cite the original datasets in your work:

```bibtex
@article{commonvoice,
    title={Common Voice: A Massively-Multilingual Speech Corpus},
    author={Ardila, Rosana and others},
    journal={arXiv preprint arXiv:1912.06670},
    year={2019}
}

@article{fleurs,
    title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
    author={Conneau, Alexis and others},
    journal={arXiv preprint arXiv:2205.12446},
    year={2022}
}

@inproceedings{aishell,
    title={AIShell-1: An Open-Source Mandarin Speech Corpus and A Speech Recognition Baseline},
    author={Bu, Hui and others},
    booktitle={Oriental COCOSDA},
    year={2017}
}
```

## 🤝 Contributing

To add support for additional languages or datasets:

1. Update `DATASET_CONFIGS` in `enhanced_asr_downloader.py`
2. Add language mappings in `LANGUAGE_MAPPING`
3. Test with the new configuration
4. Update documentation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the generated logs for error details
3. Ensure all dependencies are properly installed
4. Check Hugging Face dataset documentation for specific dataset issues 