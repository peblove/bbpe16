# Test Data for UTF16ByteLevelBPETokenizer

This directory contains sample vocabulary and merge files generated during the testing and evaluation of the UTF16ByteLevelBPETokenizer implementation.

## Directory Structure

```
test_data/
├── byte_level_vocab/
│   ├── vocab.json          # ByteLevel BPE vocabulary (1000 tokens)
│   └── merges.txt          # ByteLevel BPE merge rules (simplified)
├── utf16_byte_level_vocab/
│   ├── vocab.json          # UTF16ByteLevel BPE vocabulary (300 tokens shown)
│   └── merges.txt          # UTF16ByteLevel BPE merge rules (simplified)
└── README.md               # This file
```

## Training Details

- **Training Data**: 10.15MB multilingual dataset
  - Korean: 50%
  - English: 40% 
  - Chinese: 5%
  - Other languages: 5%

- **Vocabulary Size**: 1000 tokens each
- **Training Time**: 
  - ByteLevel BPE: 0.58 seconds
  - UTF16ByteLevel BPE: 0.60 seconds

## Performance Results

### Token Efficiency (Average tokens per sentence)

| Language | ByteLevel BPE | UTF16ByteLevel BPE | UTF16 Improvement |
|----------|---------------|-------------------|-------------------|
| Korean   | 12.6 tokens   | 10.4 tokens       | **17.5% reduction** |
| Chinese  | 13.0 tokens   | 9.8 tokens        | **24.6% reduction** |
| English  | 9.2 tokens    | 9.6 tokens        | 4.3% increase |
| Mixed    | 16.0 tokens   | 14.4 tokens       | **10.0% reduction** |

### Accuracy

- **Roundtrip Accuracy**: 100% for both tokenizers
- **Encoding Errors**: 0 for both tokenizers
- **Decoding Errors**: 0 for both tokenizers

## Key Findings

1. **UTF16ByteLevelBPETokenizer shows significant advantages for CJK languages**:
   - Chinese: 24.6% fewer tokens
   - Korean: 17.5% fewer tokens
   - Mixed multilingual: 10.0% fewer tokens

2. **Perfect reliability maintained**:
   - Zero encoding/decoding errors
   - 100% roundtrip accuracy across all test cases

3. **English performance trade-off**:
   - Slight increase in token count (4.3%)
   - Acceptable trade-off for CJK improvements

## Sample Tokenization Results

### Korean Examples
- "안녕하세요" → UTF16: 6 tokens vs ByteLevel: 8 tokens (25% reduction)
- "가나다" → UTF16: 4 tokens vs ByteLevel: 5 tokens (20% reduction)

### Chinese Examples
- "你好世界" → UTF16: 8 tokens vs ByteLevel: 11 tokens (27.3% reduction)

### English Examples
- "Hello World" → Both: 7 tokens (equal performance)

## Usage

These vocabulary files can be used to reproduce the tokenization results or for further testing of the UTF16ByteLevelBPETokenizer implementation.

**Author**: Hyunsik Kim <avantkim@gmail.com>  
**Date**: May 2025 