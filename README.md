<p align="center">
    <br>
    <img src="https://huggingface.co/landing/assets/tokenizers/tokenizers-logo.png" width="600"/>
    <br>
<p>
<p align="center">
    <img alt="Build" src="https://github.com/huggingface/tokenizers/workflows/Rust/badge.svg">
    <a href="https://github.com/huggingface/tokenizers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/tokenizers.svg?color=blue&cachedrop">
    </a>
    <a href="https://pepy.tech/project/tokenizers">
        <img src="https://pepy.tech/badge/tokenizers/week" />
    </a>
</p>

## UTF16ByteLevelBPETokenizer

A specialized tokenizer optimized for CJK (Chinese, Japanese, Korean) languages that processes text at the UTF-16 byte level while maintaining compatibility with the standard ByteLevel alphabet.

### Key Advantages:
- **26.3% fewer tokens** for Chinese text
- **9.2% fewer tokens** for Korean text  
- **6.7% fewer tokens** for multilingual content
- **100% accuracy** maintained across all languages
- Uses standard ByteLevel.alphabet (256 characters)

### Usage:
```python
from tokenizers.implementations import UTF16ByteLevelBPETokenizer

# Initialize tokenizer
tokenizer = UTF16ByteLevelBPETokenizer()

# Train on your dataset
tokenizer.train(files=["training_data.txt"], vocab_size=1000)

# Encode CJK text efficiently
output = tokenizer.encode("안녕하세요 你好 こんにちは")
print(f"Tokens: {len(output.tokens)}")  # Significantly fewer tokens for CJK text
```

### When to Use:
- Processing primarily CJK languages
- Building applications for Asian markets
- Token efficiency is critical for CJK text
- Multilingual models with significant non-ASCII content

### Evaluation Results:
Comprehensive testing on 434 test cases across Korean, English, Chinese, and mixed languages shows significant improvements for CJK languages while maintaining perfect roundtrip accuracy. See `~/work/data/utf16_tokenizer_evaluation/` for detailed results.

### Citation:
If you use UTF16ByteLevelBPETokenizer in your research or applications, please cite:

```bibtex
@misc{kim2025utf16bytelevel,
  title={UTF16ByteLevelBPETokenizer: Enhanced Tokenization for CJK Languages},
  author={Hyunsik Kim},
  year={2025},
  month={May},
  note={Implementation based on HuggingFace Tokenizers library},
  email={avantkim@gmail.com},
  url={https://github.com/peblove/tokenizers}
}
```

### Technical References:
- **Base Implementation**: HuggingFace Tokenizers - [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- **GPT-2 Byte-Level BPE**: Radford et al. (2019) - [https://github.com/openai/gpt-2](https://github.com/openai/gpt-2)
- **UTF-16 Encoding Standard**: Unicode Consortium - [https://unicode.org/standard/standard.html](https://unicode.org/standard/standard.html)

### Performance Benchmarks:
| Language | Standard BPE | UTF16ByteLevel BPE | Improvement |
|----------|-------------|-------------------|-------------|
| Chinese  | 29.7 tokens | 21.9 tokens | **26.3% reduction** |
| Korean   | 20.7 tokens | 18.8 tokens | **9.2% reduction** |
| Mixed    | 28.2 tokens | 26.3 tokens | **6.7% reduction** |
| English  | 24.5 tokens | 25.5 tokens | 4.1% increase |

*Results based on comprehensive evaluation with 1000-token vocabularies trained on 10MB multilingual dataset.*

Provides an implementation of today's most used tokenizers, with a focus on performance and
versatility.

## Main features:

 - Train new vocabularies and tokenize, using today's most used tokenizers.
 - Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes
   less than 20 seconds to tokenize a GB of text on a server's CPU.
 - Easy to use, but also extremely versatile.
 - Designed for research and production.
 - Normalization comes with alignments tracking. It's always possible to get the part of the
   original sentence that corresponds to a given token.
 - Does all the pre-processing: Truncate, Pad, add the special tokens your model needs.
 - **NEW**: UTF16ByteLevelBPETokenizer for enhanced CJK language support with up to 26% token reduction for Chinese text.

## Performances
Performances can vary depending on hardware, but running the [~/bindings/python/benches/test_tiktoken.py](bindings/python/benches/test_tiktoken.py) should give the following on a g6 aws instance:
![image](https://github.com/user-attachments/assets/2b913d4b-e488-4cbc-b542-f90a6c40643d)


## Bindings

We provide bindings to the following languages (more to come!):
  - [Rust](https://github.com/huggingface/tokenizers/tree/main/tokenizers) (Original implementation)
  - [Python](https://github.com/huggingface/tokenizers/tree/main/bindings/python)
  - [Node.js](https://github.com/huggingface/tokenizers/tree/main/bindings/node)
  - [Ruby](https://github.com/ankane/tokenizers-ruby) (Contributed by @ankane, external repo)

## Installation

You can install from source using:
```bash
pip install git+https://github.com/peblove/tokenizers.git#subdirectory=bindings/python
```

our install the released versions with

```bash
pip install tokenizers
```
 
## Quick example using Python:

Choose your model between Byte-Pair Encoding, WordPiece or Unigram and instantiate a tokenizer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE

tokenizer = Tokenizer(BPE())
```

You can customize how pre-tokenization (e.g., splitting into words) is done:

```python
from tokenizers.pre_tokenizers import Whitespace

tokenizer.pre_tokenizer = Whitespace()
```

Then training your tokenizer on a set of files just takes two lines of codes:

```python
from tokenizers.trainers import BpeTrainer

trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
```

Once your tokenizer is trained, encode any text with just one line:
```python
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
# ["Hello", ",", "y", "'", "all", "!", "How", "are", "you", "[UNK]", "?"]
```

Check the [documentation](https://huggingface.co/docs/tokenizers/index)
or the [quicktour](https://huggingface.co/docs/tokenizers/quicktour) to learn more!

