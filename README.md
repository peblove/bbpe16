## Why BBPE16 (UTF-16 Byte-level Byte-Pair Encoding)?

**BBPE16** (UTF-16 Byte-level Byte-Pair Encoding) offers significant advantages over traditional Byte-Level BPE (BBPE), especially for multilingual and CJK (Chinese, Japanese, Korean) language environments:

- **Superior Token Efficiency:** BBPE16 dramatically reduces the number of tokens required for CJK and multilingual text, resulting in fewer decoding steps for LLMs. This leads to lower service costs and faster response times, which is critical for production environments and large-scale applications.
- **Seamless Integration:** When encoding, BBPE16 automatically converts UTF-8 strings to UTF-16 internally, and during decoding, it processes tokens based on UTF-16 and returns standard UTF-8 strings. This makes it extremely easy to use—no manual encoding or decoding steps are required by the user.
- **Better for CJK and Multilingual Content:** Standard BBPE is optimized for English and other Latin-based languages, but is inefficient for CJK text. BBPE16 is specifically designed to address this inefficiency, providing up to 26% token reduction for Chinese and significant improvements for Korean and mixed-language content.
- **Plug-and-Play for LLMs:** By reducing the number of tokens, BBPE16 not only improves computational efficiency but also enhances the user experience with faster model responses and lower latency.
- **Same Alphabet, Superior Performance:** BBPE16 uses the same initial 256-character byte-level alphabet as standard BBPE, and guarantees superior performance at the same vocabulary size.
- **Automatic Model Detection:** The tokenizer automatically distinguishes between BBPE and BBPE16 models when loading, ensuring seamless integration and ease of use without manual intervention.

## UTF16ByteLevelBPETokenizer

A specialized tokenizer optimized for CJK (Chinese, Japanese, Korean) languages that processes text at the UTF-16 byte level while maintaining compatibility with the standard ByteLevel alphabet.

### Key Advantages:
- **26.3% fewer tokens** for Chinese text
- **9.2% fewer tokens** for Korean text  
- **6.7% fewer tokens** for multilingual content
- **100% accuracy** maintained across all languages
- Compatible byte-level alphabet (256 characters, same mapping strategy as ByteLevel)

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

### Performance Benchmarks:
| Language | Standard BPE | UTF16ByteLevel BPE | Improvement |
|----------|-------------|-------------------|-------------|
| Chinese  | 29.7 tokens | 21.9 tokens | **26.3% reduction** |
| Korean   | 20.7 tokens | 18.8 tokens | **9.2% reduction** |
| Mixed    | 28.2 tokens | 26.3 tokens | **6.7% reduction** |
| English  | 24.5 tokens | 25.5 tokens | 4.1% increase |

*Results based on comprehensive evaluation with 1000-token vocabularies trained on 10MB multilingual dataset.*

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
