# UTF16ByteLevelBPETokenizer Tests

This directory contains comprehensive tests for the UTF16ByteLevelBPETokenizer implementation.

## Test Files

- `test_basic.py` - Basic functionality tests for UTF16ByteLevelBPETokenizer
- `test_basic_functionality.py` - Core functionality validation tests
- `test_integration.py` - Integration tests comparing ByteLevel vs UTF16ByteLevel tokenizers
- `test_utf16_detailed.py` - Detailed UTF16 tokenizer specific tests
- `test_utf16_proper.py` - Proper implementation validation tests

## Running Tests

To run all tests in this directory:

```bash
cd tests/utf16_tokenizer
python -m pytest .
```

To run individual test files:

```bash
python test_basic.py
python test_integration.py
# etc.
```

## Test Coverage

These tests validate:
- UTF16ByteLevelBPETokenizer initialization and configuration
- Encoding/decoding accuracy across multiple languages
- Token efficiency comparison with ByteLevelBPETokenizer
- Alphabet size validation (256 characters)
- Multilingual performance (Korean, English, Chinese, Other languages)
- Error handling and edge cases

## Requirements

- Python 3.7+
- tokenizers library with UTF16ByteLevelBPETokenizer implementation
- pytest (for running test suite)

## Author

Hyunsik Kim <avantkim@gmail.com> 