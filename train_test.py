#!/usr/bin/env python3

import tokenizers
from tokenizers.implementations import UTF16ByteLevelBPETokenizer
import tempfile
import os

# Create training data
training_data = ['Hello world', '안녕하세요', '가나다라마바사']
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    for line in training_data:
        f.write(line + '\n')
    train_file = f.name

try:
    # Create and train tokenizer
    tokenizer = UTF16ByteLevelBPETokenizer()
    print('Training tokenizer...')
    tokenizer.train([train_file], vocab_size=1000, min_frequency=1)
    print('Training completed')
    
    # Test encoding
    korean_text = '가나다'
    encoding = tokenizer.encode(korean_text)
    print(f'Tokens: {encoding.tokens}')
    print(f'IDs: {encoding.ids}')
    
    # Test decoding
    decoded = tokenizer.decode(encoding.ids)
    print(f'Decoded: {decoded}')
    print(f'Roundtrip: {decoded == korean_text}')
    
finally:
    os.unlink(train_file) 