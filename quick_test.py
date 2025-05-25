#!/usr/bin/env python3

import tokenizers
from tokenizers.implementations import UTF16ByteLevelBPETokenizer
print('✓ Successfully imported UTF16ByteLevelBPETokenizer')

# Test basic creation
tokenizer = UTF16ByteLevelBPETokenizer()
print('✓ Tokenizer created successfully')

# Test Korean text
korean_text = '가나다'
print(f'Testing Korean text: {korean_text}')

# Test encoding
encoding = tokenizer.encode(korean_text)
print(f'✓ Encoding successful: {len(encoding.tokens)} tokens')
print(f'  Tokens: {encoding.tokens}')
print(f'  IDs: {encoding.ids}')

# Test decoding
decoded = tokenizer.decode(encoding.ids)
print(f'✓ Decoding successful: {decoded}')

# Check roundtrip
if decoded == korean_text:
    print('✓ Roundtrip test passed')
else:
    print(f'✗ Roundtrip test failed: {korean_text} != {decoded}') 