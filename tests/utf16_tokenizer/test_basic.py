#!/usr/bin/env python3

import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import UTF16ByteLevel, ByteLevel

print('=== Tokenizers 기본 동작 테스트 ===')
print(f'Tokenizers 버전: {tokenizers.__version__}')

# 1. UTF16ByteLevel 디코더 생성 테스트
print('\n1. UTF16ByteLevel 디코더 테스트')
utf16_decoder = UTF16ByteLevel()
print('✓ UTF16ByteLevel 디코더 생성 성공')
print(f'디코더 타입: {type(utf16_decoder)}')

# 2. ByteLevel 디코더와 비교
print('\n2. ByteLevel 디코더와 비교')
byte_decoder = ByteLevel()
print('✓ ByteLevel 디코더 생성 성공')
print(f'ByteLevel 디코더 타입: {type(byte_decoder)}')

# 3. 디코더 기본 동작 테스트
print('\n3. 디코더 기본 동작 테스트')
test_tokens = ['Hello', 'World']
try:
    result = utf16_decoder.decode(test_tokens)
    print(f'✓ UTF16ByteLevel 디코딩 성공: {repr(result)}')
except Exception as e:
    print(f'UTF16ByteLevel 디코딩 에러: {e}')

try:
    result = byte_decoder.decode(test_tokens)
    print(f'✓ ByteLevel 디코딩 성공: {repr(result)}')
except Exception as e:
    print(f'ByteLevel 디코딩 에러: {e}')

# 4. BPE 토크나이저 생성
print('\n4. BPE 토크나이저 생성 테스트')
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
print('✓ BPE 토크나이저 생성 성공')

# 5. 토크나이저에 디코더 설정 테스트
print('\n5. 토크나이저에 디코더 설정 테스트')
try:
    tokenizer.decoder = utf16_decoder
    print('✓ UTF16ByteLevel 디코더 설정 성공')
except Exception as e:
    print(f'UTF16ByteLevel 디코더 설정 에러: {e}')

try:
    tokenizer.decoder = byte_decoder
    print('✓ ByteLevel 디코더 설정 성공')
except Exception as e:
    print(f'ByteLevel 디코더 설정 에러: {e}')

print('\n=== 테스트 완료 ===') 