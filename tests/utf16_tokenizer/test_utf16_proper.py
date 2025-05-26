#!/usr/bin/env python3

import tokenizers
from tokenizers.decoders import UTF16ByteLevel, ByteLevel

print('=== UTF16ByteLevel 디코더 올바른 사용법 테스트 ===')

# UTF16ByteLevel 디코더 생성
utf16_decoder = UTF16ByteLevel()
byte_decoder = ByteLevel()

print('\n1. 빈 토큰 리스트 테스트')
empty_tokens = []
try:
    utf16_result = utf16_decoder.decode(empty_tokens)
    print(f'✓ UTF16ByteLevel 빈 리스트 디코딩 성공: {repr(utf16_result)}')
except Exception as e:
    print(f'✗ UTF16ByteLevel 빈 리스트 디코딩 에러: {e}')

try:
    byte_result = byte_decoder.decode(empty_tokens)
    print(f'✓ ByteLevel 빈 리스트 디코딩 성공: {repr(byte_result)}')
except Exception as e:
    print(f'✗ ByteLevel 빈 리스트 디코딩 에러: {e}')

print('\n2. UTF-16 바이트 매핑 문자 테스트')
# UTF-16에서 "Hello"는 다음과 같은 바이트 시퀀스가 됩니다:
# H(0x48,0x00), e(0x65,0x00), l(0x6C,0x00), l(0x6C,0x00), o(0x6F,0x00)
# 이 바이트들이 특별한 문자로 매핑되어야 합니다.

# 실제 UTF16ByteLevel pre-tokenizer에서 생성될 수 있는 문자들을 시뮬레이션
# 이는 예시이며, 실제로는 pre-tokenizer와 함께 사용되어야 합니다.

print('\n3. 디코더 객체 속성 테스트')
print(f'UTF16ByteLevel 디코더 타입: {type(utf16_decoder)}')
print(f'UTF16ByteLevel 디코더 문자열: {utf16_decoder}')
print(f'ByteLevel 디코더 타입: {type(byte_decoder)}')
print(f'ByteLevel 디코더 문자열: {byte_decoder}')

print('\n4. 디코더 비교 테스트')
utf16_decoder2 = UTF16ByteLevel()
print(f'두 UTF16ByteLevel 디코더가 같은 타입인가: {type(utf16_decoder) == type(utf16_decoder2)}')

print('\n5. 에러 케이스 테스트 (예상되는 에러)')
problematic_tokens = ['Hello', 'World']  # 이는 홀수 바이트를 생성하여 에러가 예상됨
try:
    utf16_result = utf16_decoder.decode(problematic_tokens)
    print(f'예상치 못한 성공: {repr(utf16_result)}')
except Exception as e:
    print(f'✓ 예상된 에러 발생: {e}')

# ByteLevel은 정상 작동해야 함
try:
    byte_result = byte_decoder.decode(problematic_tokens)
    print(f'✓ ByteLevel 정상 디코딩: {repr(byte_result)}')
except Exception as e:
    print(f'✗ ByteLevel 예상치 못한 에러: {e}')

print('\n=== 테스트 완료 ===')
print('\n참고: UTF16ByteLevel 디코더는 UTF16ByteLevel pre-tokenizer와')
print('함께 사용되어야 하며, 일반 텍스트를 직접 디코딩하는 용도가 아닙니다.') 