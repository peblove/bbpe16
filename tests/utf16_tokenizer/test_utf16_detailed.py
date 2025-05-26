#!/usr/bin/env python3

import tokenizers
from tokenizers.decoders import UTF16ByteLevel, ByteLevel

print('=== UTF16ByteLevel 디코더 상세 테스트 ===')

# UTF16ByteLevel 디코더 생성
utf16_decoder = UTF16ByteLevel()
byte_decoder = ByteLevel()

print('\n1. 기본 ASCII 텍스트 테스트')
ascii_tokens = ['Hello', ' ', 'World']
utf16_result = utf16_decoder.decode(ascii_tokens)
byte_result = byte_decoder.decode(ascii_tokens)
print(f'입력 토큰: {ascii_tokens}')
print(f'UTF16ByteLevel 결과: {repr(utf16_result)}')
print(f'ByteLevel 결과: {repr(byte_result)}')

print('\n2. 단일 문자 테스트')
single_chars = ['H', 'e', 'l', 'l', 'o']
utf16_result = utf16_decoder.decode(single_chars)
byte_result = byte_decoder.decode(single_chars)
print(f'입력 토큰: {single_chars}')
print(f'UTF16ByteLevel 결과: {repr(utf16_result)}')
print(f'ByteLevel 결과: {repr(byte_result)}')

print('\n3. 빈 토큰 리스트 테스트')
empty_tokens = []
utf16_result = utf16_decoder.decode(empty_tokens)
byte_result = byte_decoder.decode(empty_tokens)
print(f'입력 토큰: {empty_tokens}')
print(f'UTF16ByteLevel 결과: {repr(utf16_result)}')
print(f'ByteLevel 결과: {repr(byte_result)}')

print('\n4. 특수 문자 테스트')
special_tokens = ['!', '@', '#', '$', '%']
utf16_result = utf16_decoder.decode(special_tokens)
byte_result = byte_decoder.decode(special_tokens)
print(f'입력 토큰: {special_tokens}')
print(f'UTF16ByteLevel 결과: {repr(utf16_result)}')
print(f'ByteLevel 결과: {repr(byte_result)}')

print('\n5. 숫자 테스트')
number_tokens = ['1', '2', '3', '4', '5']
utf16_result = utf16_decoder.decode(number_tokens)
byte_result = byte_decoder.decode(number_tokens)
print(f'입력 토큰: {number_tokens}')
print(f'UTF16ByteLevel 결과: {repr(utf16_result)}')
print(f'ByteLevel 결과: {repr(byte_result)}')

print('\n6. 디코더 속성 확인')
print(f'UTF16ByteLevel 디코더 문자열 표현: {utf16_decoder}')
print(f'ByteLevel 디코더 문자열 표현: {byte_decoder}')

print('\n=== 상세 테스트 완료 ===') 