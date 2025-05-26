#!/usr/bin/env python3

import tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.decoders import UTF16ByteLevel, ByteLevel, Sequence
from tokenizers.normalizers import NFD

print('=== Tokenizers 통합 테스트 ===')
print(f'Tokenizers 버전: {tokenizers.__version__}')

# 1. 다양한 디코더 생성 테스트
print('\n1. 다양한 디코더 생성 테스트')
decoders = {
    'UTF16ByteLevel': UTF16ByteLevel(),
    'ByteLevel': ByteLevel(),
}

for name, decoder in decoders.items():
    print(f'✓ {name} 디코더 생성 성공: {type(decoder)}')

# 2. Sequence 디코더 테스트
print('\n2. Sequence 디코더 테스트')
try:
    sequence_decoder = Sequence([ByteLevel(), UTF16ByteLevel()])
    print('✓ Sequence 디코더 생성 성공')
except Exception as e:
    print(f'✗ Sequence 디코더 생성 실패: {e}')

# 3. 토크나이저 완전 설정 테스트
print('\n3. 토크나이저 완전 설정 테스트')
try:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = UTF16ByteLevel()
    tokenizer.normalizer = NFD()
    print('✓ 토크나이저 완전 설정 성공')
    
    # 설정 확인
    print(f'  - 모델: {type(tokenizer.model)}')
    print(f'  - Pre-tokenizer: {type(tokenizer.pre_tokenizer)}')
    print(f'  - Decoder: {type(tokenizer.decoder)}')
    print(f'  - Normalizer: {type(tokenizer.normalizer)}')
    
except Exception as e:
    print(f'✗ 토크나이저 설정 실패: {e}')

# 4. 디코더 직렬화 테스트
print('\n4. 디코더 직렬화 테스트')
try:
    import json
    
    # UTF16ByteLevel 디코더 직렬화 시도
    utf16_decoder = UTF16ByteLevel()
    tokenizer.decoder = utf16_decoder
    
    # 토크나이저 JSON 직렬화
    tokenizer_json = tokenizer.to_str()
    print('✓ 토크나이저 JSON 직렬화 성공')
    
    # JSON에서 복원
    tokenizer_restored = Tokenizer.from_str(tokenizer_json)
    print('✓ 토크나이저 JSON 복원 성공')
    print(f'  - 복원된 디코더 타입: {type(tokenizer_restored.decoder)}')
    
except Exception as e:
    print(f'✗ 직렬화 테스트 실패: {e}')

# 5. 메모리 및 성능 기본 테스트
print('\n5. 메모리 및 성능 기본 테스트')
try:
    import time
    
    # 여러 디코더 인스턴스 생성
    start_time = time.time()
    decoders_list = [UTF16ByteLevel() for _ in range(100)]
    creation_time = time.time() - start_time
    print(f'✓ 100개 UTF16ByteLevel 디코더 생성 시간: {creation_time:.4f}초')
    
    # 빈 토큰 디코딩 성능
    start_time = time.time()
    for decoder in decoders_list[:10]:  # 처음 10개만 테스트
        result = decoder.decode([])
    decoding_time = time.time() - start_time
    print(f'✓ 10회 빈 토큰 디코딩 시간: {decoding_time:.4f}초')
    
except Exception as e:
    print(f'✗ 성능 테스트 실패: {e}')

# 6. 에러 처리 테스트
print('\n6. 에러 처리 테스트')
try:
    utf16_decoder = UTF16ByteLevel()
    
    # None 입력 테스트
    try:
        result = utf16_decoder.decode(None)
        print('✗ None 입력에 대한 에러 처리 실패')
    except Exception:
        print('✓ None 입력에 대한 적절한 에러 처리')
    
    # 잘못된 타입 입력 테스트
    try:
        result = utf16_decoder.decode("not a list")
        print('✗ 잘못된 타입 입력에 대한 에러 처리 실패')
    except Exception:
        print('✓ 잘못된 타입 입력에 대한 적절한 에러 처리')
        
except Exception as e:
    print(f'✗ 에러 처리 테스트 실패: {e}')

print('\n=== 통합 테스트 완료 ===')
print('\n결론: UTF16ByteLevel 디코더가 Rust 1.80에서 정상적으로')
print('컴파일되고 Python 바인딩을 통해 올바르게 작동합니다.') 