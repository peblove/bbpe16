#!/usr/bin/env python3
"""
Final comprehensive test before git push

Author: Hyunsik Kim <avantkim@gmail.com>
Date: May 2025
"""

import sys
import os
import time
import json
from pathlib import Path

# Add the tokenizers package to the path
sys.path.insert(0, '../../bindings/python')

from tokenizers.implementations import ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer
from tokenizers import pre_tokenizers

def verify_implementation():
    """Verify that the implementation is working correctly"""
    print("="*80)
    print("STEP 1: IMPLEMENTATION VERIFICATION")
    print("="*80)
    
    try:
        # Test imports
        print("✓ Testing imports...")
        from tokenizers.implementations import UTF16ByteLevelBPETokenizer
        from tokenizers import pre_tokenizers, decoders, processors
        
        # Test alphabet
        print("✓ Testing alphabets...")
        bl_alphabet = pre_tokenizers.ByteLevel.alphabet()
        utf16_alphabet = pre_tokenizers.UTF16ByteLevel.alphabet()
        
        assert len(bl_alphabet) == 256, f"ByteLevel alphabet should have 256 chars, got {len(bl_alphabet)}"
        assert len(utf16_alphabet) == 256, f"UTF16ByteLevel alphabet should have 256 chars, got {len(utf16_alphabet)}"
        
        print(f"  - ByteLevel alphabet: {len(bl_alphabet)} characters")
        print(f"  - UTF16ByteLevel alphabet: {len(utf16_alphabet)} characters")
        
        # Test tokenizer creation
        print("✓ Testing tokenizer creation...")
        bl_tokenizer = ByteLevelBPETokenizer()
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        
        print("✓ All implementation tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Implementation test failed: {e}")
        return False

def train_tokenizers():
    """Train both tokenizers on the same dataset"""
    print("\n" + "="*80)
    print("STEP 2: TOKENIZER TRAINING")
    print("="*80)
    
    # Check training data
    training_file = Path("~/work/data/utf16_tokenizer_evaluation/training_data/multilingual_training_10mb.txt").expanduser()
    if not training_file.exists():
        print(f"✗ Training file not found: {training_file}")
        print("Using simple training data instead...")
        
        # Use simple training data
        training_data = [
            "Hello world",
            "가나다라마바사",
            "你好世界",
            "こんにちは",
            "Mixed: 한글 English 中文 😊"
        ] * 50  # Repeat to have enough data
        
        vocab_size = 500
        
        try:
            # Train ByteLevel BPE
            print(f"\n🔄 Training ByteLevelBPETokenizer (vocab_size={vocab_size})...")
            start_time = time.time()
            
            bl_tokenizer = ByteLevelBPETokenizer()
            bl_tokenizer.train_from_iterator(training_data, vocab_size=vocab_size, min_frequency=1)
            
            bl_train_time = time.time() - start_time
            print(f"✓ ByteLevel training completed in {bl_train_time:.2f} seconds")
            
            # Train UTF16ByteLevel BPE
            print(f"\n🔄 Training UTF16ByteLevelBPETokenizer (vocab_size={vocab_size})...")
            start_time = time.time()
            
            utf16_tokenizer = UTF16ByteLevelBPETokenizer()
            utf16_tokenizer.train_from_iterator(training_data, vocab_size=vocab_size, min_frequency=1)
            
            utf16_train_time = time.time() - start_time
            print(f"✓ UTF16ByteLevel training completed in {utf16_train_time:.2f} seconds")
            
            return {
                'bl_tokenizer': bl_tokenizer,
                'utf16_tokenizer': utf16_tokenizer,
                'bl_train_time': bl_train_time,
                'utf16_train_time': utf16_train_time
            }
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
            return None
    
    print(f"✓ Training file found: {training_file}")
    print(f"  Size: {training_file.stat().st_size / (1024*1024):.2f} MB")
    
    vocab_size = 1000
    
    try:
        # Train ByteLevel BPE
        print(f"\n🔄 Training ByteLevelBPETokenizer (vocab_size={vocab_size})...")
        start_time = time.time()
        
        bl_tokenizer = ByteLevelBPETokenizer()
        bl_tokenizer.train([str(training_file)], vocab_size=vocab_size, min_frequency=1)
        
        bl_train_time = time.time() - start_time
        print(f"✓ ByteLevel training completed in {bl_train_time:.2f} seconds")
        
        # Train UTF16ByteLevel BPE
        print(f"\n🔄 Training UTF16ByteLevelBPETokenizer (vocab_size={vocab_size})...")
        start_time = time.time()
        
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        utf16_tokenizer.train([str(training_file)], vocab_size=vocab_size, min_frequency=1)
        
        utf16_train_time = time.time() - start_time
        print(f"✓ UTF16ByteLevel training completed in {utf16_train_time:.2f} seconds")
        
        return {
            'bl_tokenizer': bl_tokenizer,
            'utf16_tokenizer': utf16_tokenizer,
            'bl_train_time': bl_train_time,
            'utf16_train_time': utf16_train_time
        }
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return None

def performance_comparison(tokenizers_info):
    """Perform comprehensive performance comparison"""
    print("\n" + "="*80)
    print("STEP 3: PERFORMANCE COMPARISON")
    print("="*80)
    
    bl_tokenizer = tokenizers_info['bl_tokenizer']
    utf16_tokenizer = tokenizers_info['utf16_tokenizer']
    
    # Test cases
    test_cases = [
        ("Korean", [
            "안녕하세요",
            "가나다라마바사",
            "한국어 토큰화 테스트",
            "서울특별시 강남구",
            "대한민국 화이팅"
        ]),
        ("Chinese", [
            "你好世界",
            "中文分词测试",
            "北京大学",
            "人工智能技术",
            "自然语言处理"
        ]),
        ("English", [
            "Hello World",
            "Natural Language Processing",
            "Machine Learning",
            "Artificial Intelligence",
            "Deep Learning"
        ]),
        ("Mixed", [
            "안녕하세요 Hello 你好",
            "Korean 한국어 Chinese 中文",
            "AI 인공지능 人工智能",
            "Seoul 서울 北京 Beijing",
            "Technology 기술 技术"
        ])
    ]
    
    results = {}
    
    for language, texts in test_cases:
        print(f"\n📊 Testing {language} language:")
        print("-" * 50)
        
        bl_tokens_total = 0
        utf16_tokens_total = 0
        bl_errors = 0
        utf16_errors = 0
        
        for text in texts:
            try:
                # ByteLevel tokenization
                bl_encoded = bl_tokenizer.encode(text)
                bl_decoded = bl_tokenizer.decode(bl_encoded.ids)
                bl_tokens = len(bl_encoded.tokens)
                bl_tokens_total += bl_tokens
                
                if bl_decoded != text:
                    bl_errors += 1
                    print(f"  ⚠️  ByteLevel roundtrip error: '{text}' -> '{bl_decoded}'")
                
            except Exception as e:
                bl_errors += 1
                bl_tokens_total += 999  # Penalty for errors
                print(f"  ✗ ByteLevel error on '{text}': {e}")
            
            try:
                # UTF16ByteLevel tokenization
                utf16_encoded = utf16_tokenizer.encode(text)
                utf16_decoded = utf16_tokenizer.decode(utf16_encoded.ids)
                utf16_tokens = len(utf16_encoded.tokens)
                utf16_tokens_total += utf16_tokens
                
                if utf16_decoded != text:
                    utf16_errors += 1
                    print(f"  ⚠️  UTF16ByteLevel roundtrip error: '{text}' -> '{utf16_decoded}'")
                
            except Exception as e:
                utf16_errors += 1
                utf16_tokens_total += 999  # Penalty for errors
                print(f"  ✗ UTF16ByteLevel error on '{text}': {e}")
        
        # Calculate averages
        bl_avg = bl_tokens_total / len(texts)
        utf16_avg = utf16_tokens_total / len(texts)
        improvement = ((bl_avg - utf16_avg) / bl_avg) * 100 if bl_avg > 0 else 0
        
        results[language.lower()] = {
            'bl_avg': bl_avg,
            'utf16_avg': utf16_avg,
            'improvement': improvement,
            'bl_errors': bl_errors,
            'utf16_errors': utf16_errors
        }
        
        print(f"  ByteLevel avg: {bl_avg:.1f} tokens")
        print(f"  UTF16ByteLevel avg: {utf16_avg:.1f} tokens")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Errors: BL={bl_errors}, UTF16={utf16_errors}")
    
    return results

def main():
    """Main test function"""
    print("🚀 UTF16ByteLevelBPE Final Test")
    print("Author: Hyunsik Kim <avantkim@gmail.com>")
    print("=" * 80)
    
    # Step 1: Verify implementation
    if not verify_implementation():
        print("❌ Implementation verification failed!")
        return 1
    
    # Step 2: Train tokenizers
    tokenizers_info = train_tokenizers()
    if not tokenizers_info:
        print("❌ Tokenizer training failed!")
        return 1
    
    # Step 3: Performance comparison
    results = performance_comparison(tokenizers_info)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total_errors = sum(r['bl_errors'] + r['utf16_errors'] for r in results.values())
    cjk_improvements = [results['korean']['improvement'], results['chinese']['improvement']]
    avg_cjk_improvement = sum(cjk_improvements) / len(cjk_improvements)
    
    print(f"✓ Total errors: {total_errors}")
    print(f"✓ Average CJK improvement: {avg_cjk_improvement:.1f}%")
    print(f"✓ Training time - ByteLevel: {tokenizers_info['bl_train_time']:.2f}s")
    print(f"✓ Training time - UTF16ByteLevel: {tokenizers_info['utf16_train_time']:.2f}s")
    
    if total_errors == 0 and avg_cjk_improvement > 0:
        print("\n🎉 ALL TESTS PASSED! Implementation is ready for production.")
        return 0
    else:
        print("\n⚠️  Some issues detected. Please review the results.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 