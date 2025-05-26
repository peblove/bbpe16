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
sys.path.insert(0, 'bindings/python')

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
        return None
    
    print(f"✓ Training file found: {training_file}")
    print(f"  Size: {training_file.stat().st_size / (1024*1024):.2f} MB")
    
    # Create output directories
    results_dir = Path("~/work/data/utf16_tokenizer_evaluation/results").expanduser()
    bl_vocab_dir = results_dir / "byte_level_vocab"
    utf16_vocab_dir = results_dir / "utf16_byte_level_vocab"
    
    bl_vocab_dir.mkdir(exist_ok=True)
    utf16_vocab_dir.mkdir(exist_ok=True)
    
    vocab_size = 1000
    
    try:
        # Train ByteLevel BPE
        print(f"\n🔄 Training ByteLevelBPETokenizer (vocab_size={vocab_size})...")
        start_time = time.time()
        
        bl_tokenizer = ByteLevelBPETokenizer()
        bl_tokenizer.train([str(training_file)], vocab_size=vocab_size, min_frequency=1)
        bl_tokenizer.save_model(str(bl_vocab_dir))
        
        bl_train_time = time.time() - start_time
        print(f"✓ ByteLevel training completed in {bl_train_time:.2f} seconds")
        
        # Train UTF16ByteLevel BPE
        print(f"\n🔄 Training UTF16ByteLevelBPETokenizer (vocab_size={vocab_size})...")
        start_time = time.time()
        
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        utf16_tokenizer.train([str(training_file)], vocab_size=vocab_size, min_frequency=1)
        utf16_tokenizer.save_model(str(utf16_vocab_dir))
        
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
        
        # Calculate improvement
        if bl_avg > 0:
            improvement = (bl_avg - utf16_avg) / bl_avg * 100
        else:
            improvement = 0
        
        results[language] = {
            'bl_avg_tokens': bl_avg,
            'utf16_avg_tokens': utf16_avg,
            'bl_errors': bl_errors,
            'utf16_errors': utf16_errors,
            'improvement_percent': improvement,
            'test_count': len(texts)
        }
        
        print(f"  ByteLevel BPE: {bl_avg:.1f} avg tokens, {bl_errors} errors")
        print(f"  UTF16ByteLevel BPE: {utf16_avg:.1f} avg tokens, {utf16_errors} errors")
        
        if improvement > 0:
            print(f"  🎉 UTF16 improvement: {improvement:.1f}% fewer tokens")
        elif improvement < 0:
            print(f"  📈 ByteLevel advantage: {-improvement:.1f}% fewer tokens")
        else:
            print(f"  ⚖️  Equal performance")
    
    return results

def main():
    """Main test function"""
    print("🚀 FINAL TEST BEFORE GIT PUSH")
    print("UTF16ByteLevelBPETokenizer Implementation")
    print("Author: Hyunsik Kim <avantkim@gmail.com>")
    print("Date: May 2025")
    
    # Step 1: Verify implementation
    if not verify_implementation():
        print("\n❌ Implementation verification failed!")
        return False
    
    # Step 2: Train tokenizers
    tokenizers_info = train_tokenizers()
    if not tokenizers_info:
        print("\n❌ Tokenizer training failed!")
        return False
    
    # Step 3: Performance comparison
    performance_results = performance_comparison(tokenizers_info)
    
    print("\n" + "="*80)
    print("🎉 FINAL TEST COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("✅ Implementation verified")
    print("✅ Tokenizers trained successfully")
    print("✅ Performance comparison completed")
    print("\n🚀 Ready for git push!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 