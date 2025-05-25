#!/usr/bin/env python3
"""
Test script for UTF16ByteLevelBPETokenizer

Author: Hyunsik Kim <avantkim@gmail.com>
Date: May 2025

This script tests the UTF16ByteLevelBPETokenizer implementation to ensure it works correctly
with various text inputs, especially Korean text.
"""

import sys
import os

# Add the tokenizers package to the path
sys.path.insert(0, 'bindings/python')

try:
    from tokenizers.implementations import UTF16ByteLevelBPETokenizer
    from tokenizers import pre_tokenizers
    print("✓ Successfully imported UTF16ByteLevelBPETokenizer")
except ImportError as e:
    print(f"✗ Failed to import UTF16ByteLevelBPETokenizer: {e}")
    sys.exit(1)

def test_alphabet():
    """Test that the alphabet has 256 characters"""
    print("\n=== Testing Alphabet ===")
    alphabet = pre_tokenizers.UTF16ByteLevel.alphabet()
    print(f"Alphabet size: {len(alphabet)}")
    assert len(alphabet) == 256, f"Expected 256 characters, got {len(alphabet)}"
    print("✓ Alphabet test passed")

def test_basic_functionality():
    """Test basic tokenizer functionality"""
    print("\n=== Testing Basic Functionality ===")
    
    # Create tokenizer
    tokenizer = UTF16ByteLevelBPETokenizer()
    print("✓ Tokenizer created successfully")
    
    # Test Korean text
    korean_text = "가나다"
    print(f"Testing Korean text: '{korean_text}'")
    
    # Test encoding (this will work even without training for basic functionality)
    try:
        encoding = tokenizer.encode(korean_text)
        print(f"✓ Encoding successful: {len(encoding.tokens)} tokens")
        print(f"  Tokens: {encoding.tokens}")
        print(f"  IDs: {encoding.ids}")
        
        # Test decoding
        decoded = tokenizer.decode(encoding.ids)
        print(f"✓ Decoding successful: '{decoded}'")
        
        # Check roundtrip
        if decoded == korean_text:
            print("✓ Roundtrip test passed")
        else:
            print(f"✗ Roundtrip test failed: '{korean_text}' != '{decoded}'")
            
    except Exception as e:
        print(f"✗ Encoding/decoding failed: {e}")

def test_with_training_data():
    """Test with small training dataset"""
    print("\n=== Testing with Training Data ===")
    
    # Create some sample training data
    training_data = [
        "Hello world",
        "안녕하세요",
        "가나다라마바사",
        "Hello 안녕하세요",
        "This is a test",
        "이것은 테스트입니다",
        "UTF-16 byte level tokenization",
        "UTF-16 바이트 레벨 토크나이제이션"
    ]
    
    # Write training data to a temporary file
    train_file = "/tmp/utf16_train.txt"
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in training_data:
            f.write(line + '\n')
    
    try:
        # Create and train tokenizer
        tokenizer = UTF16ByteLevelBPETokenizer()
        print("Training tokenizer...")
        tokenizer.train([train_file], vocab_size=1000, min_frequency=1)
        print("✓ Training completed")
        
        # Test the trained tokenizer
        test_texts = [
            "Hello",
            "안녕",
            "가나다",
            "Hello 안녕"
        ]
        
        for text in test_texts:
            encoding = tokenizer.encode(text)
            decoded = tokenizer.decode(encoding.ids)
            status = "✓" if decoded == text else "✗"
            print(f"{status} '{text}' -> {len(encoding.tokens)} tokens -> '{decoded}'")
            
    except Exception as e:
        print(f"✗ Training test failed: {e}")
    finally:
        # Clean up
        if os.path.exists(train_file):
            os.remove(train_file)

def main():
    """Run all tests"""
    print("Testing UTF16ByteLevelBPETokenizer Implementation")
    print("=" * 50)
    
    try:
        test_alphabet()
        test_basic_functionality()
        test_with_training_data()
        
        print("\n" + "=" * 50)
        print("✓ All tests completed!")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 