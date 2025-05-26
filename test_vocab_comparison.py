#!/usr/bin/env python3
"""
Test script to compare vocab usage between ByteLevel and UTF16ByteLevel tokenizers

Author: Hyunsik Kim <avantkim@gmail.com>
Date: May 2025
"""

import sys
import os
sys.path.insert(0, 'bindings/python')

from tokenizers.implementations import ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer

def main():
    # Load trained tokenizers
    print("Loading trained tokenizers...")
    
    bl_tokenizer = ByteLevelBPETokenizer()
    bl_tokenizer = bl_tokenizer.from_file(
        '/home/avantkim/work/data/utf16_tokenizer_evaluation/results/byte_level_vocab/vocab.json',
        '/home/avantkim/work/data/utf16_tokenizer_evaluation/results/byte_level_vocab/merges.txt'
    )

    utf16_tokenizer = UTF16ByteLevelBPETokenizer()
    utf16_tokenizer = utf16_tokenizer.from_file(
        '/home/avantkim/work/data/utf16_tokenizer_evaluation/results/utf16_byte_level_vocab/vocab.json',
        '/home/avantkim/work/data/utf16_tokenizer_evaluation/results/utf16_byte_level_vocab/merges.txt'
    )

    # Test texts
    test_texts = [
        "안녕하세요",
        "가나다",
        "Hello World",
        "你好世界",
        "안녕하세요 Hello 你好"
    ]

    print("\n" + "="*80)
    print("TOKENIZATION COMPARISON")
    print("="*80)

    for text in test_texts:
        print(f"\nText: '{text}'")
        print("-" * 50)
        
        # ByteLevel BPE
        bl_encoded = bl_tokenizer.encode(text)
        print(f"ByteLevel BPE ({len(bl_encoded.tokens)} tokens):")
        print(f"  Tokens: {bl_encoded.tokens}")
        print(f"  IDs: {bl_encoded.ids}")
        
        # UTF16ByteLevel BPE
        utf16_encoded = utf16_tokenizer.encode(text)
        print(f"UTF16ByteLevel BPE ({len(utf16_encoded.tokens)} tokens):")
        print(f"  Tokens: {utf16_encoded.tokens}")
        print(f"  IDs: {utf16_encoded.ids}")
        
        # Compare
        token_diff = len(bl_encoded.tokens) - len(utf16_encoded.tokens)
        if token_diff > 0:
            print(f"  → UTF16 uses {token_diff} fewer tokens ({token_diff/len(bl_encoded.tokens)*100:.1f}% reduction)")
        elif token_diff < 0:
            print(f"  → ByteLevel uses {-token_diff} fewer tokens ({-token_diff/len(utf16_encoded.tokens)*100:.1f}% reduction)")
        else:
            print(f"  → Same number of tokens")

if __name__ == "__main__":
    main() 