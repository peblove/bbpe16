#!/usr/bin/env python3
"""
Comprehensive Comparison: ByteLevelBPETokenizer vs UTF16ByteLevelBPETokenizer

This script trains both tokenizers on the same dataset and compares their performance
on various language tasks including Korean, Chinese, English, emojis, and mixed languages.

Author: Hyunsik Kim <avantkim@gmail.com>
Date: May 2025
"""

import os
import sys
import time
import json
from pathlib import Path

# Add the tokenizers package to the path
sys.path.insert(0, '/home/avantkim/work/tokenizers/bindings/python')

from tokenizers.implementations import ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer
from tokenizers import pre_tokenizers

def verify_alphabet_sizes():
    """Verify that both tokenizers have the correct initial alphabet sizes"""
    print("=== Verifying Initial Alphabet Sizes ===")
    
    # Check ByteLevel alphabet
    byte_level_alphabet = pre_tokenizers.ByteLevel.alphabet()
    print(f"ByteLevel alphabet size: {len(byte_level_alphabet)}")
    
    # Check UTF16ByteLevel alphabet  
    utf16_byte_level_alphabet = pre_tokenizers.UTF16ByteLevel.alphabet()
    print(f"UTF16ByteLevel alphabet size: {len(utf16_byte_level_alphabet)}")
    
    # Both should have 256 characters
    assert len(byte_level_alphabet) == 256, f"ByteLevel alphabet should have 256 chars, got {len(byte_level_alphabet)}"
    assert len(utf16_byte_level_alphabet) == 256, f"UTF16ByteLevel alphabet should have 256 chars, got {len(utf16_byte_level_alphabet)}"
    
    print("✓ Both alphabets have correct size (256 characters)")
    return True

def train_tokenizers(training_file, vocab_size=1000):
    """Train both tokenizers on the same training data"""
    print(f"\n=== Training Tokenizers (vocab_size={vocab_size}) ===")
    
    # Create output directories
    byte_level_dir = Path("~/work/data/byte_level_vocab").expanduser()
    utf16_byte_level_dir = Path("~/work/data/utf16_byte_level_vocab").expanduser()
    
    byte_level_dir.mkdir(exist_ok=True)
    utf16_byte_level_dir.mkdir(exist_ok=True)
    
    # Train ByteLevelBPETokenizer
    print("Training ByteLevelBPETokenizer...")
    byte_level_tokenizer = ByteLevelBPETokenizer()
    byte_level_tokenizer.train([training_file], vocab_size=vocab_size, min_frequency=1)
    
    # Save ByteLevel vocab
    byte_level_vocab_file = byte_level_dir / "vocab.json"
    byte_level_merges_file = byte_level_dir / "merges.txt"
    byte_level_tokenizer.save_model(str(byte_level_dir))
    
    print(f"✓ ByteLevelBPETokenizer saved to: {byte_level_dir}")
    
    # Train UTF16ByteLevelBPETokenizer
    print("Training UTF16ByteLevelBPETokenizer...")
    utf16_tokenizer = UTF16ByteLevelBPETokenizer()
    utf16_tokenizer.train([training_file], vocab_size=vocab_size, min_frequency=1)
    
    # Save UTF16ByteLevel vocab
    utf16_vocab_file = utf16_byte_level_dir / "vocab.json"
    utf16_merges_file = utf16_byte_level_dir / "merges.txt"
    utf16_tokenizer.save_model(str(utf16_byte_level_dir))
    
    print(f"✓ UTF16ByteLevelBPETokenizer saved to: {utf16_byte_level_dir}")
    
    return {
        'byte_level': {
            'tokenizer': byte_level_tokenizer,
            'vocab_file': str(byte_level_vocab_file),
            'merges_file': str(byte_level_merges_file),
            'dir': str(byte_level_dir)
        },
        'utf16_byte_level': {
            'tokenizer': utf16_tokenizer,
            'vocab_file': str(utf16_vocab_file),
            'merges_file': str(utf16_merges_file),
            'dir': str(utf16_byte_level_dir)
        }
    }

def load_test_data():
    """Load test datasets for different languages"""
    print("\n=== Loading Test Datasets ===")
    
    test_data = {}
    test_files = {
        'korean': '~/work/data/test_dataset_korean.txt',
        'english': '~/work/data/test_dataset_english.txt', 
        'chinese': '~/work/data/test_dataset_chinese.txt',
        'mixed': '~/work/data/test_dataset_mixed.txt',
        'emoji': '~/work/data/test_dataset_emoji.txt'
    }
    
    for lang, file_path in test_files.items():
        expanded_path = Path(file_path).expanduser()
        if expanded_path.exists():
            with open(expanded_path, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            test_data[lang] = sentences
            print(f"✓ Loaded {lang}: {len(sentences)} sentences")
        else:
            print(f"✗ Test file not found: {expanded_path}")
    
    # If test files don't exist, create minimal test data
    if not test_data:
        print("Creating minimal test data...")
        test_data = {
            'korean': [
                "안녕하세요, 반갑습니다.",
                "한국어는 아름다운 언어입니다.",
                "김치는 한국의 전통 음식입니다.",
                "서울은 한국의 수도입니다.",
                "한글은 세종대왕이 만드셨습니다."
            ],
            'english': [
                "Hello, how are you today?",
                "The weather is beautiful today.",
                "English is a global language.",
                "Technology is advancing rapidly.",
                "Education is very important."
            ],
            'chinese': [
                "你好，很高兴见到你。",
                "今天天气很好。",
                "中文是一门美丽的语言。",
                "学习中文很有趣。",
                "中国有悠久的历史。"
            ],
            'mixed': [
                "Hello 안녕하세요 你好 🌍",
                "Good morning ☀️ 좋은 아침 早上好",
                "Thank you 감사합니다 谢谢 🙏",
                "🍕 Pizza is delicious 피자는 맛있어요 披萨很好吃",
                "🎵 Music is universal 🎶"
            ],
            'emoji': [
                "😊 Happy coding! 💻",
                "🌍 Hello World! 🌎",
                "🚀 Technology advances 🔬",
                "🎨 Art 🖼️ Painting 🎭 Theater",
                "🍕 Pizza 🍜 Ramen 🥘 Curry"
            ]
        }
        print("✓ Created minimal test data")
    
    return test_data

def evaluate_tokenizer(tokenizer, test_data, tokenizer_name):
    """Evaluate a tokenizer on test data"""
    print(f"\n=== Evaluating {tokenizer_name} ===")
    
    results = {}
    
    for lang, sentences in test_data.items():
        print(f"\nTesting {lang} ({len(sentences)} sentences)...")
        
        lang_results = {
            'total_sentences': len(sentences),
            'successful_roundtrips': 0,
            'total_tokens': 0,
            'avg_tokens_per_sentence': 0,
            'encoding_errors': 0,
            'decoding_errors': 0,
            'roundtrip_errors': 0,
            'examples': []
        }
        
        for i, sentence in enumerate(sentences[:10]):  # Test first 10 sentences for detailed analysis
            try:
                # Encode
                encoding = tokenizer.encode(sentence)
                tokens = encoding.tokens
                token_ids = encoding.ids
                
                # Decode
                decoded = tokenizer.decode(token_ids)
                
                # Check roundtrip accuracy
                is_roundtrip_success = (decoded == sentence)
                if is_roundtrip_success:
                    lang_results['successful_roundtrips'] += 1
                else:
                    lang_results['roundtrip_errors'] += 1
                
                lang_results['total_tokens'] += len(tokens)
                
                # Store example for first 5 sentences
                if i < 5:
                    lang_results['examples'].append({
                        'original': sentence,
                        'tokens': tokens,
                        'token_count': len(tokens),
                        'decoded': decoded,
                        'roundtrip_success': is_roundtrip_success
                    })
                
            except Exception as e:
                print(f"Error processing sentence {i}: {e}")
                if "encoding" in str(e).lower():
                    lang_results['encoding_errors'] += 1
                elif "decoding" in str(e).lower():
                    lang_results['decoding_errors'] += 1
                else:
                    lang_results['roundtrip_errors'] += 1
        
        # Calculate averages
        if lang_results['total_sentences'] > 0:
            lang_results['avg_tokens_per_sentence'] = lang_results['total_tokens'] / min(10, len(sentences))
            lang_results['roundtrip_accuracy'] = lang_results['successful_roundtrips'] / min(10, len(sentences))
        
        results[lang] = lang_results
        
        print(f"  Roundtrip accuracy: {lang_results['roundtrip_accuracy']:.2%}")
        print(f"  Avg tokens per sentence: {lang_results['avg_tokens_per_sentence']:.1f}")
        print(f"  Encoding errors: {lang_results['encoding_errors']}")
        print(f"  Decoding errors: {lang_results['decoding_errors']}")
    
    return results

def compare_results(byte_level_results, utf16_results):
    """Compare results between the two tokenizers"""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON RESULTS")
    print("="*80)
    
    comparison = {}
    
    for lang in byte_level_results.keys():
        if lang in utf16_results:
            bl_result = byte_level_results[lang]
            utf16_result = utf16_results[lang]
            
            comparison[lang] = {
                'byte_level': {
                    'roundtrip_accuracy': bl_result['roundtrip_accuracy'],
                    'avg_tokens': bl_result['avg_tokens_per_sentence'],
                    'encoding_errors': bl_result['encoding_errors'],
                    'decoding_errors': bl_result['decoding_errors']
                },
                'utf16_byte_level': {
                    'roundtrip_accuracy': utf16_result['roundtrip_accuracy'],
                    'avg_tokens': utf16_result['avg_tokens_per_sentence'],
                    'encoding_errors': utf16_result['encoding_errors'],
                    'decoding_errors': utf16_result['decoding_errors']
                }
            }
            
            print(f"\n{lang.upper()} LANGUAGE:")
            print(f"  ByteLevel BPE:")
            print(f"    Roundtrip Accuracy: {bl_result['roundtrip_accuracy']:.2%}")
            print(f"    Avg Tokens/Sentence: {bl_result['avg_tokens_per_sentence']:.1f}")
            print(f"    Encoding Errors: {bl_result['encoding_errors']}")
            print(f"    Decoding Errors: {bl_result['decoding_errors']}")
            
            print(f"  UTF16ByteLevel BPE:")
            print(f"    Roundtrip Accuracy: {utf16_result['roundtrip_accuracy']:.2%}")
            print(f"    Avg Tokens/Sentence: {utf16_result['avg_tokens_per_sentence']:.1f}")
            print(f"    Encoding Errors: {utf16_result['encoding_errors']}")
            print(f"    Decoding Errors: {utf16_result['decoding_errors']}")
            
            # Calculate differences
            acc_diff = utf16_result['roundtrip_accuracy'] - bl_result['roundtrip_accuracy']
            token_diff = utf16_result['avg_tokens_per_sentence'] - bl_result['avg_tokens_per_sentence']
            
            print(f"  COMPARISON:")
            print(f"    Accuracy Difference: {acc_diff:+.2%} (UTF16 vs Byte)")
            print(f"    Token Count Difference: {token_diff:+.1f} (UTF16 vs Byte)")
            
            if acc_diff > 0:
                print(f"    ✓ UTF16ByteLevel has better accuracy for {lang}")
            elif acc_diff < 0:
                print(f"    ✓ ByteLevel has better accuracy for {lang}")
            else:
                print(f"    = Equal accuracy for {lang}")
    
    return comparison

def save_detailed_results(byte_level_results, utf16_results, comparison, output_file):
    """Save detailed results to JSON file"""
    detailed_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'byte_level_results': byte_level_results,
        'utf16_byte_level_results': utf16_results,
        'comparison': comparison,
        'summary': {
            'languages_tested': list(comparison.keys()),
            'total_languages': len(comparison)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Detailed results saved to: {output_file}")

def main():
    """Main comparison function"""
    print("UTF16ByteLevelBPETokenizer vs ByteLevelBPETokenizer Comparison")
    print("="*80)
    
    # Verify alphabet sizes
    verify_alphabet_sizes()
    
    # Use existing training data
    training_file = "~/work/data/multilingual_training_data.txt"
    training_path = Path(training_file).expanduser()
    
    if not training_path.exists():
        print(f"Training file not found: {training_path}")
        print("Please ensure the training data exists.")
        return
    
    print(f"Using training file: {training_path}")
    
    # Train both tokenizers
    tokenizer_info = train_tokenizers(str(training_path), vocab_size=1000)
    
    # Load test data
    test_data = load_test_data()
    
    if not test_data:
        print("No test data available. Exiting.")
        return
    
    # Evaluate ByteLevelBPETokenizer
    byte_level_results = evaluate_tokenizer(
        tokenizer_info['byte_level']['tokenizer'], 
        test_data, 
        "ByteLevelBPETokenizer"
    )
    
    # Evaluate UTF16ByteLevelBPETokenizer
    utf16_results = evaluate_tokenizer(
        tokenizer_info['utf16_byte_level']['tokenizer'], 
        test_data, 
        "UTF16ByteLevelBPETokenizer"
    )
    
    # Compare results
    comparison = compare_results(byte_level_results, utf16_results)
    
    # Save detailed results
    output_file = "~/work/data/tokenizer_comparison_results.json"
    save_detailed_results(byte_level_results, utf16_results, comparison, Path(output_file).expanduser())
    
    print("\n" + "="*80)
    print("VOCAB FILE LOCATIONS:")
    print("="*80)
    print(f"ByteLevel BPE vocab: {tokenizer_info['byte_level']['dir']}")
    print(f"  - vocab.json: {tokenizer_info['byte_level']['vocab_file']}")
    print(f"  - merges.txt: {tokenizer_info['byte_level']['merges_file']}")
    print(f"UTF16ByteLevel BPE vocab: {tokenizer_info['utf16_byte_level']['dir']}")
    print(f"  - vocab.json: {tokenizer_info['utf16_byte_level']['vocab_file']}")
    print(f"  - merges.txt: {tokenizer_info['utf16_byte_level']['merges_file']}")
    
    print("\n✓ Comparison completed successfully!")

if __name__ == "__main__":
    main() 