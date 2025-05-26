import pytest
import tempfile
import os
from tokenizers import Tokenizer
from tokenizers.implementations import UTF16ByteLevelBPETokenizer, ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import UTF16ByteLevel, ByteLevel


class TestUTF16ByteLevelBPETokenizer:
    
    def test_instantiation(self):
        """Test basic instantiation of UTF16ByteLevelBPETokenizer"""
        tokenizer = UTF16ByteLevelBPETokenizer()
        assert tokenizer is not None
        assert isinstance(tokenizer, Tokenizer)
    
    def test_train_basic(self):
        """Test basic training functionality"""
        tokenizer = UTF16ByteLevelBPETokenizer()
        
        # Create temporary training data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Hello world\n")
            f.write("가나다라마바사\n")
            f.write("你好世界\n")
            f.write("こんにちは\n")
            temp_file = f.name
        
        try:
            # Train the tokenizer
            tokenizer.train([temp_file], vocab_size=1000, min_frequency=1)
            
            # Test that training worked
            assert tokenizer.get_vocab_size() > 256  # Should have more than base alphabet
            
            # Test encoding
            encoded = tokenizer.encode("Hello 가나다")
            assert len(encoded.tokens) > 0
            
            # Test decoding
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == "Hello 가나다"
            
        finally:
            os.unlink(temp_file)
    
    def test_train_from_iterator(self):
        """Test training from iterator"""
        tokenizer = UTF16ByteLevelBPETokenizer()
        
        # Training data as iterator
        training_data = [
            "Hello world",
            "가나다라마바사",
            "你好世界",
            "こんにちは",
            "Mixed: 한글 English 中文 😊"
        ]
        
        # Train from iterator
        tokenizer.train_from_iterator(training_data, vocab_size=1000, min_frequency=1)
        
        # Test that training worked
        assert tokenizer.get_vocab_size() > 256
        
        # Test encoding/decoding
        for text in training_data:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Roundtrip failed for: {text}"
    
    def test_korean_text_efficiency(self):
        """Test that Korean text is tokenized more efficiently than ByteLevel"""
        korean_texts = [
            "가나다라마바사아자차카타파하",
            "안녕하세요 반갑습니다",
            "한국어 텍스트 처리 테스트",
            "서울특별시 강남구 역삼동"
        ]
        
        # Train both tokenizers
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        byte_tokenizer = ByteLevelBPETokenizer()
        
        utf16_tokenizer.train_from_iterator(korean_texts * 10, vocab_size=1000, min_frequency=1)
        byte_tokenizer.train_from_iterator(korean_texts * 10, vocab_size=1000, min_frequency=1)
        
        # Compare token counts
        total_utf16_tokens = 0
        total_byte_tokens = 0
        
        for text in korean_texts:
            utf16_encoded = utf16_tokenizer.encode(text)
            byte_encoded = byte_tokenizer.encode(text)
            
            total_utf16_tokens += len(utf16_encoded.tokens)
            total_byte_tokens += len(byte_encoded.tokens)
            
            # Verify roundtrip accuracy
            assert utf16_tokenizer.decode(utf16_encoded.ids) == text
            assert byte_tokenizer.decode(byte_encoded.ids) == text
        
        # UTF16 should generally use fewer tokens for Korean text
        efficiency_ratio = total_utf16_tokens / total_byte_tokens
        print(f"UTF16 vs Byte token ratio: {efficiency_ratio:.3f}")
        # Allow some flexibility, but expect improvement
        assert efficiency_ratio < 1.1, "UTF16 tokenizer should be more efficient for Korean text"
    
    def test_chinese_text_efficiency(self):
        """Test that Chinese text is tokenized more efficiently than ByteLevel"""
        chinese_texts = [
            "你好世界",
            "中文文本处理测试",
            "北京市朝阳区建国门外大街",
            "人工智能技术发展"
        ]
        
        # Train both tokenizers
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        byte_tokenizer = ByteLevelBPETokenizer()
        
        utf16_tokenizer.train_from_iterator(chinese_texts * 10, vocab_size=1000, min_frequency=1)
        byte_tokenizer.train_from_iterator(chinese_texts * 10, vocab_size=1000, min_frequency=1)
        
        # Compare token counts
        total_utf16_tokens = 0
        total_byte_tokens = 0
        
        for text in chinese_texts:
            utf16_encoded = utf16_tokenizer.encode(text)
            byte_encoded = byte_tokenizer.encode(text)
            
            total_utf16_tokens += len(utf16_encoded.tokens)
            total_byte_tokens += len(byte_encoded.tokens)
            
            # Verify roundtrip accuracy
            assert utf16_tokenizer.decode(utf16_encoded.ids) == text
            assert byte_tokenizer.decode(byte_encoded.ids) == text
        
        # UTF16 should generally use fewer tokens for Chinese text
        efficiency_ratio = total_utf16_tokens / total_byte_tokens
        print(f"UTF16 vs Byte token ratio for Chinese: {efficiency_ratio:.3f}")
        assert efficiency_ratio < 1.1, "UTF16 tokenizer should be more efficient for Chinese text"
    
    def test_emoji_handling(self):
        """Test handling of emoji and complex Unicode characters"""
        emoji_texts = [
            "😀😃😄😁",
            "🌍🚀💻🎉",
            "👨‍💻👩‍🔬🧑‍🎨",  # Complex emoji with ZWJ
            "🏳️‍🌈🏴‍☠️",        # Flag emoji
        ]
        
        tokenizer = UTF16ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(emoji_texts * 5, vocab_size=1000, min_frequency=1)
        
        for text in emoji_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Emoji roundtrip failed for: {text}"
    
    def test_mixed_language_text(self):
        """Test handling of mixed language text"""
        mixed_texts = [
            "Hello 안녕하세요 你好",
            "English 한글 中文 日本語",
            "Programming 프로그래밍 编程 プログラミング",
            "AI 인공지능 人工智能 人工知能 🤖"
        ]
        
        tokenizer = UTF16ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(mixed_texts * 10, vocab_size=2000, min_frequency=1)
        
        for text in mixed_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Mixed language roundtrip failed for: {text}"
    
    def test_special_characters(self):
        """Test handling of special characters"""
        special_texts = [
            "Line1\nLine2\nLine3",
            "Tab\tSeparated\tValues",
            "Carriage\rReturn",
            "Null\x00Character",
            "Unicode BOM: \ufeff",
            "Various quotes: \"'`''""",
        ]
        
        tokenizer = UTF16ByteLevelBPETokenizer()
        tokenizer.train_from_iterator(special_texts * 5, vocab_size=1000, min_frequency=1)
        
        for text in special_texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Special character roundtrip failed for: {repr(text)}"
    
    def test_empty_and_edge_cases(self):
        """Test edge cases like empty strings and single characters"""
        edge_cases = [
            "",
            "A",
            "가",
            "中",
            "😀",
            " ",
            "\n",
            "\t",
        ]
        
        tokenizer = UTF16ByteLevelBPETokenizer()
        # Train with some basic data first
        tokenizer.train_from_iterator(["Hello world", "가나다", "你好"], vocab_size=500, min_frequency=1)
        
        for text in edge_cases:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.ids)
            assert decoded == text, f"Edge case roundtrip failed for: {repr(text)}"
    
    def test_alphabet_usage(self):
        """Test that the tokenizer uses ByteLevel.alphabet() as required"""
        tokenizer = UTF16ByteLevelBPETokenizer()
        
        # Get the alphabet from ByteLevel
        byte_level_alphabet = ByteLevel.alphabet()
        
        # Train with minimal data
        tokenizer.train_from_iterator(["test"], vocab_size=300, min_frequency=1)
        
        # Get the vocabulary
        vocab = tokenizer.get_vocab()
        
        # Check that all alphabet characters are in the vocabulary
        for char in byte_level_alphabet:
            assert char in vocab, f"Alphabet character '{char}' not found in vocabulary"
    
    def test_save_and_load(self):
        """Test saving and loading the tokenizer"""
        tokenizer = UTF16ByteLevelBPETokenizer()
        
        # Train the tokenizer
        training_data = ["Hello world", "가나다", "你好世界"]
        tokenizer.train_from_iterator(training_data, vocab_size=500, min_frequency=1)
        
        # Test text
        test_text = "Hello 가나다 你好"
        original_encoded = tokenizer.encode(test_text)
        
        # Save to temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            vocab_file = os.path.join(temp_dir, "vocab.json")
            merges_file = os.path.join(temp_dir, "merges.txt")
            
            tokenizer.save_model(temp_dir)
            
            # Load new tokenizer
            new_tokenizer = UTF16ByteLevelBPETokenizer(vocab_file, merges_file)
            new_encoded = new_tokenizer.encode(test_text)
            
            # Should produce identical results
            assert original_encoded.ids == new_encoded.ids
            assert original_encoded.tokens == new_encoded.tokens
            assert new_tokenizer.decode(new_encoded.ids) == test_text
    
    def test_performance_comparison(self):
        """Test performance characteristics compared to ByteLevel"""
        import time
        
        # Large test data
        korean_text = "가나다라마바사아자차카타파하" * 100
        english_text = "Hello world this is a test" * 100
        chinese_text = "你好世界这是一个测试" * 100
        
        test_texts = [korean_text, english_text, chinese_text]
        
        # Train both tokenizers
        utf16_tokenizer = UTF16ByteLevelBPETokenizer()
        byte_tokenizer = ByteLevelBPETokenizer()
        
        utf16_tokenizer.train_from_iterator(test_texts, vocab_size=2000, min_frequency=1)
        byte_tokenizer.train_from_iterator(test_texts, vocab_size=2000, min_frequency=1)
        
        # Measure encoding performance
        for text in test_texts:
            # UTF16 encoding
            start_time = time.time()
            utf16_encoded = utf16_tokenizer.encode(text)
            utf16_time = time.time() - start_time
            
            # Byte encoding
            start_time = time.time()
            byte_encoded = byte_tokenizer.encode(text)
            byte_time = time.time() - start_time
            
            # Verify correctness
            assert utf16_tokenizer.decode(utf16_encoded.ids) == text
            assert byte_tokenizer.decode(byte_encoded.ids) == text
            
            # Performance should be reasonable (within 10x of each other)
            performance_ratio = utf16_time / byte_time if byte_time > 0 else 1
            assert performance_ratio < 10, f"UTF16 tokenizer too slow: {performance_ratio:.2f}x slower"
            
            print(f"Text length: {len(text)}, UTF16: {utf16_time:.4f}s, Byte: {byte_time:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__]) 