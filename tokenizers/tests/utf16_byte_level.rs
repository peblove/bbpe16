use tokenizers::pre_tokenizers::utf16_byte_level::UTF16ByteLevel;
use tokenizers::pre_tokenizers::PreTokenizer;
use tokenizers::decoders::utf16_byte_level::UTF16ByteLevel as UTF16ByteLevelDecoder;
use tokenizers::decoders::Decoder;
use tokenizers::{Encoding, InputSequence, OffsetReferential, OffsetType};

#[test]
fn test_utf8_to_utf16_conversion() {
    // Test basic ASCII
    let ascii_text = "Hello";
    let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(ascii_text);
    assert_eq!(utf16_bytes, vec![72, 0, 101, 0, 108, 0, 108, 0, 111, 0]);
    
    // Test Korean characters
    let korean_text = "가나다";
    let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(korean_text);
    // 가 = U+AC00 = 0xAC00 = [0x00, 0xAC] in little-endian
    // 나 = U+B098 = 0xB098 = [0x98, 0xB0] in little-endian  
    // 다 = U+B2E4 = 0xB2E4 = [0xE4, 0xB2] in little-endian
    assert_eq!(utf16_bytes, vec![0x00, 0xAC, 0x98, 0xB0, 0xE4, 0xB2]);
    
    // Test emoji (surrogate pair)
    let emoji_text = "😀";
    let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(emoji_text);
    // 😀 = U+1F600 = surrogate pair: 0xD83D 0xDE00
    assert_eq!(utf16_bytes, vec![0x3D, 0xD8, 0x00, 0xDE]);
}

#[test]
fn test_utf16_to_utf8_conversion() {
    // Test basic ASCII
    let utf16_bytes = vec![72, 0, 101, 0, 108, 0, 108, 0, 111, 0];
    let utf8_text = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
    assert_eq!(utf8_text, "Hello");
    
    // Test Korean characters
    let utf16_bytes = vec![0x00, 0xAC, 0x98, 0xB0, 0xE4, 0xB2];
    let utf8_text = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
    assert_eq!(utf8_text, "가나다");
    
    // Test emoji (surrogate pair)
    let utf16_bytes = vec![0x3D, 0xD8, 0x00, 0xDE];
    let utf8_text = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
    assert_eq!(utf8_text, "😀");
}

#[test]
fn test_roundtrip_conversion() {
    let test_cases = vec![
        "Hello World",
        "가나다라마바사",
        "你好世界",
        "こんにちは",
        "🌍🚀💻",
        "Mixed: 한글 English 中文 😊",
        "",
        "A",
        "🏳️‍🌈", // Complex emoji with ZWJ
    ];
    
    for text in test_cases {
        let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(text);
        let recovered = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
        assert_eq!(text, recovered, "Roundtrip failed for: {}", text);
    }
}

#[test]
fn test_alphabet_mapping() {
    let utf16_byte_level = UTF16ByteLevel::new(true, true);
    
    // Test that all 256 bytes map to valid characters
    for byte in 0..=255u8 {
        let mapped_char = utf16_byte_level.byte_to_char(byte);
        assert!(mapped_char.is_some(), "Byte {} should map to a character", byte);
        
        // Test reverse mapping
        if let Some(ch) = mapped_char {
            let mapped_byte = utf16_byte_level.char_to_byte(ch);
            assert_eq!(mapped_byte, Some(byte), "Character {} should map back to byte {}", ch, byte);
        }
    }
}

#[test]
fn test_pre_tokenizer() {
    let pre_tokenizer = UTF16ByteLevel::new(true, true);
    
    // Test Korean text
    let korean_text = "가나다";
    let mut encoding = Encoding::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        None,
    );
    
    pre_tokenizer.pre_tokenize(&mut encoding).unwrap();
    
    // Should have one token representing the UTF-16 byte sequence
    assert_eq!(encoding.get_tokens().len(), 1);
    
    // Test mixed text
    let mixed_text = "Hello 가나다";
    let mut encoding = Encoding::new(
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        vec![],
        None,
    );
    
    pre_tokenizer.pre_tokenize(&mut encoding).unwrap();
    assert_eq!(encoding.get_tokens().len(), 1);
}

#[test]
fn test_decoder() {
    let decoder = UTF16ByteLevelDecoder::new();
    
    // Test decoding Korean characters
    let korean_tokens = vec!["Ā¬ĺ".to_string(), "°ä".to_string(), "²".to_string()];
    let decoded = decoder.decode(korean_tokens).unwrap();
    assert_eq!(decoded, "가나다");
    
    // Test decoding ASCII
    let ascii_tokens = vec!["H".to_string(), "e".to_string(), "l".to_string(), "l".to_string(), "o".to_string()];
    let decoded = decoder.decode(ascii_tokens).unwrap();
    assert_eq!(decoded, "Hello");
}

#[test]
fn test_error_handling() {
    // Test invalid UTF-16 bytes (odd number of bytes)
    let invalid_utf16_bytes = vec![0x00, 0xAC, 0x98]; // Incomplete UTF-16 sequence
    let result = UTF16ByteLevel::utf16_bytes_to_utf8(&invalid_utf16_bytes);
    assert!(result.is_err(), "Should fail with incomplete UTF-16 sequence");
    
    // Test invalid surrogate pairs
    let invalid_surrogate = vec![0x3D, 0xD8]; // High surrogate without low surrogate
    let result = UTF16ByteLevel::utf16_bytes_to_utf8(&invalid_surrogate);
    assert!(result.is_err(), "Should fail with incomplete surrogate pair");
}

#[test]
fn test_special_characters() {
    let special_chars = vec![
        "\n",      // Newline
        "\t",      // Tab
        "\r",      // Carriage return
        " ",       // Space
        "\u{0000}", // Null character
        "\u{FEFF}", // BOM (should be handled correctly)
    ];
    
    for ch in special_chars {
        let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(ch);
        let recovered = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
        assert_eq!(ch, recovered, "Special character roundtrip failed: {:?}", ch);
    }
}

#[test]
fn test_performance_characteristics() {
    // Test with large text to ensure reasonable performance
    let large_text = "가나다라마바사아자차카타파하".repeat(1000);
    
    let start = std::time::Instant::now();
    let utf16_bytes = UTF16ByteLevel::utf8_to_utf16_bytes(&large_text);
    let conversion_time = start.elapsed();
    
    let start = std::time::Instant::now();
    let recovered = UTF16ByteLevel::utf16_bytes_to_utf8(&utf16_bytes).unwrap();
    let recovery_time = start.elapsed();
    
    assert_eq!(large_text, recovered);
    
    // Performance should be reasonable (less than 100ms for this size)
    assert!(conversion_time.as_millis() < 100, "UTF-8 to UTF-16 conversion too slow: {}ms", conversion_time.as_millis());
    assert!(recovery_time.as_millis() < 100, "UTF-16 to UTF-8 conversion too slow: {}ms", recovery_time.as_millis());
} 