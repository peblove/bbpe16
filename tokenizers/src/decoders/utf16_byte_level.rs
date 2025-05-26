use crate::tokenizer::{Decoder, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use once_cell::sync::Lazy;

/// UTF-16 byte level decoder
/// 
/// This decoder is to be used in tandem with the UTF16ByteLevel PreTokenizer.
/// It converts UTF-16 byte-level tokens back to their original UTF-8 string representation.
/// 
/// Author: Hyunsik Kim <avantkim@gmail.com>
/// Date: May 2025
/// 
/// This implementation is based on the original ByteLevel decoder from the tokenizers library
/// but adapted to work with UTF-16 encoding instead of UTF-8.

/// Converts UTF-16 bytes to unicode characters for UTF-16 byte level encoding.
/// Same mapping as used in the pre-tokenizer.
fn utf16_bytes_char() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = vec![];
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');

    let mut cs: Vec<u32> = bs.iter().map(|i| *i as u32).collect();
    let mut n = 0;

    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(u32::pow(2, 8) + n);
            n += 1;
        }
    }

    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

/// Converts UTF-16 bytes back to UTF-8 string
fn utf16_bytes_to_utf8(bytes: &[u8]) -> Result<String> {
    if bytes.len() % 2 != 0 {
        return Err("Invalid UTF-16 byte sequence: odd number of bytes".into());
    }
    
    let mut utf16_units = Vec::with_capacity(bytes.len() / 2);
    
    for chunk in bytes.chunks_exact(2) {
        // Little-endian decoding
        let unit = u16::from_le_bytes([chunk[0], chunk[1]]);
        utf16_units.push(unit);
    }
    
    String::from_utf16(&utf16_units)
        .map_err(|e| format!("Invalid UTF-16 sequence: {}", e).into())
}

static CHAR_UTF16_BYTES: Lazy<HashMap<char, u8>> =
    Lazy::new(|| utf16_bytes_char().into_iter().map(|(c, b)| (b, c)).collect());

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
/// UTF16ByteLevel Decoder
/// 
/// This decoder is to be used in tandem with the UTF16ByteLevel PreTokenizer.
/// It converts UTF-16 byte-level character representations back to their original UTF-8 strings.
pub struct UTF16ByteLevel;

impl Default for UTF16ByteLevel {
    fn default() -> Self {
        Self
    }
}

impl Decoder for UTF16ByteLevel {
    fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>> {
        let utf16_bytes = tokens
            .into_iter()
            .flat_map(|t| {
                t.chars()
                    .try_fold(vec![], |mut acc, c| {
                        CHAR_UTF16_BYTES.get(&c).map(|b| {
                            acc.push(*b);
                            acc
                        })
                    })
                    .unwrap_or_else(|| t.as_bytes().to_vec())
            })
            .collect::<Vec<u8>>();
            
        // Convert UTF-16 bytes back to UTF-8 string
        let decoded = utf16_bytes_to_utf8(&utf16_bytes)?;
        Ok(vec![decoded])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf16_bytes_to_utf8() {
        // Test ASCII
        let bytes = vec![0x48, 0x00, 0x65, 0x00, 0x6C, 0x00, 0x6C, 0x00, 0x6F, 0x00];
        let result = utf16_bytes_to_utf8(&bytes).unwrap();
        assert_eq!(result, "Hello");
        
        // Test Korean characters
        let bytes = vec![0x00, 0xAC, 0x98, 0xB0, 0xE4, 0xB2];
        let result = utf16_bytes_to_utf8(&bytes).unwrap();
        assert_eq!(result, "가나다");
    }

    #[test]
    fn test_decoder() {
        let decoder = UTF16ByteLevel::default();
        
        // Test with simple ASCII tokens
        // This would be the result of encoding "Hello" through UTF16ByteLevel pre-tokenizer
        // Each byte of the UTF-16 representation gets mapped to a character
        let tokens = vec!["H".to_string(), "\u{0100}".to_string()]; // Example tokens
        
        // Note: In practice, the tokens would be the character representations
        // of the UTF-16 bytes, but this is a simplified test
        let result = decoder.decode_chain(tokens);
        assert!(result.is_ok());
    }
} 