use std::collections::{HashMap, HashSet};
use once_cell::sync::Lazy;

use crate::utils::SysRegex;
use serde::{Deserialize, Serialize};

use crate::tokenizer::{
    Decoder, Encoding, PostProcessor, PreTokenizedString, PreTokenizer, Result,
    SplitDelimiterBehavior,
};
use crate::utils::macro_rules_attribute;

/// Converts UTF-16 bytes to unicode characters for UTF-16 byte level encoding.
/// Similar to GPT-2's byte level encoding but operates on UTF-16 bytes instead of UTF-8 bytes.
/// 
/// Author: Hyunsik Kim <avantkim@gmail.com>
/// Date: May 2025
/// 
/// This implementation is based on the original ByteLevel tokenizer from the tokenizers library
/// but adapted to work with UTF-16 encoding instead of UTF-8.
/// 
/// Reference: https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
pub(crate) fn utf16_bytes_char() -> HashMap<u8, char> {
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

    // Safety: cs contains all values from bs (between 0 and 255),
    // and some values of value 2⁸ + n, where n is between 0 and 255. This is between 255 and 512.
    // Both ranges are valid UTF-32 values (which is fully saturated until 0xD000)
    bs.into_iter()
        .zip(cs)
        .map(|(f, t)| (f, unsafe { std::char::from_u32_unchecked(t) }))
        .collect()
}

/// Converts UTF-8 string to UTF-16 bytes (little-endian, no BOM)
/// 
/// This function takes a UTF-8 string and converts it to UTF-16 little-endian encoding
/// without a BOM (Byte Order Mark). The resulting bytes are what will be processed
/// by the BPE algorithm.
/// 
/// Args:
///     text: UTF-8 string to convert
/// 
/// Returns:
///     Vector of bytes representing the UTF-16 little-endian encoding
pub(crate) fn utf8_to_utf16_bytes(text: &str) -> Vec<u8> {
    let utf16_units: Vec<u16> = text.encode_utf16().collect();
    let mut bytes = Vec::with_capacity(utf16_units.len() * 2);
    
    for unit in utf16_units {
        // Little-endian encoding
        bytes.push((unit & 0xFF) as u8);
        bytes.push((unit >> 8) as u8);
    }
    
    bytes
}

/// Converts UTF-16 bytes back to UTF-8 string
/// 
/// This function takes UTF-16 bytes (little-endian, no BOM) and converts them back
/// to a UTF-8 string. This is used during decoding.
/// 
/// Args:
///     bytes: Vector of bytes representing UTF-16 little-endian encoding
/// 
/// Returns:
///     Result containing the UTF-8 string or an error if the bytes are invalid
pub(crate) fn utf16_bytes_to_utf8(bytes: &[u8]) -> Result<String> {
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

/// Regex that matches exactly one token.
/// Same as the original ByteLevel tokenizer regex from GPT-2
static RE: Lazy<SysRegex> = Lazy::new(|| {
    SysRegex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .unwrap()
});

static UTF16_BYTES_CHAR: Lazy<HashMap<u8, char>> = Lazy::new(utf16_bytes_char);
static CHAR_UTF16_BYTES: Lazy<HashMap<char, u8>> =
    Lazy::new(|| utf16_bytes_char().into_iter().map(|(c, b)| (b, c)).collect());

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
/// Provides all the necessary steps to handle the BPE tokenization at the UTF-16 byte-level.
/// Takes care of all the required processing steps to transform a UTF-8 string to UTF-16 bytes
/// as needed before and after the BPE model does its job.
/// 
/// Author: Hyunsik Kim <avantkim@gmail.com>
/// Date: May 2025
/// 
/// This is based on the original ByteLevel tokenizer but adapted for UTF-16 encoding.
#[macro_rules_attribute(impl_serde_type!)]
#[non_exhaustive]
pub struct UTF16ByteLevel {
    /// Whether to add a leading space to the first word. This allows to treat the leading word
    /// just as any other word.
    pub add_prefix_space: bool,
    /// Whether the post processing step should trim offsets to avoid including whitespaces.
    pub trim_offsets: bool,

    /// Whether to use the standard GPT2 regex for whitespace splitting
    /// Set it to False if you want to use your own splitting.
    #[serde(default = "default_true")]
    pub use_regex: bool,
}

fn default_true() -> bool {
    true
}

impl Default for UTF16ByteLevel {
    fn default() -> Self {
        Self {
            add_prefix_space: true,
            trim_offsets: true,
            use_regex: true,
        }
    }
}

impl UTF16ByteLevel {
    pub fn new(add_prefix_space: bool, trim_offsets: bool, use_regex: bool) -> Self {
        Self {
            add_prefix_space,
            trim_offsets,
            use_regex,
        }
    }

    /// Returns the alphabet used by this PreTokenizer.
    /// Since UTF16ByteLevel works at the byte level on UTF-16 encoded text,
    /// it encodes each byte value to a unique visible character.
    /// This means that there is a total of 256 different characters composing this alphabet.
    pub fn alphabet() -> HashSet<char> {
        UTF16_BYTES_CHAR.values().copied().collect()
    }

    #[must_use]
    pub fn add_prefix_space(mut self, v: bool) -> Self {
        self.add_prefix_space = v;
        self
    }

    #[must_use]
    pub fn trim_offsets(mut self, v: bool) -> Self {
        self.trim_offsets = v;
        self
    }

    #[must_use]
    pub fn use_regex(mut self, v: bool) -> Self {
        self.use_regex = v;
        self
    }
}

/// As a `PreTokenizer`, `UTF16ByteLevel` is in charge of transforming all the unicode characters
/// into their UTF-16 byte-level counterpart. It also splits the input according to the configured regex.
impl PreTokenizer for UTF16ByteLevel {
    fn pre_tokenize(&self, pretokenized: &mut PreTokenizedString) -> Result<()> {
        let re_ref: &SysRegex = &RE;
        pretokenized.split(|_, mut normalized| {
            if self.add_prefix_space && !normalized.get().starts_with(' ') {
                normalized.prepend(" ");
            }
            if self.use_regex {
                normalized.split(re_ref, SplitDelimiterBehavior::Isolated)
            } else {
                Ok(vec![normalized])
            }
        })?;
        pretokenized.normalize(|normalized| {
            let s = normalized.get();
            
            // Convert UTF-8 string to UTF-16 bytes
            let utf16_bytes = utf8_to_utf16_bytes(s);
            
            // Transform each UTF-16 byte to its character representation
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(utf16_bytes.len());
            
            for (i, &byte) in utf16_bytes.iter().enumerate() {
                transformations.push((UTF16_BYTES_CHAR[&byte], if i > 0 { 1 } else { 0 }));
            }
            
            // Apply the transformations to convert the original string to byte-level representation
            normalized.transform(transformations, 0);
            Ok(())
        })
    }
}

/// As a `Decoder`, `UTF16ByteLevel` is in charge of converting any UTF-16 byte-level characters
/// to their unicode counterpart, before merging everything back into a single String.
/// This decoder will consume the tokens and merge them in one step to alleviate
/// the fact that single token decoded might be a byte not representable as a String.
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

/// As a `PostProcessor`, `UTF16ByteLevel` is in charge of trimming the offsets if necessary.
impl PostProcessor for UTF16ByteLevel {
    fn added_tokens(&self, _is_pair: bool) -> usize {
        0
    }

    fn process_encodings(
        &self,
        mut encodings: Vec<Encoding>,
        _add_special_tokens: bool,
    ) -> Result<Vec<Encoding>> {
        if self.trim_offsets {
            for encoding in encodings.iter_mut() {
                process_utf16_offsets(encoding, self.add_prefix_space);
                encoding
                    .get_overflowing_mut()
                    .iter_mut()
                    .for_each(|encoding| process_utf16_offsets(encoding, self.add_prefix_space));
            }
        }
        for (i, encoding) in encodings.iter_mut().enumerate() {
            encoding.set_sequence_id(i);
        }
        Ok(encodings)
    }
}

/// Process offsets for UTF-16 byte level encoding
/// This function adjusts offsets to account for the UTF-16 byte level transformation
pub fn process_utf16_offsets(encoding: &mut Encoding, add_prefix_space: bool) {
    encoding.process_tokens_with_offsets_mut(|(i, (token, offsets))| {
        let mut leading_spaces = token
            .chars()
            .take_while(|c| *c == UTF16_BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();
        let trailing_spaces = token
            .chars()
            .rev()
            .take_while(|c| *c == UTF16_BYTES_CHAR[&b' '] || c.is_whitespace())
            .count();

        if leading_spaces > 0 || trailing_spaces > 0 {
            if leading_spaces > 0 {
                // If user uses `is_pretokenized=True` we might have
                // offsets that might begin at the start of the string but are
                // NOT the first token.
                let is_first = i == 0 || offsets.0 == 0;
                if is_first && add_prefix_space && leading_spaces == 1 {
                    // If we are processing the first pair of offsets, with `add_prefix_space`,
                    // then we shouldn't remove anything we added. If there are more than one
                    // leading spaces though, it means we didn't add them, and they should be
                    // removed.
                    leading_spaces = 0;
                }
                offsets.0 = std::cmp::min(offsets.0 + leading_spaces, offsets.1);
            }
            if trailing_spaces > 0 && offsets.1 >= trailing_spaces {
                offsets.1 = std::cmp::max(offsets.1 - trailing_spaces, offsets.0);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utf8_to_utf16_bytes() {
        // Test ASCII
        let ascii = "Hello";
        let bytes = utf8_to_utf16_bytes(ascii);
        assert_eq!(bytes, vec![0x48, 0x00, 0x65, 0x00, 0x6C, 0x00, 0x6C, 0x00, 0x6F, 0x00]);
        
        // Test Korean characters
        let korean = "가나다";
        let bytes = utf8_to_utf16_bytes(korean);
        // 가 = U+AC00, 나 = U+B098, 다 = U+B2E4
        assert_eq!(bytes, vec![0x00, 0xAC, 0x98, 0xB0, 0xE4, 0xB2]);
    }

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
    fn test_roundtrip() {
        let test_strings = vec![
            "Hello World",
            "안녕하세요",
            "你好世界",
            "🌍🌎🌏",
            "Hello 안녕하세요 你好",
        ];
        
        for s in test_strings {
            let bytes = utf8_to_utf16_bytes(s);
            let recovered = utf16_bytes_to_utf8(&bytes).unwrap();
            assert_eq!(s, recovered);
        }
    }

    #[test]
    fn test_alphabet() {
        let alphabet = UTF16ByteLevel::alphabet();
        assert_eq!(alphabet.len(), 256);
    }
} 