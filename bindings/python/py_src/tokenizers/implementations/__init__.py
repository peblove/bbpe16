from .base_tokenizer import BaseTokenizer
from .bert_wordpiece import BertWordPieceTokenizer
from .byte_level_bpe import ByteLevelBPETokenizer
from .char_level_bpe import CharBPETokenizer
from .sentencepiece_bpe import SentencePieceBPETokenizer
from .sentencepiece_unigram import SentencePieceUnigramTokenizer
from .utf16_byte_level_bpe import UTF16ByteLevelBPETokenizer

import json
from typing import Union, Dict, Any
from tokenizers import Tokenizer


def smart_tokenizer_from_file(path: str) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader that automatically detects the tokenizer type from JSON file.
    
    This function analyzes the JSON structure to determine whether it's a UTF16ByteLevel
    tokenizer or a regular ByteLevel tokenizer, then loads the appropriate implementation.
    
    Args:
        path (str): Path to the tokenizer JSON file
        
    Returns:
        The appropriate tokenizer instance based on the detected type
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    return smart_tokenizer_from_dict(tokenizer_data)


def smart_tokenizer_from_str(json_str: str) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader that automatically detects the tokenizer type from JSON string.
    
    Args:
        json_str (str): JSON string containing tokenizer configuration
        
    Returns:
        The appropriate tokenizer instance based on the detected type
    """
    tokenizer_data = json.loads(json_str)
    return smart_tokenizer_from_dict(tokenizer_data)


def smart_tokenizer_from_dict(tokenizer_data: Dict[str, Any]) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader that automatically detects the tokenizer type from dictionary.
    
    This function analyzes the tokenizer configuration to determine the correct type:
    1. Checks pre_tokenizer type for UTF16ByteLevel vs ByteLevel
    2. Checks decoder type for UTF16ByteLevel vs ByteLevel  
    3. Checks post_processor type for UTF16ByteLevel vs ByteLevel
    4. Falls back to generic Tokenizer if type cannot be determined
    
    Args:
        tokenizer_data (dict): Dictionary containing tokenizer configuration
        
    Returns:
        The appropriate tokenizer instance based on the detected type
    """
    def is_utf16_component(component_data):
        """Check if a component is UTF16ByteLevel type"""
        if not isinstance(component_data, dict):
            return False
        return component_data.get('type') == 'UTF16ByteLevel'
    
    # Check pre_tokenizer
    pre_tokenizer = tokenizer_data.get('pre_tokenizer', {})
    is_utf16_pre = is_utf16_component(pre_tokenizer)
    
    # Check decoder  
    decoder = tokenizer_data.get('decoder', {})
    is_utf16_decoder = is_utf16_component(decoder)
    
    # Check post_processor
    post_processor = tokenizer_data.get('post_processor', {})
    is_utf16_post = is_utf16_component(post_processor)
    
    # Determine tokenizer type based on components
    if is_utf16_pre or is_utf16_decoder or is_utf16_post:
        # This is a UTF16ByteLevel tokenizer
        print("🔍 Detected UTF16ByteLevelBPE tokenizer")
        
        # Load the base tokenizer first
        base_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))
        
        # Extract parameters for UTF16ByteLevelBPETokenizer
        model_data = tokenizer_data.get('model', {})
        vocab = model_data.get('vocab', {})
        merges = model_data.get('merges', [])
        
        # Convert merges from list format to tuple format if needed
        if merges and isinstance(merges[0], list):
            merges = [tuple(merge) for merge in merges]
        
        # Extract other parameters
        add_prefix_space = pre_tokenizer.get('add_prefix_space', False)
        trim_offsets = post_processor.get('trim_offsets', False)
        
        # Create UTF16ByteLevelBPETokenizer with the loaded vocab and merges
        utf16_tokenizer = UTF16ByteLevelBPETokenizer(
            vocab=vocab,
            merges=merges,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets
        )
        
        return utf16_tokenizer
        
    elif (pre_tokenizer.get('type') == 'ByteLevel' or 
          decoder.get('type') == 'ByteLevel' or 
          post_processor.get('type') == 'ByteLevel'):
        # This is a regular ByteLevel tokenizer
        print("🔍 Detected ByteLevelBPE tokenizer")
        
        # Extract parameters for ByteLevelBPETokenizer
        model_data = tokenizer_data.get('model', {})
        vocab = model_data.get('vocab', {})
        merges = model_data.get('merges', [])
        
        # Convert merges from list format to tuple format if needed
        if merges and isinstance(merges[0], list):
            merges = [tuple(merge) for merge in merges]
        
        # Extract other parameters
        add_prefix_space = pre_tokenizer.get('add_prefix_space', False)
        trim_offsets = post_processor.get('trim_offsets', False)
        
        # Create ByteLevelBPETokenizer with the loaded vocab and merges
        byte_tokenizer = ByteLevelBPETokenizer(
            vocab=vocab,
            merges=merges,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets
        )
        
        return byte_tokenizer
    
    else:
        # Unknown or other tokenizer type, return generic BaseTokenizer
        print("🔍 Detected generic tokenizer type")
        base_tokenizer = Tokenizer.from_str(json.dumps(tokenizer_data))
        return BaseTokenizer(base_tokenizer, {})


def detect_tokenizer_type(path: str) -> str:
    """
    Detect the tokenizer type from a JSON file without loading the full tokenizer.
    
    Args:
        path (str): Path to the tokenizer JSON file
        
    Returns:
        str: The detected tokenizer type ('UTF16ByteLevelBPE', 'ByteLevelBPE', or 'Unknown')
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    def is_utf16_component(component_data):
        if not isinstance(component_data, dict):
            return False
        return component_data.get('type') == 'UTF16ByteLevel'
    
    # Check components
    pre_tokenizer = tokenizer_data.get('pre_tokenizer', {})
    decoder = tokenizer_data.get('decoder', {})
    post_processor = tokenizer_data.get('post_processor', {})
    
    if (is_utf16_component(pre_tokenizer) or 
        is_utf16_component(decoder) or 
        is_utf16_component(post_processor)):
        return 'UTF16ByteLevelBPE'
    elif (pre_tokenizer.get('type') == 'ByteLevel' or 
          decoder.get('type') == 'ByteLevel' or 
          post_processor.get('type') == 'ByteLevel'):
        return 'ByteLevelBPE'
    else:
        return 'Unknown'


__all__ = [
    "BaseTokenizer",
    "BertWordPieceTokenizer", 
    "ByteLevelBPETokenizer",
    "CharBPETokenizer",
    "SentencePieceBPETokenizer",
    "UTF16ByteLevelBPETokenizer",
    "smart_tokenizer_from_file",
    "smart_tokenizer_from_str", 
    "smart_tokenizer_from_dict",
    "detect_tokenizer_type",
]
