from .base_tokenizer import BaseTokenizer
from .bert_wordpiece import BertWordPieceTokenizer
from .byte_level_bpe import ByteLevelBPETokenizer
from .char_level_bpe import CharBPETokenizer
from .sentencepiece_bpe import SentencePieceBPETokenizer
from .sentencepiece_unigram import SentencePieceUnigramTokenizer
from .utf16_byte_level_bpe import UTF16ByteLevelBPETokenizer

import json
from typing import Union, Dict, Any
from tokenizers.tokenizers import Tokenizer


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


def smart_tokenizer_from_file_with_original(path: str, original_from_file_func) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader with original from_file function to avoid recursion.
    """
    with open(path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    
    return smart_tokenizer_from_dict_with_original(tokenizer_data, original_from_file_func)


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


def smart_tokenizer_from_str_with_original(json_str: str, original_from_str_func) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader with original from_str function to avoid recursion.
    """
    tokenizer_data = json.loads(json_str)
    return smart_tokenizer_from_dict_with_original(tokenizer_data, original_from_str_func)


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
    decoder = tokenizer_data.get('decoder')
    is_utf16_decoder = is_utf16_component(decoder)
    
    # Check post_processor
    post_processor = tokenizer_data.get('post_processor')
    is_utf16_post = is_utf16_component(post_processor)
    
    # Load the base tokenizer first using the original method to avoid recursion
    try:
        # Import the original Tokenizer class to avoid recursion
        from tokenizers.tokenizers import Tokenizer as _OriginalTokenizer
        base_tokenizer = _OriginalTokenizer.from_str(json.dumps(tokenizer_data))
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None
    
    # Determine tokenizer type based on components and wrap appropriately
    if is_utf16_pre or is_utf16_decoder or is_utf16_post:
        # This is a UTF16ByteLevel tokenizer
        print("🔍 Detected UTF16ByteLevelBPE tokenizer")
        return BaseTokenizer(base_tokenizer, {"type": "UTF16ByteLevelBPE"})
        
    elif (pre_tokenizer.get('type') == 'ByteLevel' or 
          (decoder and decoder.get('type') == 'ByteLevel') or 
          (post_processor and post_processor.get('type') == 'ByteLevel')):
        # This is a regular ByteLevel tokenizer
        print("🔍 Detected ByteLevelBPE tokenizer")
        return BaseTokenizer(base_tokenizer, {"type": "ByteLevelBPE"})
    
    else:
        # Unknown or other tokenizer type, return generic BaseTokenizer
        print("🔍 Detected generic tokenizer type")
        return BaseTokenizer(base_tokenizer, {"type": "Unknown"})


def smart_tokenizer_from_dict_with_original(tokenizer_data: Dict[str, Any], original_from_str_func) -> Union[ByteLevelBPETokenizer, UTF16ByteLevelBPETokenizer, BaseTokenizer]:
    """
    Smart tokenizer loader with original from_str function to avoid recursion.
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
    decoder = tokenizer_data.get('decoder')
    is_utf16_decoder = is_utf16_component(decoder)
    
    # Check post_processor
    post_processor = tokenizer_data.get('post_processor')
    is_utf16_post = is_utf16_component(post_processor)
    
    # Load the base tokenizer using the original method to avoid recursion
    try:
        json_str = json.dumps(tokenizer_data)
        base_tokenizer = original_from_str_func(json_str)
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        return None
    
    # Determine tokenizer type based on components and create appropriate wrapper
    if is_utf16_pre or is_utf16_decoder or is_utf16_post:
        # This is a UTF16ByteLevel tokenizer
        print("🔍 Detected UTF16ByteLevelBPE tokenizer")
        
        # Fix decoder if it's None (common issue with UTF16ByteLevel)
        if not decoder or decoder.get('type') != 'UTF16ByteLevel':
            print("🔧 Fixing missing UTF16ByteLevel decoder")
            from tokenizers import decoders
            base_tokenizer.decoder = decoders.UTF16ByteLevel()
        
        # Create a UTF16ByteLevelBPETokenizer wrapper
        wrapper = UTF16ByteLevelBPETokenizer()
        wrapper._tokenizer = base_tokenizer
        wrapper._parameters = {"model": "UTF16ByteLevelBPE", "auto_detected": True}
        return wrapper
        
    elif (pre_tokenizer.get('type') == 'ByteLevel' or 
          (decoder and decoder.get('type') == 'ByteLevel') or 
          (post_processor and post_processor.get('type') == 'ByteLevel')):
        # This is a regular ByteLevel tokenizer
        print("🔍 Detected ByteLevelBPE tokenizer")
        # Create a ByteLevelBPETokenizer wrapper
        wrapper = ByteLevelBPETokenizer()
        wrapper._tokenizer = base_tokenizer
        wrapper._parameters = {"model": "ByteLevelBPE", "auto_detected": True}
        return wrapper
    
    else:
        # Unknown or other tokenizer type, return the base tokenizer
        print("🔍 Detected generic tokenizer type")
        return base_tokenizer


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
    decoder = tokenizer_data.get('decoder')
    post_processor = tokenizer_data.get('post_processor')
    
    if (is_utf16_component(pre_tokenizer) or 
        is_utf16_component(decoder) or 
        is_utf16_component(post_processor)):
        return 'UTF16ByteLevelBPE'
    elif (pre_tokenizer.get('type') == 'ByteLevel' or 
          (decoder and decoder.get('type') == 'ByteLevel') or 
          (post_processor and post_processor.get('type') == 'ByteLevel')):
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
    "smart_tokenizer_from_file_with_original",
    "smart_tokenizer_from_str_with_original",
    "smart_tokenizer_from_dict_with_original",
    "detect_tokenizer_type",
]
