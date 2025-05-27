from enum import Enum
from typing import List, Tuple, Union


Offsets = Tuple[int, int]

TextInputSequence = str
"""A :obj:`str` that represents an input sequence """

PreTokenizedInputSequence = Union[List[str], Tuple[str]]
"""A pre-tokenized input sequence. Can be one of:

    - A :obj:`List` of :obj:`str`
    - A :obj:`Tuple` of :obj:`str`
"""

TextEncodeInput = Union[
    TextInputSequence,
    Tuple[TextInputSequence, TextInputSequence],
    List[TextInputSequence],
]
"""Represents a textual input for encoding. Can be either:

    - A single sequence: :data:`~tokenizers.TextInputSequence`
    - A pair of sequences:

      - A :obj:`Tuple` of :data:`~tokenizers.TextInputSequence`
      - Or a :obj:`List` of :data:`~tokenizers.TextInputSequence` of size 2
"""

PreTokenizedEncodeInput = Union[
    PreTokenizedInputSequence,
    Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence],
    List[PreTokenizedInputSequence],
]
"""Represents a pre-tokenized input for encoding. Can be either:

    - A single sequence: :data:`~tokenizers.PreTokenizedInputSequence`
    - A pair of sequences:

      - A :obj:`Tuple` of :data:`~tokenizers.PreTokenizedInputSequence`
      - Or a :obj:`List` of :data:`~tokenizers.PreTokenizedInputSequence` of size 2
"""

InputSequence = Union[TextInputSequence, PreTokenizedInputSequence]
"""Represents all the possible types of input sequences for encoding. Can be:

    - When ``is_pretokenized=False``: :data:`~TextInputSequence`
    - When ``is_pretokenized=True``: :data:`~PreTokenizedInputSequence`
"""

EncodeInput = Union[TextEncodeInput, PreTokenizedEncodeInput]
"""Represents all the possible types of input for encoding. Can be:

    - When ``is_pretokenized=False``: :data:`~TextEncodeInput`
    - When ``is_pretokenized=True``: :data:`~PreTokenizedEncodeInput`
"""


class OffsetReferential(Enum):
    ORIGINAL = "original"
    NORMALIZED = "normalized"


class OffsetType(Enum):
    BYTE = "byte"
    CHAR = "char"


class SplitDelimiterBehavior(Enum):
    REMOVED = "removed"
    ISOLATED = "isolated"
    MERGED_WITH_PREVIOUS = "merged_with_previous"
    MERGED_WITH_NEXT = "merged_with_next"
    CONTIGUOUS = "contiguous"


from .tokenizers import (
    AddedToken,
    Encoding,
    NormalizedString,
    PreTokenizedString,
    Regex,
    Token,
    Tokenizer as _Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    __version__,
)

# Import implementations will be done lazily to avoid circular import

# Add smart loading methods to the existing Tokenizer class
def _smart_from_file(path: str, smart_load: bool = True):
    """
    Instantiate a new Tokenizer from the file at the given path.
    
    Args:
        path (str): A path to a local JSON file representing a previously serialized Tokenizer
        smart_load (bool): If True, automatically detect and load the appropriate tokenizer type.
                          If False, use the standard loading method.
    
    Returns:
        Tokenizer or specialized tokenizer: The appropriate tokenizer instance
    """
    if smart_load:
        try:
            # Import here to avoid circular import
            from .implementations import smart_tokenizer_from_file
            return smart_tokenizer_from_file(path)
        except Exception:
            # Fall back to standard loading if smart loading fails
            pass
    
    # Standard loading
    return _Tokenizer.from_file(path)

def _smart_from_str(json_str: str, smart_load: bool = True):
    """
    Instantiate a new Tokenizer from the given JSON string.
    
    Args:
        json_str (str): A valid JSON string representing a previously serialized Tokenizer
        smart_load (bool): If True, automatically detect and load the appropriate tokenizer type.
                          If False, use the standard loading method.
    
    Returns:
        Tokenizer or specialized tokenizer: The appropriate tokenizer instance
    """
    if smart_load:
        try:
            # Import here to avoid circular import
            from .implementations import smart_tokenizer_from_str
            return smart_tokenizer_from_str(json_str)
        except Exception:
            # Fall back to standard loading if smart loading fails
            pass
    
    # Standard loading
    return _Tokenizer.from_str(json_str)

def _detect_type(path: str) -> str:
    """
    Detect the tokenizer type from a JSON file without loading the full tokenizer.
    
    Args:
        path (str): Path to the tokenizer JSON file
        
    Returns:
        str: The detected tokenizer type ('UTF16ByteLevelBPE', 'ByteLevelBPE', or 'Unknown')
    """
    # Import here to avoid circular import
    from .implementations import detect_tokenizer_type
    return detect_tokenizer_type(path)

# Monkey patch the Tokenizer class to add smart loading capabilities
_Tokenizer.smart_from_file = staticmethod(_smart_from_file)
_Tokenizer.smart_from_str = staticmethod(_smart_from_str)
_Tokenizer.detect_type = staticmethod(_detect_type)

# Export the original Tokenizer class with added functionality
Tokenizer = _Tokenizer
