from typing import Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers, processors, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, Sequence, unicode_normalizer_from_str

from .base_tokenizer import BaseTokenizer


class UTF16ByteLevelBPETokenizer(BaseTokenizer):
    """UTF16ByteLevelBPETokenizer

    Represents a UTF-16 Byte-level BPE tokenizer that converts UTF-8 strings to UTF-16 
    and then performs BPE tokenization at the byte level on the UTF-16 representation.
    
    Author: Hyunsik Kim <avantkim@gmail.com>
    Date: May 2025
    
    This implementation is based on the original ByteLevelBPETokenizer but adapted 
    to work with UTF-16 encoding instead of UTF-8. This can be beneficial for 
    languages that are more efficiently represented in UTF-16.
    
    Reference: Based on OpenAI's GPT-2 byte-level BPE approach but adapted for UTF-16
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        add_prefix_space: bool = False,
        lowercase: bool = False,
        dropout: Optional[float] = None,
        unicode_normalizer: Optional[str] = None,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
        trim_offsets: bool = False,
    ):
        if vocab is not None and merges is not None:
            tokenizer = Tokenizer(
                BPE(
                    vocab,
                    merges,
                    dropout=dropout,
                    continuing_subword_prefix=continuing_subword_prefix or "",
                    end_of_word_suffix=end_of_word_suffix or "",
                )
            )
        else:
            tokenizer = Tokenizer(BPE())

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = pre_tokenizers.UTF16ByteLevel(add_prefix_space=add_prefix_space)
        tokenizer.decoder = decoders.UTF16ByteLevel()
        tokenizer.post_processor = processors.UTF16ByteLevel(trim_offsets=trim_offsets)

        parameters = {
            "model": "UTF16ByteLevelBPE",
            "add_prefix_space": add_prefix_space,
            "lowercase": lowercase,
            "dropout": dropout,
            "unicode_normalizer": unicode_normalizer,
            "continuing_subword_prefix": continuing_subword_prefix,
            "end_of_word_suffix": end_of_word_suffix,
            "trim_offsets": trim_offsets,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab_filename: str, merges_filename: str, **kwargs):
        vocab, merges = BPE.read_file(vocab_filename, merges_filename)
        return UTF16ByteLevelBPETokenizer(vocab, merges, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
    ):
        """Train the model using the given files"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        show_progress: bool = True,
        special_tokens: List[Union[str, AddedToken]] = [],
        length: Optional[int] = None,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        ) 