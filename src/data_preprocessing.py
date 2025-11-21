"""
Data preprocessing: convert raw text files into tokenized sequences for training.

This module handles:
- Reading text files
- Simple character-level or word-level tokenization
- Creating source-target pairs for sequence-to-sequence learning
- Building vocabulary and encoding sequences
"""

import os
import re
from typing import List, Tuple, Dict
import numpy as np


def read_text_files(data_dir: str) -> List[str]:
    """
    Read all text files from a directory and return their contents.
    
    Args:
        data_dir: path to directory containing text files
    
    Returns:
        List of text strings, one per file
    """
    texts = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
    return texts


def simple_tokenize(text: str, level: str = 'word') -> List[str]:
    """
    Simple tokenization: split text into tokens.
    
    Args:
        text: input text string
        level: 'word' or 'char' for word-level or character-level tokenization
    
    Returns:
        List of tokens
    """
    if level == 'word':
        # Simple word tokenization: split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
    else:  # character level
        tokens = list(text.lower())
    return tokens


def build_vocab(texts: List[str], level: str = 'word', min_freq: int = 1) -> Dict[str, int]:
    """
    Build vocabulary from tokenized texts.
    
    Special tokens:
    - 0: PAD (padding)
    - 1: START (sequence start)
    - 2: END (sequence end)
    - 3: UNK (unknown/rare tokens)
    
    Args:
        texts: list of text strings
        level: 'word' or 'char' tokenization level
        min_freq: minimum frequency for a token to be included
    
    Returns:
        Dictionary mapping tokens to integer IDs
    """
    # Count token frequencies
    token_counts = {}
    for text in texts:
        tokens = simple_tokenize(text, level)
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Build vocabulary: special tokens first, then frequent tokens
    vocab = {'<PAD>': 0, '<START>': 1, '<END>': 2, '<UNK>': 3}
    token_id = 4
    
    for token, count in sorted(token_counts.items(), key=lambda x: -x[1]):
        if count >= min_freq:
            vocab[token] = token_id
            token_id += 1
    
    return vocab


def encode_sequence(tokens: List[str], vocab: Dict[str, int], max_len: int = None) -> np.ndarray:
    """
    Encode token sequence to integer IDs with padding.
    
    Args:
        tokens: list of token strings
        vocab: vocabulary dictionary
        max_len: maximum sequence length (None for no padding)
    
    Returns:
        NumPy array of token IDs
    """
    unk_id = vocab.get('<UNK>', 3)
    encoded = [vocab.get(token, unk_id) for token in tokens]
    
    if max_len is not None:
        # Pad or truncate to max_len
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded = encoded + [vocab['<PAD>']] * (max_len - len(encoded))
    
    return np.array(encoded, dtype=np.int32)


def create_sequence_pairs(texts: List[str], vocab: Dict[str, int], 
                         max_len: int = 50, level: str = 'word') -> Tuple[np.ndarray, np.ndarray]:
    """
    Create source-target pairs for sequence-to-sequence learning.
    
    For simplicity, we use the same sequence as both source and target
    (autoencoder-style), but shift the target by one position for next-token prediction.
    
    Args:
        texts: list of text strings
        vocab: vocabulary dictionary
        max_len: maximum sequence length
        level: tokenization level
    
    Returns:
        Tuple of (source_sequences, target_sequences) as NumPy arrays
    """
    source_seqs = []
    target_seqs = []
    
    for text in texts:
        tokens = simple_tokenize(text, level)
        if len(tokens) < 2:  # Skip very short sequences
            continue
        
        # Encode sequence
        encoded = encode_sequence(tokens, vocab, max_len)
        
        # Create source (all tokens) and target (shifted by 1 for next-token prediction)
        source = encoded
        target = np.roll(encoded, -1)  # Shift left by 1
        target[-1] = vocab['<PAD>']  # Last position becomes padding
        
        source_seqs.append(source)
        target_seqs.append(target)
    
    if not source_seqs:
        # Return empty arrays with correct shape if no data
        return np.array([], dtype=np.int32).reshape(0, max_len), np.array([], dtype=np.int32).reshape(0, max_len)
    
    return np.array(source_seqs), np.array(target_seqs)


def preprocess_data(data_dir: str, max_len: int = 50, level: str = 'word', 
                   min_freq: int = 1) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Complete preprocessing pipeline: read files, build vocab, create training pairs.
    
    Args:
        data_dir: directory containing text files
        max_len: maximum sequence length
        level: 'word' or 'char' tokenization
        min_freq: minimum token frequency for vocabulary
    
    Returns:
        Tuple of (source_sequences, target_sequences, vocabulary)
    """
    # Read text files
    texts = read_text_files(data_dir)
    if not texts:
        raise ValueError(f"No text files found in {data_dir}")
    
    # Build vocabulary
    vocab = build_vocab(texts, level=level, min_freq=min_freq)
    
    # Create sequence pairs
    source_seqs, target_seqs = create_sequence_pairs(texts, vocab, max_len=max_len, level=level)
    
    return source_seqs, target_seqs, vocab

