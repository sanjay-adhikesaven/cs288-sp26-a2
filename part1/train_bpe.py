"""
BPE (Byte Pair Encoding) training implementation.

This module implements the BPE algorithm for learning a tokenizer vocabulary
from a text corpus, compatible with GPT-2 style tokenization.
"""

from __future__ import annotations

import regex as re
from collections import Counter
from pathlib import Path
from typing import Iterator


# GPT-2 pre-tokenization pattern
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


def get_pairs(word: tuple[bytes, ...]) -> set[tuple[bytes, bytes]]:
    """Get all adjacent pairs in a word (tuple of byte tokens)."""
    pairs = set()
    for i in range(len(word) - 1):
        pairs.add((word[i], word[i + 1]))
    return pairs

def merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    """Merge all occurrences of a pair in a word."""
    first, second = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)


def pre_tokenize(text: str, special_tokens: list[str] | None = None) -> Iterator[str]:
    """
    Pre-tokenize text using GPT-2 pattern, preserving special tokens.
    
    Special tokens are yielded as-is (not split by the regex pattern).
    """
    special_tokens = special_tokens or []
    
    if not special_tokens:
        # No special tokens, just use the pattern
        for match in GPT2_PAT.finditer(text):
            yield match.group()
        return
    
    # Sort special tokens by length (longest first) for greedy matching
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    
    # Build a pattern that matches special tokens
    import re as std_re
    special_pattern = "|".join(std_re.escape(s) for s in sorted_specials)
    split_pattern = f"({special_pattern})"
    
    # Split text by special tokens
    parts = std_re.split(split_pattern, text)
    
    for part in parts:
        if part in special_tokens:
            # Special token - yield as-is, but it won't be BPE-encoded
            # (we skip special tokens in the word frequency counting)
            continue
        elif part:
            # Regular text - apply GPT-2 pre-tokenization
            for match in GPT2_PAT.finditer(part):
                yield match.group()

def compute_pair_freqs(word_frequencies):
    pair_freqs = Counter()
    for word, freq in word_frequencies.items():
        current_pairs = get_pairs(word)
        for pair in current_pairs:
            pair_freqs[pair] += freq
    
    return pair_freqs

def train_bpe(
    input_path: Path,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer from a text file.
    
    Args:
        input_path: Path to the input text file
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
        
    Returns:
        Tuple of (vocab, merges) where:
        - vocab: dict mapping token_id (int) -> token (bytes)
        - merges: list of merge pairs (bytes, bytes)
    
    Algorithm:
        1. Read and pre-tokenize the corpus using GPT-2 pattern
        2. Initialize vocabulary with special tokens + all 256 byte values
        3. Count word frequencies (each word is tuple of single-byte tokens)
        4. Count initial pair frequencies across all words
        5. Repeat until vocab reaches target size:
           a. Find most frequent pair (use lexicographic ordering for ties)
           b. Add merged token to vocabulary
           c. Update word representations by applying merge
           d. Update pair counts
        6. Return vocabulary and list of merges
    """
    special_tokens = special_tokens or []
    
    # Read the corpus
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    
    # Build set of "forbidden" substrings from special tokens
    forbidden_substrings = set()
    for special in special_tokens:
        special_bytes = special.encode("utf-8")
        for i in range(2, len(special_bytes) + 1):
            forbidden_substrings.add(special_bytes[:i])
    
    pre_tokenized = pre_tokenize(text, special_tokens = special_tokens)
    
    current_vocab = {}
    next_vocab_number = 0
    for token in special_tokens:
        current_vocab[next_vocab_number] = token.encode("utf-8")
        next_vocab_number += 1
    
    for i in range(256):
        current_vocab[next_vocab_number] = bytes([i])
        next_vocab_number += 1

    word_freqs = Counter()
    for word in pre_tokenized:
        bytes_word = tuple(bytes([b]) for b in word.encode("utf-8"))
        word_freqs[bytes_word] += 1

    all_pair_freqs = compute_pair_freqs(word_freqs)
    
    all_merges = [] # list of (bytes, bytes) merges
    while len(current_vocab) < vocab_size:
        to_merge = max(all_pair_freqs, key = lambda x: (all_pair_freqs[x], x))
        current_vocab[next_vocab_number] = to_merge[0] + to_merge[1]
        next_vocab_number += 1
        all_merges.append(to_merge)
        
        new_word_freqs = Counter()
        for word, count in word_freqs.items():
            merged_word = merge_word(word, to_merge)
            new_word_freqs[merged_word] += count
        word_freqs = new_word_freqs

        all_pair_freqs = compute_pair_freqs(word_freqs)
        if not all_pair_freqs:
            return current_vocab, all_merges
    
    return current_vocab, all_merges

