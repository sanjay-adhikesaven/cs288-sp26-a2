"""
BPE Tokenizer implementation compatible with GPT-2 / tiktoken.
"""

from __future__ import annotations

import regex as re
from typing import Iterator


class Tokenizer:
    """
    A BPE (Byte Pair Encoding) tokenizer compatible with GPT-2.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initialize the tokenizer.

        Args:
            vocab: Mapping from token ID to bytes
            merges: List of BPE merge pairs (bytes, bytes)
            special_tokens: List of special token strings
        """
        self.vocab = vocab  # id -> bytes
        self.inverse_vocab = {v: k for k, v in vocab.items()}  # bytes -> id (also used as rank)
        self.merges = merges
        # Note: We use inverse_vocab for BPE ranking, not the merges list.
        # In GPT-2/tiktoken, the token ID serves as the rank - lower ID = higher priority.
        # This is different from naive BPE which uses merge order.
        
        # Handle special tokens
        self.special_tokens = special_tokens or []
        # Sort special tokens by length (descending) for longest-match-first
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        
        # Build special token to ID mapping
        self.special_token_ids = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.inverse_vocab:
                self.special_token_ids[token] = self.inverse_vocab[token_bytes]
        
        # GPT-2 regex pattern for pre-tokenization
        # This splits text into chunks that are tokenized independently
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE
        )

    def _get_pairs(self, tokens: list[bytes]) -> set[tuple[bytes, bytes]]:
        """Get all adjacent pairs of tokens."""
        pairs = set()
        for i in range(len(tokens) - 1):
            pairs.add((tokens[i], tokens[i + 1]))
        return pairs

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        """
        Apply BPE to a single token (sequence of bytes).
        Returns a list of merged byte sequences.
        
        Uses vocab ranks (token IDs) to determine merge priority.
        Lower token ID = higher priority (more common/earlier merge).
        
        Algorithm:
            1. Start with individual bytes as tokens
            2. While there are pairs that can be merged:
               a. Find the pair whose merged result has the lowest vocab rank
               b. Merge all occurrences of that pair
            3. Return final token list
        """
        # Start with individual bytes
        tokens = [bytes([b]) for b in token_bytes]
        
        if len(tokens) <= 1:
            return tokens
        
        # approach: slide through tokens and look at all tokens (tokens[i] + tokens[i+1])
        # find the rank of this via self.inverse.vocab and after each iteration, save the lowest vocab rank + occurances
        # merge all occurences of that pair together by combining the tokens (effecitvely reducing list length 1 at a time)
        # keep going until no pairs can be merged - this occurs if no pair is within the vocab 

        while True:
            lowest_merge_rank = float("inf")
            best_pair_bytes = None
            for i in range(len(tokens) - 1):
                current_pair = tokens[i] + tokens[i+1] # current pair bytes
                if current_pair in self.inverse_vocab and self.inverse_vocab[current_pair] < lowest_merge_rank:
                    lowest_merge_rank = self.inverse_vocab[current_pair]
                    best_pair_bytes = current_pair
                
            if best_pair_bytes is None:
                return tokens
        
            new_tokens = []
            current_index = 0
            while current_index < len(tokens):
                if current_index < (len(tokens) - 1) and (tokens[current_index] + tokens[current_index+1] == best_pair_bytes):
                    new_tokens.append(best_pair_bytes)
                    current_index += 2
                else:
                    new_tokens.append(tokens[current_index])
                    current_index += 1
            
            tokens = new_tokens
        

    def _split_with_special_tokens(self, text: str) -> list[tuple[str, bool]]:
        """
        Split text by special tokens, preserving them.
        Returns list of (substring, is_special) tuples.
        """
        if not self.special_tokens_sorted:
            return [(text, False)] if text else []

        escaped_special_tokens = map(re.escape, self.special_tokens_sorted)
        pattern = f"({'|'.join(escaped_special_tokens)})"

        special_toks_set = set(self.special_tokens_sorted)

        return [(p, p in special_toks_set) for p in re.split(pattern, text) if p != ""]

    def _encode_chunk(self, text: str) -> list[int]:
        """
        Encode a text chunk (without special tokens) to token IDs.
        
        Algorithm:
            1. Use regex pattern (self.pat) to split text into pre-tokens
            2. For each pre-token:
               a. Convert to bytes
               b. Apply BPE to get list of byte sequences
               c. Convert each byte sequence to token ID using inverse_vocab
               d. Handle unknown tokens by falling back to individual bytes
        """
        if not text:
            return []
        
        ids = []

        for chunk in self.pat.findall(text):
            chunk_bytes = chunk.encode("utf-8")
            bpe_tokens = self._bpe(chunk_bytes)

            current_token_ids = []
            for bpe in bpe_tokens:
                if bpe in self.inverse_vocab:
                    current_token_ids.append(self.inverse_vocab[bpe])
                else:
                    for byte in bpe:
                        current_token_ids.append(self.inverse_vocab[bytes([byte])])

            ids.extend(current_token_ids)
        
        return ids

    def encode(self, text: str) -> list[int]:
        """
        Encode a string to a list of token IDs.
        
        Args:
            text: Input string to encode
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        ids = []
        
        # Split by special tokens first
        parts = self._split_with_special_tokens(text)
        
        for part, is_special in parts:
            if is_special:
                # Add special token ID
                ids.append(self.special_token_ids[part])
            else:
                # Encode regular text
                ids.extend(self._encode_chunk(part))
        
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs to a string.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded string
        
        Algorithm:
            1. For each token_id, look up corresponding bytes in self.vocab
            2. Concatenate all byte chunks
            3. Decode as UTF-8 with errors="replace"
        """
        if not ids:
            return ""

        bytes_list = [self.vocab[id] for id in ids]
        concatenated_bytes = b"".join(bytes_list)

        return concatenated_bytes.decode("utf-8", errors="replace")
        

    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]:
        """
        Memory-efficient encoding of an iterable of strings.
        Yields token IDs one at a time without loading entire input into memory.
        
        Args:
            iterable: An iterable of strings (e.g., file handle)
            
        Yields:
            Token IDs one at a time
        """
        # Buffer for handling text that spans multiple lines
        buffer = ""
        
        for chunk in iterable:
            buffer += chunk
            
            # Process complete portions, keeping potential partial special tokens
            # Find the last safe split point
            safe_end = self._find_safe_split_point(buffer)
            
            if safe_end > 0:
                to_process = buffer[:safe_end]
                buffer = buffer[safe_end:]
                
                for token_id in self.encode(to_process):
                    yield token_id
        
        # Process remaining buffer
        if buffer:
            for token_id in self.encode(buffer):
                yield token_id

    def _find_safe_split_point(self, text: str) -> int:
        """
        Find a safe point to split text for streaming encoding.
        We need to be careful not to split in the middle of:
        1. A potential special token
        2. A whitespace sequence (to preserve tokens like '\\n\\n')
        """
        if not text:
            return 0
        
        # Check if any special token could be starting at the end
        max_special_len = max((len(s) for s in self.special_tokens), default=0)
        
        # We need to keep at least max_special_len - 1 characters in buffer
        # to avoid splitting a special token
        min_keep = max_special_len - 1 if max_special_len > 0 else 0
        
        if len(text) <= min_keep:
            return 0
        
        safe_end = len(text)
        
        # Check for partial special token matches at the end
        for special in self.special_tokens:
            # Check if any prefix of special token matches end of text
            for prefix_len in range(1, len(special)):
                prefix = special[:prefix_len]
                if text.endswith(prefix):
                    safe_end = min(safe_end, len(text) - prefix_len)
        
        # Don't split in the middle of trailing whitespace
        # This prevents breaking up tokens like '\n\n'
        if safe_end > 0:
            # Find the last non-whitespace character
            last_non_ws = safe_end - 1
            while last_non_ws >= 0 and text[last_non_ws].isspace():
                last_non_ws -= 1
            
            # If there's trailing whitespace, don't include it in this chunk
            # unless the entire text is whitespace
            if last_non_ws >= 0 and last_non_ws < safe_end - 1:
                safe_end = last_non_ws + 1
        
        return safe_end


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    """
    Create a tokenizer from vocabulary and merge rules.
    
    Args:
        vocab: Mapping from token ID to bytes
        merges: List of BPE merge pairs
        special_tokens: Optional list of special token strings
        
    Returns:
        Tokenizer instance
    """
    return Tokenizer(vocab, merges, special_tokens)