"""
Subword and Byte-Pair Encoding (BPE) tokenization for Genomic Language Modeling (DNA-BERT style).
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Set, Optional
from collections import Counter
import re


class GenomicBpeTokenizer:
    """
    Learns and applies Byte-Pair Encoding (BPE) subword vocabularies directly on genomic nucleotide sequences.
    """

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Set[str] = {"A", "C", "G", "T", "<UNK>", "<PAD>", "<CLS>", "<SEP>"}

    def fit(self, sequences: List[str], num_merges: Optional[int] = None) -> GenomicBpeTokenizer:
        """Learns frequent adjacent nucleotide pair merges iteratively."""
        if num_merges is None:
            num_merges = self.vocab_size - len(self.vocab)

        # Tokenize sequences into individual base characters
        corpus = [list("".join(s.split()).upper()) for s in sequences]

        for _ in range(max(0, num_merges)):
            pairs = Counter()
            for seq_tokens in corpus:
                for i in range(len(seq_tokens) - 1):
                    pairs[(seq_tokens[i], seq_tokens[i + 1])] += 1

            if not pairs:
                break

            best_pair, count = pairs.most_common(1)[0]
            if count < 2:
                break

            self.merges.append(best_pair)
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add(merged_token)

            # Apply merge across corpus
            new_corpus = []
            for seq_tokens in corpus:
                new_seq = []
                i = 0
                while i < len(seq_tokens):
                    if i < len(seq_tokens) - 1 and (seq_tokens[i], seq_tokens[i + 1]) == best_pair:
                        new_seq.append(merged_token)
                        i += 2
                    else:
                        new_seq.append(seq_tokens[i])
                        i += 1
                new_corpus.append(new_seq)
            corpus = new_corpus

        return self

    def tokenize(self, sequence: str) -> List[str]:
        """Tokenizes a DNA sequence into learned BPE subword genomic units."""
        tokens = list("".join(sequence.split()).upper())
        for pair in self.merges:
            merged = pair[0] + pair[1]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, sequence: str) -> List[int]:
        """Encodes sequence into integer token IDs."""
        tokens = self.tokenize(sequence)
        vocab_list = sorted(list(self.vocab))
        token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
        unk_id = token_to_id.get("<UNK>", 0)
        return [token_to_id.get(tok, unk_id) for tok in tokens]
