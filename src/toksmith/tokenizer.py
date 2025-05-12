"""BPE Tokenizer Implementation (follows gpt-2 assumptions)"""
from typing import Sequence, Tuple, TypeVar

import regex as re

# from here: https://github.com/openai/tiktoken/pull/234/files
GPT2_SPLIT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer():
  """GPT2-like tokenizer, using BPE algorithm"""

  def __init__(self) -> None:
    self.pattern = re.compile(GPT2_SPLIT_PAT)

  def _pretoken_count(self, text: str) -> dict[tuple[int, ...], int]:
    """Pre-tokenizes the text and produces the counter
    of pre-tokens represented as tuple of utf-8 encoded bytes"""
    pretokens = dict()
    for mt in self.pattern.finditer(text):
      pt = mt.group() # -> str; match will have one pretoken per group
      pt = tuple(pt.encode('utf-8'))
      pretokens[pt] = pretokens.get(pt, 0) + 1
    return pretokens

  def _pairs_count(self, pretokens: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
    """Iterates through pre-tokens dict and produces
    the counter of all sequential pairs of units.
    Unit is an element of pre-token"""
    pair_counts = dict()
    for pt, cnt in pretokens.items():
      for p in zip(pt, pt[1:]):
        pair_counts[p] = pair_counts.get(p, 0) + cnt
    return pair_counts
