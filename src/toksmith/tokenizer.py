"""BPE Tokenizer Implementation (follows gpt-2 assumptions)"""
from typing import Sequence, Tuple, TypeVar

import regex as re

T = TypeVar("T")

# from here: https://github.com/openai/tiktoken/pull/234/files
GPT2_SPLIT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# helpers ============================

def _merge(seq: Sequence[T], pair: Tuple[T,T], new_ix: T) -> Tuple[T, ...]:
  """Given a sequence of elements produces a new
  sequence with all non-overlapping occurrences of `pair`
  replaced by `new_ix`"""
  if not isinstance(pair, tuple) or len(pair) != 2:
    raise ValueError("`pair` must be a 2-tuple")
  new_seq = []
  i = 0
  while i < len(seq):
    # check in range and if match
    if i+1 < len(seq) and (seq[i], seq[i+1]) == pair:
      new_seq.append(new_ix)
      i += 2 # correct step
    else:
      new_seq.append(seq[i]) # only current position
      i += 1
  return tuple(new_seq)

# tokenizer code ================================
class Tokenizer():
  """GPT2-like tokenizer, using BPE algorithm"""

  def __init__(self) -> None:
    self.pattern = re.compile(GPT2_SPLIT_PAT)
    self._reset_state()

  def _reset_state(self):
      self.vocab = {i: int.to_bytes(i) for i in range(256)}
      self.merges = []

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

  def train(self, text: str, vocab_size: int, special_tokens: list[str]) -> None:
    """Trains a BPE tokenizer on provided text, updates tokenizer state

    Any existing merges or vocab entries beyond the initial 256 byte-vocab
    will be cleared before training begins.

    Note: special token occurrences will be stripped off from the text before
    training

    Args:
        text (str): unicode text
        vocab_size (int): total size after training (== 256 + #merges + #special_tokens)
        special_tokens (list[str]): list of special tokens, i.e. <|endoftext|>

    Raises:
        ValueError: if vocab_size < # init bytes + # special tokens
    """
    # ensure clean state before training
    self._reset_state()
    # start training
    min_size = 256 + len(special_tokens)
    if vocab_size < min_size:
      raise ValueError(f"vocab_size must be >= {min_size}")
    if special_tokens:
      delim = "|".join(map(re.escape, special_tokens))
      # we strip off special tokens and join back right away
      # we substitute special token with one white space
      # to ensure we dont accidentally smash tokens together
      # might add multiprocessing for large texts and do _pretoken_count
      # in parallel
      text = re.sub(f"(?:{delim})+", " ", text)
    pretokens = self._pretoken_count(text)
    ix = 256
    num_iters = vocab_size - min_size # keep vocab space for special tokens
    for _ in range(num_iters):
      pair_counts = self._pairs_count(pretokens)
      if not pair_counts: break # for small text examples with large vocab size
      # find most frequent pair, ties resolved in lexicographical order
      top_pair, _ = max(pair_counts.items(), key=lambda it: [it[1], it[0]])
      # merge
      # Each merge introduces a new token (pair → new token) that wasn’t in the vocabulary before
      # Pretoken keys are sequences of current tokens.
      # Until you merge ('t', 'e') into 'te', there's no way 'te' appears as a unit inside any key
      # Only keys that contain the exact pair ('t', 'e') in adjacent positions will be modified.
      # The output of merge() depends deterministically on the input key.
      # Therefore, at most one original key can produce any given new_pt in the merge step.
      for pt in list(pretokens): # static copy of keys (prevents RuntimeError if we iterate original dict)
        new_pt = _merge(pt, top_pair, ix)
        if new_pt != pt: # update only if we merged new index
        # even though we proved it can't happen (see above), we want this assertions and perhaps test against it
        # so we are sure not to mess up with implementation
          assert new_pt not in pretokens, f"Collision: {new_pt} already in pretokens"
          pretokens[new_pt] = pretokens.pop(pt) #  safe from key collisions under the BPE merge assumptions
      # updata tokenizer state
      self.merges.append(top_pair)
      self.vocab[ix] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
      # increment index
      ix += 1
    # finished training, let't add special tokens to vocab
    for i, s_tok in enumerate(special_tokens, ix):
      self.vocab[i] = bytes(s_tok, encoding="utf-8")