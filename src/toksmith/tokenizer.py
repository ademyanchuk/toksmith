"""BPE Tokenizer Implementation (follows gpt-2 assumptions)"""

import json
import os
import tempfile
from pathlib import Path
from typing import Sequence, Tuple, TypeVar

import regex as re

T = TypeVar('T')

# from here: https://github.com/openai/tiktoken/pull/234/files
GPT2_SPLIT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# helpers ============================


def _merge(seq: Sequence[T], pair: Tuple[T, T], new_ix: T) -> Tuple[T, ...]:
  """Given a sequence of elements produces a new
  sequence with all non-overlapping occurrences of `pair`
  replaced by `new_ix`"""
  if not isinstance(pair, tuple) or len(pair) != 2:
    raise ValueError('`pair` must be a 2-tuple')
  new_seq = []
  i = 0
  while i < len(seq):
    # check in range and if match
    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
      new_seq.append(new_ix)
      i += 2  # correct step
    else:
      new_seq.append(seq[i])  # only current position
      i += 1
  return tuple(new_seq)


# tokenizer code ================================
class Tokenizer:
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
      pt = mt.group()  # -> str; match will have one pretoken per group
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
      raise ValueError(f'vocab_size must be >= {min_size}')
    if special_tokens:
      delim = '|'.join(map(re.escape, special_tokens))
      # we strip off special tokens and join back right away
      # we substitute special token with one white space
      # to ensure we dont accidentally smash tokens together
      # might add multiprocessing for large texts and do _pretoken_count
      # in parallel
      text = re.sub(f'(?:{delim})+', ' ', text)
    pretokens = self._pretoken_count(text)
    ix = 256
    num_iters = vocab_size - min_size  # keep vocab space for special tokens
    for _ in range(num_iters):
      pair_counts = self._pairs_count(pretokens)
      if not pair_counts:
        break  # for small text examples with large vocab size
      # find most frequent pair, ties resolved in lexicographical order
      top_pair, _ = max(pair_counts.items(), key=lambda it: [it[1], it[0]])
      # merge
      # Each merge introduces a new token (pair → new token) that wasn’t in the vocabulary before
      # Pretoken keys are sequences of current tokens.
      # Until you merge ('t', 'e') into 'te', there's no way 'te' appears as a unit inside any key
      # Only keys that contain the exact pair ('t', 'e') in adjacent positions will be modified.
      # The output of merge() depends deterministically on the input key.
      # Therefore, at most one original key can produce any given new_pt in the merge step.
      for pt in list(pretokens):  # static copy of keys (prevents RuntimeError if we iterate original dict)
        new_pt = _merge(pt, top_pair, ix)
        if new_pt != pt:  # update only if we merged new index
          # even though we proved it can't happen (see above), we want this assertions and perhaps test against it
          # so we are sure not to mess up with implementation
          assert new_pt not in pretokens, f'Collision: {new_pt} already in pretokens'
          pretokens[new_pt] = pretokens.pop(pt)  #  safe from key collisions under the BPE merge assumptions
      # updata tokenizer state
      self.merges.append(top_pair)
      self.vocab[ix] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
      # increment index
      ix += 1
    # finished training, let't add special tokens to vocab
    for i, s_tok in enumerate(special_tokens, ix):
      self.vocab[i] = bytes(s_tok, encoding='utf-8')

  def save_state(self, prefix: str, folder: str | Path) -> Path:
    """
    Save the tokenizer's merges+vocab to `<folder>/<prefix>_tokenizer.json`,
    creating `folder` if needed.
    """

    # 1) Resolve the folder path safely
    folder_path = Path(folder).expanduser().resolve()
    if folder_path.exists() and not folder_path.is_dir():
      raise NotADirectoryError(f'{folder!r} exists but is not a directory')
    folder_path.mkdir(parents=True, exist_ok=True)

    # 2) Sanitize the prefix so it can’t escape the folder
    #    (e.g. someone passing "../bad" or absolute paths)
    safe_prefix = Path(prefix).name  # strips any path components
    filename = f'{safe_prefix}_tokenizer.json'
    target = folder_path / filename

    # 3) Write atomically (avoid half-written files on crash/interruption)
    #    – write to a temp file in the same folder
    #    – then os.replace() to move it into place
    with tempfile.NamedTemporaryFile(
      mode='w',
      dir=folder_path,
      delete=False,
      prefix=safe_prefix,
      suffix='.json',
    ) as tf:
      json.dump(
        {
          'version': 1,
          'merges': self.merges,
          'vocab': {str(k): self.vocab[k].hex() for k in self.vocab},
        },
        tf,
        indent=2,
        ensure_ascii=False,
      )
      tempname = tf.name

    os.replace(tempname, target)  # atomic on most OSes

    return target

  def load_state(self, prefix: str, folder: str | Path) -> None:
    """
    Load merges+vocab from `<folder>/<prefix>_tokenizer.json`
    and overwrite self.merges and self.vocab.

    Args:
        prefix:       the prefix used when saving (filename = prefix_tokenizer.json)
        folder:       path to the folder containing that file

    Raises:
        FileNotFoundError: if the JSON file doesn't exist
        NotADirectoryError: if `folder` exists but isn't a directory
        ValueError: if the JSON version is unsupported or malformed
    """
    # 1) Resolve folder
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists() or not folder_path.is_dir():
      raise NotADirectoryError(f'{folder!r} is not an existing directory')
    # 2) Build & check filepath
    safe_prefix = Path(prefix).name
    filepath = folder_path / f'{safe_prefix}_tokenizer.json'
    if not filepath.is_file():
      raise FileNotFoundError(f'Tokenizer state file not found: {filepath}')

    # 3) Load JSON
    with open(filepath, 'r', encoding='utf-8') as f:
      data = json.load(f)

    # 4) Version check
    version = data.get('version')
    if version != 1:
      raise ValueError(f'Unsupported tokenizer state version: {version!r}')

    # 5) Restore merges
    #    Expecting [[i,j], [k,l], …]
    self.merges = [tuple(pair) for pair in data['merges']]

    # 6) Restore vocab
    #    Stored as { "123": "6162", … } where "6162" is hex of b"ab"
    vocab = {}
    for tok_id_str, hexstr in data['vocab'].items():
      try:
        tok_id = int(tok_id_str)
      except ValueError:
        raise ValueError(f'Invalid token ID in state file: {tok_id_str!r}')
      try:
        vocab[tok_id] = bytes.fromhex(hexstr)
      except ValueError:
        raise ValueError(f'Invalid hex for token {tok_id}: {hexstr!r}')
    self.vocab = vocab
