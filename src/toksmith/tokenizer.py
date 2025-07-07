"""BPE Tokenizer Implementation (follows gpt-2 assumptions)"""

import json
import logging
import multiprocessing
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Sequence, Tuple, TypeVar

import regex as re

from toksmith.merger import FastMerger
from toksmith.pretokenizer import count_tokens_multi, count_tokens_single, generate_text_chunks

T = TypeVar('T')

# helpers ============================


def _pairs_count(pretokens: dict[tuple[int, ...], int]) -> dict[tuple[int, int], int]:
  """Iterates through pre-tokens dict and produces
  the counter of all sequential pairs of units.
  Unit is an element of pre-token"""
  pair_counts = dict()
  for pt, cnt in pretokens.items():
    for p in zip(pt, pt[1:]):
      pair_counts[p] = pair_counts.get(p, 0) + cnt
  return pair_counts


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


## encode helpers ===============================


def get_lowest_rank_pair(
  pretoken: tuple[int, ...],
  pair_to_idx: dict[tuple[int, int], int],
) -> Optional[tuple[tuple[int, int], int]]:
  """Scans pretoken for the pair which exist in `pair_to_idx`
  and has lowest index, if none found returns None

  Args:
      pretoken (tuple[int]): sequence of integers
      pair_to_idx (dict[tuple[int, int], int]): map of the pair to index
      (it is merges converted to dict)

  Returns:
      Optional[tuple[tuple[int, int], int]]: lowest rank (pair, ix) or None
  """
  result = None
  for pair in zip(pretoken, pretoken[1:]):
    ix = pair_to_idx.get(pair)
    if ix is not None:
      if result is None or result[1] > ix:
        result = (pair, ix)
  return result


def encode_pretoken(
  pretoken: tuple[int, ...],
  pair_to_idx: dict[tuple[int, int], int],
) -> tuple[int, ...]:
  """Encodes single pretoken by consecutively merging pairs
  of tokens (if they are also in `pair_to_idx`). Merges occur
  in the same relative order as they did during BPE training

  Args:
      pretoken (tuple[int]): sequence of token ids
      pair_to_idx (dict[tuple[int, int], int]): maps pair of tokens ids
      to their corresponding merge id

  Returns:
      tuple[int]: encoded pretoken
  """
  # do we have a pair to merge
  pair_and_id = get_lowest_rank_pair(pretoken, pair_to_idx)
  # iterate while we have one
  while pair_and_id is not None:
    pair, idx = pair_and_id  # unpack
    pretoken = _merge(pretoken, pair, idx)  # we have pair to merge
    pair_and_id = get_lowest_rank_pair(pretoken, pair_to_idx)  # now check for a new pair to merge
  # encoded (if found any pair to merge) or the same
  return pretoken


def _encode_iterable(
  pretokenized_text: Iterable[str],
  pair_to_idx: dict[tuple[int, ...], int],
  special: dict[str, int],
) -> list[int]:
  """Applies encoding to sequence of strings `pretokenized_text`.
  Checks if element is special token first and maps directly to
  its token id. If not, performs proper encoding via iterative merging.

  Args:
      pretokenized_text (Iterable[str]): sequence of strings built with
      pre-tokenization
      pair_to_idx (dict[tuple[int, ...], int]): mapping of pair to it's
      corresponding merge idx
      special (dict[str, int]): mapping of special token to its token id

  Returns:
      list[int]: encoded `pretokenized_text`
  """
  encoded_text = []
  for s in pretokenized_text:
    if s in special:
      encoded_text.append(special[s])
    else:
      pretoken = tuple(bytes(s, encoding='utf-8'))
      encoded_text.extend(encode_pretoken(pretoken, pair_to_idx))
  return encoded_text


# tokenizer code ================================
logger = logging.getLogger(__name__)


class BasicMerger:
  """
  Implements merging of top pair in tne sequence of tokens
  in its `step` method. Refactored from Tokenizer `train` method
  to allow FastMerger drop-in
  """

  def __init__(self, pretoken_count: Counter[tuple[int, ...]]):
    """
    Initializes Merger state required to do merge (train) step.
    Requires caller to provide `pretoken_count` map of integer
    sequence (representing pretoken formed from first 256 utf-8 bytes
    or later merges) to integer (count). Note: Merger takes ownership of
    the `pretoken_count` dict and mutates it later during training
    """
    self.pretoken_count = pretoken_count

  def step(self, ix: int) -> Optional[tuple[int, int]]:
    """
    Implements merging of next top pair tokens into new index `ix`,
    return this top pair or None, if all pairs have been merged in
    the self.pretoken_count corpus
    """
    pair_counts = _pairs_count(self.pretoken_count)
    if not pair_counts:
      # all sequences in pretoken_count have been merged already
      return None  # for small text examples with large vocab size
    # find most frequent pair, ties resolved in lexicographical order
    top_pair, _ = max(pair_counts.items(), key=lambda it: [it[1], it[0]])
    # merge
    # Each merge introduces a new token (pair → new token) that wasn’t in the vocabulary before
    # Pretoken keys are sequences of current tokens.
    # Until you merge ('t', 'e') into 'te', there's no way 'te' appears as a unit inside any key
    # Only keys that contain the exact pair ('t', 'e') in adjacent positions will be modified.
    # The output of merge() depends deterministically on the input key.
    # Therefore, at most one original key can produce any given new_pt in the merge step.
    for pt in list(self.pretoken_count):  # static copy of keys (prevents RuntimeError if we iterate original dict)
      new_pt = _merge(pt, top_pair, ix)
      if new_pt != pt:  # update only if we merged new index
        # even though we proved it can't happen (see above), we want this assertions and perhaps test against it
        # so we are sure not to mess up with implementation
        assert new_pt not in self.pretoken_count, f'Collision: {new_pt} already in pretokens'
        self.pretoken_count[new_pt] = self.pretoken_count.pop(pt)  #  safe from key collisions under the BPE merge assumptions
    return top_pair


class Tokenizer:
  """GPT2-like tokenizer, using BPE algorithm"""

  def __init__(self) -> None:
    self._reset_state()

  def _reset_state(self):
    self.vocab = {i: int.to_bytes(i) for i in range(256)}
    self.merges = []

  def train(
    self,
    text: str,
    vocab_size: int,
    special_tokens: list[str],
    use_fast_merge: bool = False,
    verbose: bool = False,
  ) -> None:
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
      # Note: it can accidentally smash tokens together, if two
      # parts only separated by special token
      text = re.sub(f'(?:{delim})+', '', text)
    pretokens = count_tokens_single(text)
    Merger = BasicMerger
    if use_fast_merge:
      Merger = FastMerger
    merger = Merger(pretokens)
    ix = 256
    num_iters = vocab_size - min_size  # keep vocab space for special tokens
    for _ in range(num_iters):
      top_pair = merger.step(ix)
      if top_pair is None:
        logger.debug(f'Early stop at {ix=}, no more pairs to merge!')
      self.merges.append(top_pair)
      self.vocab[ix] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
      if verbose:
        logger.debug(
          'Merged %d: %r + %d: %r -> %d: %r',
          top_pair[0],
          self.vocab[top_pair[0]],
          top_pair[1],
          self.vocab[top_pair[1]],
          ix,
          self.vocab[ix],
        )
      # increment index
      ix += 1
    # finished training, let't add special tokens to vocab
    for i, s_tok in enumerate(special_tokens, ix):
      self.vocab[i] = bytes(s_tok, encoding='utf-8')

  def train_from_file(
    self,
    file_path: str,
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = False,
  ) -> None:
    """
    Same training routine but it trains from file and fast by default

    Any existing merges or vocab entries beyond the initial 256 byte-vocab
    will be cleared before training begins.

    Note: special token/tokens are required for this code to run fast, as we
    use it for splitting the text before pre-tokenization. Without special tokens
    to split by, it will run the same as `train` using fast merger

    Note: special token occurrences will be stripped off from the text before
    training

    Args:
        file_path (str): path to utf-8 encoded text file
        vocab_size (int): total size after training (== 256 + #merges + #special_tokens)
        special_tokens (list[str]): list of special tokens, i.e. <|endoftext|>

    Raises:
        ValueError: if vocab_size < # init bytes + # special tokens
    """
    # ensure clean state before training
    self._reset_state()
    # check correct arguments
    min_size = 256 + len(special_tokens)
    if vocab_size < min_size:
      raise ValueError(f'vocab_size must be >= {min_size}')
    if not special_tokens:
      raise ValueError('at least one special token required')
    # setup stage
    # join and create regex pattern
    delim = '|'.join(map(re.escape, special_tokens))
    delim = f'(?:{delim})+'
    # Pre-tokenization
    chunk_gen = generate_text_chunks(file_path, delim, chunk_size=4096, overlap_size=len(delim) + 64)
    pretokens = count_tokens_multi(chunk_gen, n_proc=multiprocessing.cpu_count(), n_chunks=100)
    # Instantiate merger, next index and number of iterations
    merger = FastMerger(pretokens)
    ix = 256
    num_iters = vocab_size - min_size  # keep vocab space for special tokens
    # training stage
    for _ in range(num_iters):
      top_pair = merger.step(ix)
      if top_pair is None:
        logger.debug(f'Early stop at {ix=}, no more pairs to merge!')
      self.merges.append(top_pair)
      self.vocab[ix] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
      if verbose:
        logger.debug(
          'Merged %d: %r + %d: %r -> %d: %r',
          top_pair[0],
          self.vocab[top_pair[0]],
          top_pair[1],
          self.vocab[top_pair[1]],
          ix,
          self.vocab[ix],
        )
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
