"""Merger implements logic for a Tokenizer training step
and keeps required state updated. It requires pretoken counter,
and doesn't do i/o or regex-grouping itself"""

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class HeapEntry:
  """
  Helper class to use python min heap as max heap
  with additional tie-breaking logic for the second field (pair)
  """

  # sort_index is the FIRST field, so it is compared first in ordering
  sort_index: int

  # reverse_pair is compared next (if sort_index ties)
  reverse_pair: tuple[int, int] = field(compare=True)

  # original_pair is not used for comparisons, only stored for retrieval
  original_pair: tuple[int, int] = field(compare=False)

  def __init__(self, count: int, pair: tuple[int, int]):
    # We want largest count → smallest sort_index, so store -count
    object.__setattr__(self, 'sort_index', -count)

    # We want lex‐largest pair to compare “smaller,” so store (-p0, -p1)
    object.__setattr__(self, 'reverse_pair', (-pair[0], -pair[1]))

    # Keep the original pair around so we can read it back
    object.__setattr__(self, 'original_pair', pair)


class FastMerger:
  """
  Merger implements Tokenizer training step, and is based on two optimizations:
  - don't recount all pairs all the time
  - catch top pair from max heap in log(n) time
  """

  def __init__(self, pretoken_count: dict[tuple[int, ...], int]):
    """
    Initializes Merger state required to do merge (train) step.
    Requires caller to provide `pretoken_count` map of integer
    sequence (representing pretoken formed from first 256 utf-8 bytes
    or later merges) to integer (count). Note: Merger takes ownership of
    the `pretoken_count` dict and mutates it later during training
    """
    self.pretoken_count = pretoken_count
    self.pair_count, self.pair_to_pretoken_set = _build_pair_index(pretoken_count)
    self.pair_heap = [HeapEntry(count, pair) for pair, count in self.pair_count.items()]
    heapq.heapify(self.pair_heap)


def _build_pair_index(pretoken_count: dict[tuple[int, ...], int]):
  """
  Given pretoken counter returns 2 dicts:
  - `pair_count` - simple counter of adjacent pairs of tokens
  - `pair_to_pretoken_set` - adjacent pair of tokens -> set of pretokens containing it
  """
  pair_count, pair_to_pretoken_set = dict(), dict()
  for pt, freq in pretoken_count.items():
    _process_pretoken(pt, freq, pair_count, pair_to_pretoken_set)
  return pair_count, pair_to_pretoken_set


def _process_pretoken(
  pretoken: tuple[int, ...],
  freq: int,
  pair_count: dict[tuple[int, int], int],
  pair_to_pretoken: dict[tuple[int, int], set[tuple[int, ...]]],
):
  """
  Given sequence `pretoken` and its frequency updates pair counter and pair to
  pretoken adjacency dict
  """
  for pair in zip(pretoken, pretoken[1:]):
    pair_count[pair] = pair_count.get(pair, 0) + freq
    if pair in pair_to_pretoken:
      pair_to_pretoken[pair].add(pretoken)
    else:
      pair_to_pretoken[pair] = {pretoken}
