"""Merger implements logic for a Tokenizer training step
and keeps required state updated. It requires pretoken counter,
and doesn't do i/o or regex-grouping itself"""

import heapq
from collections import Counter
from dataclasses import dataclass, field


############### utilities ##########################
def _build_pair_index(
  pretoken_count: dict[tuple[int, ...], int],
) -> tuple[
  Counter[tuple[int, int]],
  dict[tuple[int, int], set[tuple[int, ...]]],
]:
  """
  Given pretoken counter returns 2 dicts:
  - `pair_count` - simple counter of adjacent pairs of tokens
  - `pair_to_pretoken_set` - adjacent pair of tokens -> set of pretokens containing it
  """
  pair_count, pair_to_pretoken_set = Counter(), dict()
  for pt, freq in pretoken_count.items():
    _process_pretoken(pt, freq, pair_count, pair_to_pretoken_set)
  return pair_count, pair_to_pretoken_set


def _process_pretoken(
  pretoken: tuple[int, ...],
  freq: int,
  pair_count: Counter[tuple[int, int]],
  pair_to_pretoken: dict[tuple[int, int], set[tuple[int, ...]]],
):
  """
  Given sequence `pretoken` and its frequency updates pair counter and pair to
  pretoken adjacency dict
  """
  for pair in zip(pretoken, pretoken[1:]):
    pair_count[pair] += freq
    if pair in pair_to_pretoken:
      pair_to_pretoken[pair].add(pretoken)
    else:
      pair_to_pretoken[pair] = {pretoken}


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


################### Merger ##############################################
class FastMerger:
  """
  Merger implements Tokenizer training step, and is based on two optimizations:
  - don't recount all pairs all the time
  - catch top pair from max heap in log(n) time
  """

  def __init__(self, pretoken_count: Counter[tuple[int, ...]]):
    """
    Initializes Merger state required to do merge (train) step.
    Requires caller to provide `pretoken_count` map of integer
    sequence (representing pretoken formed from first 256 utf-8 bytes
    or later merges) to integer (count). Note: Merger takes ownership of
    the `pretoken_count` dict and mutates it later during training

    Invariants: counters and adjacency dict (pretoken counter, pair counter and
    pair to pretoken set) are not allowed <= 0 or empty set values
    """
    self.pretoken_count = pretoken_count
    self.pair_count, self.pair_to_pretoken_set = _build_pair_index(pretoken_count)
    self.pair_heap = [HeapEntry(count, pair) for pair, count in self.pair_count.items()]
    heapq.heapify(self.pair_heap)

  def _most_common_pair(self):
    """
    Helper to pop most frequent pair from Merger heap (destructive for self.pair_heap).
    It relies on the healthy state of pair counter and heap itself:
      - we must not be in this function if counter and/or heap are empty
      - heap entries can be stale due to pushes during merge phase
      - the final source of truth is pair counter
    """
    assert self.pair_count, 'Expected non-empty pair counter'
    assert self.pair_heap, 'Expected non-empty pair heap'
    while True:
      entry = heapq.heappop(self.pair_heap)
      pair, count = entry.original_pair, -entry.sort_index
      if self.pair_count[pair] == count:
        break  # fresh entry
      # else, stale entry, continue
    return pair, count

  def _update_pair(self, pair: tuple[int, int], freq: int):
    """
    Helper method to update pair state in `pair_count`, `pair_heap`, and `pair_to_pretoken_set`.
    It increments or decrements count depending on `freq` sign.
    Note: this method is destructive for `pair_count` and `pair_to_pretoken_set`
    as it removes entry with count <= 0
    """
    # freq == 0 -> no-op
    if freq == 0:
      return
    cnt = self.pair_count[pair] + freq
    if cnt > 0:  # update state
      self.pair_count[pair] = cnt
      heapq.heappush(self.pair_heap, HeapEntry(cnt, pair))
    else:  # drop entry
      # both datastructures are synced
      del self.pair_count[pair]
      self.pair_to_pretoken_set.pop(pair, None)

  def _merge_sequence(
    self,
    old_seq: tuple[int, ...],
    seq_freq: int,
    top_pair: tuple[int, int],
    new_ix: int,
  ) -> None:
    """
    Performs merge of `top_pair` in the `old_seq` substituting it
    with `new_ix`.
    Note: this function updates all required datastructures' state,
    except the state of `top_pair` itself as merge routine supposed
    to be called with the same `top_pair` on one or more `old_seq`
    sequences
    """
    # we want this to blow on `seq_freq` == 0, as we don't allow stale entries
    # with count == 0 in our pair counter and pair to pretoken set mapping
    # this indicate an implementation bug and we want to investigate
    assert seq_freq > 0, 'Stale entry came from pair_to_pretoken_set, not allowed!'
    new_builder = []
    i = 0

    while i < len(old_seq):
      # found top_pair position
      if i + 1 < len(old_seq) and (old_seq[i], old_seq[i + 1]) == top_pair:
        # grab neighbors, if any
        u = new_builder[-1] if new_builder else None
        v = old_seq[i + 2] if i + 2 < len(old_seq) else None

        # decrement counts of outgoing neighbor pairs, e.g. if top pair is (x,y)
        # we decrement u,x and y,v counts (in pair_count and pair_heap)
        # adn discard old_seq from outgoing pair set (in pair_to_pretoken_set)
        if u is not None:
          self._update_pair((u, top_pair[0]), -seq_freq)
          s = self.pair_to_pretoken_set[(u, top_pair[0])]
          if s is not None:
            s.discard(old_seq)
        if v is not None:
          self._update_pair((top_pair[1], v), -seq_freq)
          s = self.pair_to_pretoken_set[(top_pair[1], v)]
          if s is not None:
            s.discard(old_seq)

        # increment counts for incoming pairs
        if u is not None:
          self._update_pair((u, new_ix), seq_freq)
        if v is not None:
          self._update_pair((new_ix, v), seq_freq)

        # merge
        new_builder.append(new_ix)
        i += 2
      else:
        new_builder.append(old_seq[i])
        i += 1

    new_seq = tuple(new_builder)
    # remove old tok_seq from pretoken_count, add new_seq
    del self.pretoken_count[old_seq]
    self.pretoken_count[new_seq] = seq_freq

    # Important: update pair to pretoken set adjacency dict
    # Note: incoming pairs -> pretoken set mappings got updated here too
    for pair in zip(new_seq, new_seq[1:]):
      s = self.pair_to_pretoken_set.setdefault(pair, set())
      # add new_seq to all pairs forming it
      s.add(new_seq)
      # maybe remove old_seq from pairs forming new_seq
      s.discard(old_seq)
