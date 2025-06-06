import heapq
from collections import Counter

import pytest

from toksmith.merger import HeapEntry, _build_pair_index, _process_pretoken


# test HeapEntry
def test_heapentry_orders_by_count():
  a = HeapEntry(10, (0, 0))
  b = HeapEntry(5, (9, 9))
  assert a < b  # because -10 < -5


def test_heapentry_tie_breaks_on_pair():
  # same count, pair (2,3) is lex‐larger than (2,2)
  c = HeapEntry(7, (2, 3))
  d = HeapEntry(7, (2, 2))
  assert c < d  # because reversed (−2,−3) < (−2,−2)


def test_heap_pop_sequence():
  entries = [
    HeapEntry(3, (1, 1)),
    HeapEntry(5, (0, 1)),
    HeapEntry(5, (0, 0)),
  ]
  heapq.heapify(entries)
  popped = [heapq.heappop(entries).original_pair for _ in range(3)]
  # Should pop 5, (0,1) first (lex-larger), then 5, (0,0), then 3, (1,1)
  assert popped == [(0, 1), (0, 0), (1, 1)]


# test _get_pair_stats
@pytest.mark.parametrize(
  'pretoken_count, expected',
  [
    # 1) Empty pretoken counter
    (Counter(), (Counter(), dict())),
    # 2) Single unit pretoken
    (Counter({(42,): 2}), (Counter(), dict())),
    # 3) Happy case
    (
      Counter({(1, 2, 3): 2, (2, 3): 3}),
      (
        Counter({(1, 2): 2, (2, 3): 5}),
        {(1, 2): {(1, 2, 3)}, (2, 3): {(1, 2, 3), (2, 3)}},
      ),
    ),
  ],
)
def test_get_pair_stats(pretoken_count, expected):
  """
  Ensure that _get_pair_stats:
    - Returns empty dicts for empty input
    - Doesn't produce pair stats for single unit pretoken
    - Produces correct pair stats from valid tiny input
  """
  result = _build_pair_index(pretoken_count)
  assert result == expected


# test _process_token
@pytest.fixture
def base_state():
  """
  Return a fresh pair_count and pair_to_pretoken with one existing pair (2,3)→count=5
  and that pair mapped to the pretoken (1,2,3).
  """
  return (Counter({(2, 3): 5}), {(2, 3): {(1, 2, 3)}})


def test_noop_on_empty_sequence(base_state):
  pair_count, pair_to_pretoken = base_state
  _process_pretoken((), 42, pair_count, pair_to_pretoken)

  # Nothing should have changed
  assert pair_count == Counter({(2, 3): 5})
  assert pair_to_pretoken == {(2, 3): {(1, 2, 3)}}


def test_noop_on_singleton_sequence(base_state):
  pair_count, pair_to_pretoken = base_state
  _process_pretoken((1,), 42, pair_count, pair_to_pretoken)

  # Still nothing should have changed
  assert pair_count == Counter({(2, 3): 5})
  assert pair_to_pretoken == {(2, 3): {(1, 2, 3)}}


def test_add_length_two_sequence(base_state):
  # (1,2) with freq=3 should create one new pair (1,2)→count=3
  pair_count, pair_to_pretoken = base_state
  _process_pretoken((1, 2), 3, pair_count, pair_to_pretoken)

  assert pair_count == Counter({(2, 3): 5, (1, 2): 3})
  assert pair_to_pretoken == {
    (2, 3): {(1, 2, 3)},
    (1, 2): {(1, 2)},
  }


def test_extend_existing_pair_and_add_new(base_state):
  # (1,2,3) with freq=2 should:
  #   - bump (2,3) from 5→7
  #   - create (1,2)→2
  pair_count, pair_to_pretoken = base_state
  _process_pretoken((1, 2, 3), 2, pair_count, pair_to_pretoken)

  assert pair_count == Counter({(2, 3): 7, (1, 2): 2})
  assert pair_to_pretoken == {
    (2, 3): {(1, 2, 3)},
    (1, 2): {(1, 2, 3)},
  }


def test_accumulate_multiple_sequences_for_same_pair():
  """
  If (2,3) already appears in two different sequences,
  they should both be recorded in the set.
  """
  pair_count = Counter({(2, 3): 5})
  pair_to_pretoken = {(2, 3): {(1, 2, 3)}}

  # Now add a second sequence (2,3,4) with freq=4
  _process_pretoken((2, 3, 4), 4, pair_count, pair_to_pretoken)

  # (2,3) count: 5 + 4 = 9
  # (3,4) count: new = 4
  assert pair_count == Counter({(2, 3): 9, (3, 4): 4})
  assert pair_to_pretoken == {
    (2, 3): {(1, 2, 3), (2, 3, 4)},
    (3, 4): {(2, 3, 4)},
  }
