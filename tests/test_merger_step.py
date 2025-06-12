import heapq
from collections import Counter

import pytest

from toksmith.merger import FastMerger, HeapEntry

# test _most_common_pair


# simple case with real init
def test_most_common_pair_after_init():
  # 1) Build a tiny pretoken_count so that pair_counts are easy to predict
  #    Suppose pretoken_count={(1,2,3):2, (2,3):1} → (2,3) has freq 3, (1,2) has 2.
  pretoken_count = Counter(
    {
      (1, 2, 3): 2,
      (2, 3): 1,
    }
  )
  merger = FastMerger(pretoken_count)

  # 2) Now call the helper
  top_pair, top_count = merger._most_common_pair()

  assert top_pair == (2, 3)
  assert top_count == 3

  # 3) Second call should give (1,2) and 2
  again_pair, again_count = merger._most_common_pair()
  assert again_pair == (1, 2)
  assert again_count == 2


def test_most_common_pair_skips_stale_entries():
  """
  Ensure that if a stale heap entry (one whose stored count no longer matches
  self.pair_count) is pushed, _most_common_pair skips it and returns the next valid pair.
  """
  # 1) Bypass __init__ and construct a FastMerger with three pairs:
  #    (10,20) count=5, (30,40) count=5, (50,60) count=3.
  fm = object.__new__(FastMerger)
  fm.pair_count = Counter(
    {
      (10, 20): 5,
      (30, 40): 5,
      (50, 60): 3,
    }
  )
  # Build the corresponding heap entries and heapify:
  initial_entries = [
    HeapEntry(5, (10, 20)),  # ties with (30,40) at count=5
    HeapEntry(5, (30, 40)),  # lex‐larger than (10,20), so should come out first
    HeapEntry(3, (50, 60)),
  ]
  heapq.heapify(initial_entries)
  fm.pair_heap = initial_entries

  # 2) First call → should return (30,40) because 5 == 5 and (30,40) > (10,20)
  pair1, count1 = fm._most_common_pair()
  assert pair1 == (30, 40)
  assert count1 == 5

  # 3) Second call → should return (10,20)
  pair2, count2 = fm._most_common_pair()
  assert pair2 == (10, 20)
  assert count2 == 5

  # 4) Now mutate pair_count to remove (10,20) so any heap entry for (10,20) is stale:
  del fm.pair_count[(10, 20)]

  # 5) Push a “stale” entry for (10,20) with a count of 4 onto the heap:
  #    In reality, pair_count no longer has (10,20), so this entry is stale.
  heapq.heappush(fm.pair_heap, HeapEntry(4, (10, 20)))

  # 6) Third call → _most_common_pair should pop the stale (10,20,4) entry,
  #    see that pair_count.get((10,20), 0) != 4 (it’s 0), skip it, and then return (50,60).
  pair3, count3 = fm._most_common_pair()
  assert pair3 == (50, 60)
  assert count3 == 3


# test _update_pair helper


@pytest.fixture
def base_count():
  """Base pretoken count to initialize Merger"""
  # Effectively pairs being (1,2): 2 and (2,3): 2+3
  return Counter({(1, 2, 3): 2, (2, 3): 3})


def test_update_pair_add_new(base_count):
  """Test adding state for new pair"""
  fm = FastMerger(base_count)
  pair = (256, 1)
  freq = 4
  fm._update_pair(pair, freq)
  # check correct state of pair counter
  assert fm.pair_count[pair] == freq
  # check pair heap has corresponding entry
  assert HeapEntry(freq, pair) in fm.pair_heap


def test_update_pair_increment(base_count):
  """Test existing pair count correctly increments"""
  fm = FastMerger(base_count)
  pair = (1, 2)  # exists with count 2
  freq = 2
  fm._update_pair(pair, freq)
  # check that count is adjusted
  assert fm.pair_count[pair] == freq + 2
  # check updated entry make it to the heap
  assert HeapEntry(freq + 2, pair) in fm.pair_heap


def test_update_pair_decrement(base_count):
  """Test existing pair count correctly decrements"""
  fm = FastMerger(base_count)
  pair = (2, 3)  # exists with count 5
  freq = -3
  fm._update_pair(pair, freq)
  # check that count is adjusted
  assert fm.pair_count[pair] == 5 + freq
  # check entry is in the heap
  assert HeapEntry(5 + freq, pair) in fm.pair_heap
  # check pair is in adjacency dict
  assert pair in fm.pair_to_pretoken_set


def test_update_pair_decrement_delete(base_count):
  """Test existing pair decrements and being deleted"""
  fm = FastMerger(base_count)
  pair = (1, 2)  # exists with count 2
  freq = -2
  fm._update_pair(pair, freq)
  # check pair is not in the counter
  assert pair not in fm.pair_count
  # check stale entry is still in the heap
  assert HeapEntry(2, pair) in fm.pair_heap
  # check pair is not in adjacency dict
  assert pair not in fm.pair_to_pretoken_set


def test_update_pair_no_op():
  """Test with freq==0 does nothing"""
  fm = FastMerger(Counter())  # just empty for simplicity
  fm._update_pair((1, 2), 0)
  assert not fm.pair_count
  assert not fm.pair_heap


# Micro-tests for _merge_sequence


def test_merge_sequence():
  """Test a happy path"""
  pretoken_count = Counter({(1, 2, 3, 4): 2, (2, 3): 1})
  # building a FastMerger from it gives us following state:
  # pair_count = {(1,2): 2, (2,3): 3, (3,4): 2}
  # pair_heap ~ [(3, (2, 3)), (2, (1, 2)), (2, (3, 4))]
  # pair_to_pretoken_set = {(1, 2): {(1,2,3,4)}, (2,3): {(1,2,3,4),(2,3)}, (3,4): {(1,2,3,4)}}
  fm = FastMerger(pretoken_count)
  old_seq = (1, 2, 3, 4)
  seq_freq, top_pair, new_ix = 2, (2, 3), 99
  fm._merge_sequence(old_seq, seq_freq, top_pair, new_ix)
  # we expect
  expect_new_seq = (1, 99, 4)
  # old sequence is not in pretoken count and new is in with correct frequency
  expect_pretoken_count = Counter({expect_new_seq: seq_freq, (2, 3): 1})
  assert fm.pretoken_count == expect_pretoken_count
  # (1,2) and (3,4) are not in pair_count and pair_to_pretoken_set anymore
  # (1,99) and (99,4) are in those data structures
  # top_pair (2,3) state is properly updated
  expect_pair_count = Counter({top_pair: 1, (1, 99): 2, (99, 4): 2})
  assert fm.pair_count == expect_pair_count
  expect_pair_to_pretoken = {(2, 3): {(2, 3)}, (1, 99): {expect_new_seq}, (99, 4): {expect_new_seq}}
  assert fm.pair_to_pretoken_set == expect_pair_to_pretoken
  # the function doesn't pop stale entries from the heap, so we expect only 3 new entries
  assert len(fm.pair_heap) == 6
  for entry in [HeapEntry(1, top_pair), HeapEntry(2, (1, 99)), HeapEntry(2, (99, 4))]:
    assert entry in fm.pair_heap


def test_merge_sequence_two_merges():
  """Test correctness for sequence with two instances of top pair"""
  pretoken_count = Counter({(1, 2, 3, 1, 2, 3): 2})
  # FastMerger state:
  # pair_count = {(1,2):4, (2,3):4, (3,1):2}
  # pair_heap = [(4, (2,3)), (4, (1,2)), (2, (3,1))]
  # pair_to_pretoken_set = {(1,2):{seq}, (2,3):{seq}, (3,1):{seq}}
  fm = FastMerger(pretoken_count)
  old_seq = (1, 2, 3, 1, 2, 3)
  seq_freq, top_pair, new_ix = 2, (2, 3), 42
  fm._merge_sequence(old_seq, seq_freq, top_pair, new_ix)
  # we expect
  expect_new_seq = (1, 42, 1, 42)
  assert fm.pretoken_count == Counter({expect_new_seq: seq_freq})
  # pairs (1,2), (2,3), and (3,1) are gone
  # pairs (1,42) and (42,1) added
  assert fm.pair_count == Counter({(1, 42): 4, (42, 1): 2})  # for (1,42) 2 instances of pair * 2 instances of sequence
  assert fm.pair_to_pretoken_set == {(1, 42): {expect_new_seq}, (42, 1): {expect_new_seq}}
  # now to heap
  # first instance of top pair merge: incoming 1,42 and 42,1 with count 2 should be added,
  # outgoing 1,2 should have new entry with decremented count 4-2
  # second instance: incoming 1,42 should have new entry added with count 4
  # no new entries for outgoing pairs should be added as their counts drop to 0
  for entry in [HeapEntry(2, (1, 42)), HeapEntry(2, (42, 1)), HeapEntry(2, (1, 2)), HeapEntry(4, (1, 42))]:
    assert entry in fm.pair_heap
  assert len(fm.pair_heap) == 7
