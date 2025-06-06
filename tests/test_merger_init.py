# tests/test_merger_init.py
import heapq
from collections import Counter

from toksmith.merger import FastMerger, HeapEntry


def test_fastmerger_init_builds_correct_pair_index():
  """
  Given a simple pretoken_count:
    {(1,2,3): 2, (2,3): 1}
  We expect:
    - pair_count:
        (1,2) -> 2
        (2,3) -> 3    (because 2 from (1,2,3) + 1 from (2,3))
    - pair_to_pretoken_set:
        (1,2) -> {(1,2,3)}
        (2,3) -> {(1,2,3), (2,3)}
    - pair_heap (max-heap on count, then lex order):
        top entry is (2,3) with count=3
        next entry is (1,2) with count=2
  """
  # 1) Prepare the initial pretokens
  pretoken_count = Counter(
    {
      (1, 2, 3): 2,
      (2, 3): 1,
    }
  )

  # 2) Initialize the FastMerger
  merger = FastMerger(pretoken_count)

  # 3) Check pair_count
  expected_pair_count = {
    (1, 2): 2,
    (2, 3): 3,
  }
  assert merger.pair_count == expected_pair_count

  # 4) Check pair_to_pretoken_set
  expected_pair_to_pretoken = {
    (1, 2): {(1, 2, 3)},
    (2, 3): {(1, 2, 3), (2, 3)},
  }
  assert merger.pair_to_pretoken_set == expected_pair_to_pretoken

  # 5) Check that the heap yields entries in the correct order
  #    We expect first to pop (2,3) with count=3, then (1,2) with count=2.
  heap_copy = list(merger.pair_heap)
  heapq.heapify(heap_copy)

  entry1 = heapq.heappop(heap_copy)
  assert isinstance(entry1, HeapEntry)
  assert entry1.original_pair == (2, 3)
  assert -entry1.sort_index == 3  # original count

  entry2 = heapq.heappop(heap_copy)
  assert isinstance(entry2, HeapEntry)
  assert entry2.original_pair == (1, 2)
  assert -entry2.sort_index == 2  # original count

  # 6) No more entries
  assert not heap_copy
