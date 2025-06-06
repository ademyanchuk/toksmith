import heapq
from collections import Counter

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
