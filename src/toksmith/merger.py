"""Merger implements logic for a Tokenizer training step
and keeps required state updated. It requires pretoken counter,
and doesn't do i/o or regex-grouping itself"""


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
    or later merges) to integer (count)
    """
    self.pretoken_count = pretoken_count
    self.pair_count, self.pair_to_pretoken_set = _get_pair_stats(pretoken_count)


def _get_pair_stats(pretoken_count: dict[tuple[int, ...], int]):
  """
  Given pretoken counter returns 2 dicts:
  - `pair_count` - simple counter of adjacent pairs of tokens
  - `pair_to_pretoken_set` - adjacent pair of tokens -> set of pretokens containing it
  """

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
