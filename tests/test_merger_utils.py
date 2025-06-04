import pytest

from toksmith.merger import _get_pair_stats


# test _get_pair_stats
@pytest.mark.parametrize(
  'pretoken_count, expected',
  [
    # 1) Empty pretoken counter
    (dict(), (dict(), dict())),
    # 2) Single unit pretoken
    ({(42,): 2}, (dict(), dict())),
    # 3) Happy case
    (
      {(1, 2, 3): 2, (2, 3): 3},
      ({(1, 2): 2, (2, 3): 5}, {(1, 2): {(1, 2, 3)}, (2, 3): {(1, 2, 3), (2, 3)}}),
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
  result = _get_pair_stats(pretoken_count)
  assert result == expected
