import pytest

from toksmith.merger import _get_pair_stats


# test _get_pair_stats
@pytest.mark.parametrize(
  'pretoken_count, expected',
  [
    # 1) Empty pretoken counter
    (dict(), (dict(), dict())),
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
