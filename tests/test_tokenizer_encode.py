import pytest

from toksmith.tokenizer import _encode_iterable, encode_pretoken, get_lowest_rank_pair


# test function to find a proper pair to merge
@pytest.mark.parametrize(
  'pretoken, pair_to_idx, want',
  [
    ((1, 2, 3), dict(), None),  # no pair in dict -> None
    ((1,), {(1, 2): 5}, None),  # pretoken has no pair -> None
    ((1, 2, 3), {(1, 2): 10, (2, 3): 5}, ((2, 3), 5)),  # takes correct pair and index
  ],
)
def test_get_lowest_rank_pair(pretoken, pair_to_idx, want):
  got = get_lowest_rank_pair(pretoken, pair_to_idx)
  assert got == want


# test if we can encode one pretoken
@pytest.mark.parametrize(
  'pretoken, pair_to_idx, want',
  [
    ((1, 2, 3), dict(), (1, 2, 3)),  # identity with empty dict
    ((3,), {(1, 2): 3}, (3,)),  # identity with only one token in pretoken
    ((1, 2, 3), {(1, 2): 5, (2, 3): 10, (5, 3): 42}, (42,)),  # (1,2,3) -> (5,3) -> (42,)
  ],
)
def test_encode_pretoken(pretoken, pair_to_idx, want):
  got = encode_pretoken(pretoken, pair_to_idx)
  assert got == want


# test private encode iterable function
@pytest.mark.parametrize(
  'pretok_text, pair_to_idx, special, want',
  [
    # no merges, but one special token
    (['ab', '123'], dict(), {'123': 100}, [ord('a'), ord('b'), 100]),
    # empty iterable gives empty encoding list
    ([], {(1, 2): 3}, {'ab': 100}, []),
    # simple merge, special, more complex merge
    (
      ['ab', '<!>', 'abcd'],
      {(97, 98): 256, (98, 99): 257, (256, 99): 258},
      {'<!>': 300},
      [256, 300, 258, 100],
    ),
  ],
)
def test_encode_iterable_private(pretok_text, pair_to_idx, special, want):
  got = _encode_iterable(pretok_text, pair_to_idx, special)
  assert got == want
