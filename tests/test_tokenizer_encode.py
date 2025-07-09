import pytest

from toksmith.tokenizer import Encoder, _encode_iterable, encode_pretoken, get_lowest_rank_pair


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


# test token generator function
def test_generate_tokens_with_specials():
  """Test if we yield correct tokens in correct order"""
  enc = Encoder({}, special={'<|endoftext|>': 777, '<|endoffile|>': 888})
  example = 'Hello <|endoftext|><|endoffile|> 🙂'
  assert list(enc._generate_tokens(example)) == ['Hello', ' ', '<|endoftext|>', '<|endoffile|>', ' 🙂']


def test_generate_tokens_no_specials():
  """Test if we yield correct tokens in correct order (no specials)"""
  enc = Encoder({}, special={})
  example = 'Hello <|endoftext|> 🙂'
  assert list(enc._generate_tokens(example)) == ['Hello', ' <|', 'endoftext', '|>', ' 🙂']


# test end-to-end encode
def test_encode():
  """Test full encoding pipeline"""
  text = 'abc bcd<|endoffile|> '
  # (a, b) -> 256, (b, c) -> 257, (' ', 'bc') -> 258
  merges = {(97, 98): 256, (98, 99): 257, (32, 257): 258}
  special = {'<|endoffile|>': 777}
  # encode -> [('ab')=256, ('c')=99, (' bc')=258, ('d')=100, (special)=777, (' ')=32]
  enc = Encoder(merges, special)
  assert enc.encode(text) == [256, 99, 258, 100, 777, 32]
