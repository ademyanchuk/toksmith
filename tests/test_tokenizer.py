import pytest

from toksmith.tokenizer import Tokenizer, _merge


@pytest.fixture
def tok():
  # This will compile GPT2_SPLIT_PAT internally
  return Tokenizer()


# test _pretoken_count function
@pytest.mark.parametrize(
  'text, expected',
  [
    # 1) A single two‐byte letter group "hé"
    ('hé', {tuple('hé'.encode('utf-8')): 1}),
    # 2) ASCII words + space: "hello" and " world"
    (
      'hello world',
      {
        tuple('hello'.encode('utf-8')): 1,
        tuple(' world'.encode('utf-8')): 1,
      },
    ),
    # 3) Contraction: splits into ["it", "'s"]
    (
      "it's",
      {
        tuple('it'.encode('utf-8')): 1,
        tuple("'s".encode('utf-8')): 1,
      },
    ),
    # 4) Mixed letters + digits
    (
      'test123',
      {
        tuple('test'.encode('utf-8')): 1,
        tuple('123'.encode('utf-8')): 1,
      },
    ),
    # 5) Emoji + surrounding spaces
    (
      ' A🌟B ',
      {
        tuple(' A'.encode('utf-8')): 1,
        tuple('🌟'.encode('utf-8')): 1,
        tuple('B'.encode('utf-8')): 1,
        tuple(' '.encode('utf-8')): 1,
      },
    ),
  ],
)
def test_pretoken_count_unicode(tok, text, expected):
  """
  Ensure that _pretoken_count:
    - Groups letters (including accents) as single pretokens
    - Keeps leading spaces when the pattern dictates
    - Splits numbers, punctuation, emojis correctly
  """
  result = tok._pretoken_count(text)
  assert result == expected


# test _pairs_count
def test_pairs_count_unit_manual(tok):
  """
  Unit-test _pairs_count in isolation:
    - pretend we have two pretokens: (1,2,3) occurring twice, and (3,4) once
    - so pairs are:
         (1,2): 2
         (2,3): 2
         (3,4): 1
  """
  manual_pretoks = {
    (1, 2, 3): 2,
    (3, 4): 1,
  }
  expected = {
    (1, 2): 2,
    (2, 3): 2,
    (3, 4): 1,
  }
  result = tok._pairs_count(manual_pretoks)
  assert result == expected


# test _pretoken_count integration with _pairs_count
@pytest.mark.parametrize(
  'text, expected_pairs',
  [
    # "abc" in UTF-8 is b'abc' → bytes [97,98,99]
    # so zip → (97,98),(98,99), each once
    (
      'abc',
      {
        (97, 98): 1,
        (98, 99): 1,
      },
    ),
    # "éé" in UTF-8 is b'\xc3\xa9\xc3\xa9' → [195,169,195,169]
    # so zip → (195,169),(169,195),(195,169) → counts 2,1
    (
      'éé',
      {
        (195, 169): 2,
        (169, 195): 1,
      },
    ),
  ],
)
def test_pairs_count_integration(tok, text, expected_pairs):
  """
  Integration test: run the real regex→pretoks→pairs pipeline.
  """
  pretoks = tok._pretoken_count(text)
  pairs = tok._pairs_count(pretoks)
  assert pairs == expected_pairs


# test merge
@pytest.mark.parametrize(
  'seq, pair, new_ix, expected',
  [
    # 1) Empty sequence → empty
    ((), (1, 2), 3, ()),
    ([], (1, 2), 3, ()),
    # 2) Length 1 sequence → no possible pair
    ((1,), (1, 2), 3, (1,)),
    (['a'], ('a', 'b'), 'z', ('a',)),
    # 3) Single exact match → collapse to new_ix
    ((1, 2), (1, 2), 99, (99,)),
    (['x', 'y'], ('x', 'y'), 'z', ('z',)),
    # 4) No match anywhere → identity
    ((1, 2, 3), (4, 5), 0, (1, 2, 3)),
    ((0, 0, 0), (1, 1), 2, (0, 0, 0)),
    # 5) Multiple non‐overlapping matches
    ((1, 2, 1, 2), (1, 2), 9, (9, 9)),
    ('abab', ('a', 'b'), 'Z', ('Z', 'Z')),
    # 6) Overlapping matches: (x,x,x) with pair (x,x)
    ((7, 7, 7), (7, 7), 0, (0, 7)),
    ((42, 42, 42, 42), (42, 42), -1, (-1, -1)),
    # 7) Adjacent but non‐overlapping: (a,b,a)
    (('a', 'b', 'a'), ('a', 'b'), 'X', ('X', 'a')),
  ],
)
def test_merge_various(seq, pair, new_ix, expected):
  """
  Defensive tests for:
    - empty inputs
    - no matches
    - single & multiple matches
    - overlapping & adjacent cases
    - mixed types & edge values
  """
  result = _merge(seq, pair, new_ix)
  assert isinstance(result, tuple), 'Should always return a tuple'
  assert result == expected


def test_merge_original_unchanged():
  """Ensure original sequence is not mutated if it's a list."""
  orig = [1, 2, 3, 4]
  seq_copy = orig.copy()
  _ = _merge(orig, (2, 3), 99)
  assert orig == seq_copy, 'Input sequence must not be modified in place'


def test_invalid_pair_length():
  """Passing a pair that isn’t of length 2 should raise."""
  with pytest.raises(ValueError):
    _merge((1, 2, 3), (1,), 0)  # type: ignore
  with pytest.raises(ValueError):
    _merge((1, 2, 3), (1, 2, 3), 0)  # type: ignore
