# test count_token with same examples as for basic implementation
import pytest

from toksmith.pretokenizer import count_tokens, count_tokens_multi, count_tokens_single


# test worker count_tokens
@pytest.mark.parametrize(
  'text, expected',
  [
    # 1) A single two‐byte letter group "hé"
    ('hé', {'hé': 1}),
    # 2) ASCII words + space: "hello" and " world"
    (
      'hello world',
      {
        'hello': 1,
        ' world': 1,
      },
    ),
    # 3) Contraction: splits into ["it", "'s"]
    (
      "it's",
      {
        'it': 1,
        "'s": 1,
      },
    ),
    # 4) Mixed letters + digits
    (
      'test123',
      {
        'test': 1,
        '123': 1,
      },
    ),
    # 5) Emoji + surrounding spaces
    (
      ' A🌟B ',
      {
        ' A': 1,
        '🌟': 1,
        'B': 1,
        ' ': 1,
      },
    ),
  ],
)
def test_count_tokens_unicode(text, expected):
  """
  Ensure that _pretoken_count:
    - Groups letters (including accents) as single pretokens
    - Keeps leading spaces when the pattern dictates
    - Splits numbers, punctuation, emojis correctly
  """
  result = count_tokens(text)
  assert result == expected


# test single process count_tokens
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
def test_pretoken_count_single_process(text, expected):
  """
  Ensure that _pretoken_count:
    - Groups letters (including accents) as single pretokens
    - Keeps leading spaces when the pattern dictates
    - Splits numbers, punctuation, emojis correctly
  """
  result = count_tokens_single(text)
  assert result == expected


# test that single process and multiprocess version produce the same output
def test_single_and_multi():
  """Same input to both versions produces the same output"""
  text_iter = ['hello there second with äöß', ' just third  last! 123and me']
  text = ''.join(text_iter)
  s_res = count_tokens_single(text)
  m_res = count_tokens_multi(text_iter, n_proc=2)
  assert s_res == m_res
