# test count_token with same examples as for basic implementation
import pytest
import regex

from toksmith.pretokenizer import count_tokens, count_tokens_multi, count_tokens_single, generate_text_chunks


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


# test text chunk generator
@pytest.mark.parametrize(
  'text, delimiter, output',
  [
    # 1. Empty file -> empty generator
    ('', 'x', []),
    # 2. File size is <= overlap size -> yield all content in one chunk
    ('abcd', 'x', ['abcd']),
    # 3. Last read is <= overlap size -> yield pieces w/o delimiter
    ('abcdxabcd', 'x', ['abcd', 'abcd']),
    # 4. Multiple reads, no delimiter -> single piece
    ('1234abcd5678efgh', 'x', ['1234abcd5678efgh']),
    # 5. Delimiter in two pieces + leftover w/o delimiter
    ('12345<>8ab<>c', '<>', ['12345', '8ab', 'c']),
    # 6. Go to all branches
    ('1234.ab.Z', regex.escape('.'), ['1234', 'ab', 'Z']),
  ],
)
def test_generate_text_chunks(tmp_path, text, delimiter, output):
  file_path = tmp_path / 'tmp.txt'
  file_path.write_text(text, encoding='utf-8')
  text_gen = generate_text_chunks(file_path, delimiter, chunk_size=8, overlap_size=4)
  assert list(text_gen) == output


# real life scenario special tokens as delimiter
def test_generate_text_chunks_special_tokens(tmp_path):
  """Test with one or more special tokens"""
  special_tokens = ['<|eof|>', '<|eol|>']  # somewhat real special tokens
  first_chunk = 'some text for starters '
  second_chunk = 'I am the second part with 🌟 '
  third_chunk = 'ends here!'
  # text with unicode and punctuation, interspersed with special tokens
  # with one occasion of back to back special tokens
  text = f'{first_chunk}{special_tokens[0]}{second_chunk}{special_tokens[0]}{special_tokens[1]}{third_chunk}{special_tokens[1]}'
  # join special tokens into regex pattern with escaping
  delim = '|'.join(map(regex.escape, special_tokens))
  # and make a final pattern to be non-capturing group of one or more of those special tokens
  delim = f'(?:{delim})+'
  # write our text to temporary file
  file_path = tmp_path / 'tmp.txt'
  file_path.write_text(text, encoding='utf-8')

  text_gen = generate_text_chunks(file_path, delim, chunk_size=32, overlap_size=16)
  assert list(text_gen) == [first_chunk, second_chunk, third_chunk]
