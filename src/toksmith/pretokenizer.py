"""Single/multiprocess pretoken counter implementation"""

from collections import Counter

import regex

# --- pattern -----------------------------------------------------------


# from here: https://github.com/openai/tiktoken/pull/234/files
GPT2_SPLIT_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
TOKEN_RE = regex.compile(GPT2_SPLIT_PAT)


# --- single -------------------------------------------------------------
def count_tokens(text: str) -> Counter[str]:
  """
  Performs regex `TOKEN_RE` pattern matching and
  returns the counter of matched tokens in `text`
  """
  c = Counter()
  for m in TOKEN_RE.finditer(text):
    c[m.group()] += 1
  return c


def count_tokens_single(text: str) -> Counter[tuple[int, ...]]:
  """
  Pre-tokenizes the text in a single process and produces the counter
  of pre-tokens represented as tuple of utf-8 encoded bytes
  """
  c = count_tokens(text)
  return Counter({tuple(tok.encode('utf-8')): cnt for tok, cnt in c.items()})
