"""Single/multiprocess pretoken counter implementation"""

import multiprocessing
from collections import Counter
from collections.abc import Iterable, Iterator

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


# --- multiprocessing -----------------------------------------------------
def count_tokens_multi(
  text_iter: Iterable[str],
  n_proc: int = 4,
  n_chunks: int = 1,
) -> Counter[tuple[int, ...]]:
  """
  Multiprocessing version of our pre-tokenizer
  Given that `text_iter` is the chunkated `text`
  this function must produce the same result as
  `count_token_single(text)`
  n_proc: number of processes to use (positive integer)
  n_chunks: number of chunks to transfer to process at once (positive integer)
  """
  total = Counter()
  with multiprocessing.Pool(processes=n_proc) as pool:
    for chunk in pool.imap_unordered(count_tokens, text_iter, chunksize=n_chunks):
      total.update(chunk)
  return Counter({tuple(tok.encode('utf-8')): cnt for tok, cnt in total.items()})


# --- generate chunks of text from file -------------------------------------
def generate_text_chunks(
  file_path: str,
  delimiter: regex.Pattern[str],
  chunk_size: int = 4096,
  overlap_size: int = 512,
) -> Iterator[str]:
  """
  Yields text chunks from the file, splitted by delimiter

  Note: delimiter is stripped off of the chunk. It is a caller
  responsibility to provide a valid path to the text file, which
  contains `delimiter` we use for splitting the text
  """
  pass
