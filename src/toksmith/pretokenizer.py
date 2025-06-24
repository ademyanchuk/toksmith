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
  buffer = ''  # overlapping part from the previously read chunk
  file_is_finished = False
  with open(file_path, 'rt', encoding='utf-8') as file:
    while not file_is_finished:
      new_read = file.read(chunk_size)
      if not new_read:
        file_is_finished = True
      current_chunk = buffer + new_read
      # the part of the chunk beyond effective size will become next buffer
      effective_size = len(current_chunk) - overlap_size
      # chunk is smaller than our dedicated buffer
      # this can be true at the end of the file:
      # 1. file is smaller than buffer size
      # 2. last buffer and new read is smaller than buffer size
      # so at the eof we need to let buffer utilization branch work
      if effective_size <= 0:
        buffer = current_chunk
        if not file_is_finished:
          continue
      # we have a chunk + overlapping part
      if not file_is_finished:
        cur_end = 0
        for match in regex.finditer(delimiter, current_chunk):
          start, end = match.span()
          # we form a segment to yield only if we found delimiter
          segment = current_chunk[cur_end:start]
          cur_end = end
          # and yield if it's non-empty (can happen with back to back delimiters)
          if segment:
            yield segment
          if start >= effective_size:  # rest is buffer
            break
        # rest should be accumulated as we only split by delimiter
        buffer = current_chunk[cur_end:]
      # eof, we want to yield what's left
      else:
        cur_end = 0
        for match in regex.finditer(delimiter, current_chunk):
          start, end = match.span()
          # as before yield segment only delimiter was found
          segment = current_chunk[cur_end:start]
          if segment:
            yield segment
          cur_end = end
        # we can still have something left if file doesn't end with delimiter
        if cur_end < len(current_chunk):
          # leftover or never got into the previous for loop
          yield current_chunk[cur_end:]
