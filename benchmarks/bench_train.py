"""
Benchmark bpe training routine:
 - runs on TinyStories train data (2GB file)
 - with vocabulary size = 10000

Use:
  `uv run python -m pyperf timeit -p 1 -n 1 -l 1 -w 0 -s "from benchmarks.bench_train import bench" "bench(arg)" -o bench.json`
  `arg` must be one of 'base', 'merge', 'full'
  `base` - naive implementation of bpe training
  `merge` - optimized merge subroutine (see merger.py for details)
  `full` - same as merge + parallel pre-tokenization (see pretokenizer.py for details)
"""

from pathlib import Path

from toksmith.tokenizer import Tokenizer

PROJ_DIR = Path(__file__).parent.parent
INP_FILE = (PROJ_DIR / 'data' / 'TinyStoriesV2-GPT4-train.txt').resolve()

VOCAB_SIZE = 10000
SPECIAL_TOKENS = ['<|endoftext|>']


def bench(impl: str):
  assert impl in ['base', 'merge', 'full'], f'{impl=} is invalid!'
  tok = Tokenizer()
  if impl == 'base':
    text = INP_FILE.read_text(encoding='utf-8')
    tok.train(text, VOCAB_SIZE, SPECIAL_TOKENS, use_fast_merge=False)
  elif impl == 'merge':
    text = INP_FILE.read_text(encoding='utf-8')
    tok.train(text, VOCAB_SIZE, SPECIAL_TOKENS, use_fast_merge=True)
  elif impl == 'full':
    tok.train_from_file(INP_FILE, VOCAB_SIZE, SPECIAL_TOKENS)
