#!/usr/bin/env python3
"""
train.py

Train a BPE tokenizer on a UTF-8 text file.

This script demonstrates, profiles, and exposes a CLI for:
  1. Training your Tokenizer on a given text.
  2. Saving its state (merges + vocab).
  3. Profiling via Scalene by running:

       scalene scripts/train.py --input data.txt --vocab-size 5000

Usage:
  python scripts/train.py \
    --input path/to/text.txt \
    --vocab-size 5000 \
    [--special-tokens <tok1> <tok2> ...] \
    [--output-dir out/] \
    [--prefix mytok] \
    [-q | --quiet] \
    [-v | -vv]

Examples:
  # Basic run, default output-dir=out/, prefix=train
  python scripts/train.py -i data/train.txt -v 8000

  # With special tokens
  python scripts/train.py -i data/train.txt -v 8200 \
      --special-tokens <bos> <eos>

  # More verbosity (shows merges in real time)
  python scripts/train.py -i data/train.txt -v 3000 -vv
"""

import argparse
import codecs
import logging
import sys
from pathlib import Path

from toksmith.tokenizer import Tokenizer


def parse_args():
  p = argparse.ArgumentParser(prog='train.py', description='Train a BPE tokenizer on a UTF-8 text file.')
  p.add_argument('-i', '--input', required=True, help='Path to UTF-8 text file for training')
  p.add_argument('-N', '--vocab-size', type=int, required=True, help='Total vocab size (256 + #merges + #special_tokens)')
  p.add_argument('-s', '--special-tokens', nargs='*', default=[], metavar='TOK', help='List of special tokens to reserve (e.g. <bos> <eos>)')
  p.add_argument('-o', '--output-dir', default='out', help='Directory to save trained tokenizer state (default: out/)')
  p.add_argument('-p', '--prefix', default=None, help='Filename prefix for saved files (default: input file stem)')
  grp = p.add_mutually_exclusive_group()
  grp.add_argument('-q', '--quiet', action='store_true', help='Suppress all but errors')
  grp.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity (once for INFO, twice for DEBUG)')
  return p.parse_args()


def configure_logging(verbosity: int, quiet: bool):
  if quiet:
    level = logging.ERROR
  else:
    if verbosity >= 2:
      level = logging.DEBUG
    elif verbosity == 1:
      level = logging.INFO
    else:
      level = logging.WARNING

  logging.basicConfig(
    level=level,
    format='%(asctime)s %(levelname)-8s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
  )


def quick_utf8_check(path: Path, n_bytes: int = 4096):
  """
  Read up to n_bytes from path in binary, and try to UTF-8 decode
  incrementally. Raises UnicodeDecodeError if invalid.
  """
  decoder = codecs.getincrementaldecoder('utf-8')()
  with path.open('rb') as f:
    chunk = f.read(n_bytes)
    # try to decode just this chunk
    decoder.decode(chunk, final=False)
  # We don’t need to drain the whole file here—this is just a quick screen.


def validate_paths(input_path: Path, output_dir: Path) -> bool:
  # Input file
  if not input_path.is_file():
    logging.error('Input path %r is not a file.', str(input_path))
    return False

  # Quick UTF-8 check
  try:
    quick_utf8_check(input_path)
  except Exception as e:
    logging.error('Failed to read %r as UTF-8: %s', str(input_path), e)
    return False

  # Output directory
  if output_dir.exists():
    if not output_dir.is_dir():
      logging.error('Output path %r exists and is not a directory.', str(output_dir))
      return False
  else:
    try:
      output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
      logging.error('Could not create output directory %r: %s', str(output_dir), e)
      return False
  return True  # passed all validation steps


def run(args: argparse.Namespace) -> int:
  """Main script logic, refactored into own function for testability"""
  configure_logging(args.verbose, args.quiet)

  input_path = Path(args.input).expanduser().resolve()
  output_dir = Path(args.output_dir).expanduser().resolve()
  prefix_name = args.prefix or input_path.stem

  if not validate_paths(input_path, output_dir):
    return 1

  logging.info('Training tokenizer on %r', str(input_path))
  logging.info('Vocab size: %d (reserving %d special tokens)', args.vocab_size, len(args.special_tokens))
  if args.special_tokens:
    logging.info('Special tokens: %s', args.special_tokens)
  logging.debug('Saving to %r with prefix %r', str(output_dir), prefix_name)

  # Read and train
  text = input_path.read_text(encoding='utf-8')
  tok = Tokenizer()
  tok.train(
    text=text,
    vocab_size=args.vocab_size,
    special_tokens=args.special_tokens,
    verbose=(args.verbose >= 2),
  )
  # Save
  out_path = tok.save_state(prefix_name, output_dir)
  logging.info('Saved tokenizer state to %r', str(out_path))
  return 0


def main():
  args = parse_args()
  return run(args)


if __name__ == '__main__':
  sys.exit(main())
