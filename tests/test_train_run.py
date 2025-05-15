import json
import sys
from pathlib import Path

import pytest

import toksmith.train as train_mod


@pytest.fixture(autouse=True)
def dummy_tokenizer(monkeypatch):
  """
  Replace train_mod.Tokenizer with a dummy that records train/save calls
  and writes out a placeholder JSON file.
  """
  calls = {}

  class DummyTokenizer:
    def __init__(self):
      # capture the instance
      calls['inst'] = self

    def train(self, text, vocab_size, special_tokens, verbose):
      # record exactly what we were passed
      calls['inst'].train_args = {
        'text': text,
        'vocab_size': vocab_size,
        'special_tokens': special_tokens,
        'verbose': verbose,
      }

    def save_state(self, prefix, output_dir):
      # record and write a dummy file
      calls['inst'].save_args = {
        'prefix': prefix,
        'output_dir': output_dir,
      }
      out_dir = Path(output_dir)
      out_dir.mkdir(parents=True, exist_ok=True)
      out_file = out_dir / f'{prefix}_tokenizer.json'
      # write minimal valid JSON
      out_file.write_text(json.dumps({'dummy': True}), encoding='utf-8')
      return out_file

  # Monkey-patch the Tokenizer class in the tokenizer module
  monkeypatch.setattr(train_mod, 'Tokenizer', DummyTokenizer)
  return calls


def test_run_happy_path(tmp_path, dummy_tokenizer, monkeypatch):
  # 1) Create a tiny UTF-8 input file
  input_file = tmp_path / 'input.txt'
  input_file.write_text('hello world', encoding='utf-8')

  # 2) Prepare args: -i, -N, -s, -p, -vv, -o
  out_dir = tmp_path / 'outdir'
  prefix = 'myprefix'
  # fmt: off
  argv = [
    'train.py',
    '-i', str(input_file),
    '-N', '300',
    '-s', '<bos>', '<eos>',
    '-p', prefix,
    '-vv',
    '-o', str(out_dir),
  ]
  # fmt: on
  monkeypatch.setattr(sys, 'argv', argv)

  # 3) Run
  args = train_mod.parse_args()
  code = train_mod.run(args)
  assert code == 0, 'run() should return 0 on success'

  inst = dummy_tokenizer['inst']

  # 4) Check that train() got the expected args
  train_args = inst.train_args
  assert train_args['text'] == 'hello world'
  assert train_args['vocab_size'] == 300
  assert train_args['special_tokens'] == ['<bos>', '<eos>']
  assert train_args['verbose'] is True

  # 5) Check that save_state() wrote the file with the right prefix & folder
  save_args = inst.save_args
  assert save_args['prefix'] == prefix
  assert Path(save_args['output_dir']).resolve() == out_dir.resolve()

  # 6) Ensure the dummy JSON file actually exists
  out_file = out_dir / f'{prefix}_tokenizer.json'
  assert out_file.is_file()
  data = json.loads(out_file.read_text(encoding='utf-8'))
  assert data == {'dummy': True}


def test_run_fails_on_missing_input(tmp_path, monkeypatch):
  # Missing input file → run() should log an error and return non-zero
  missing = tmp_path / 'no_such.txt'
  argv = ['train.py', '-i', str(missing), '-N', '300']
  monkeypatch.setattr(sys, 'argv', argv)
  args = train_mod.parse_args()
  code = train_mod.run(args)
  assert code != 0
