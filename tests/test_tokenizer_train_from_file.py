import pytest

from toksmith.tokenizer import Tokenizer


# ---- raise value error ---------------------------
def test_raises_on_small_vocab():
  tok = Tokenizer()
  with pytest.raises(ValueError) as ei:
    tok.train_from_file('not_a_file', 255, ['<>'])
  assert 'vocab_size must be >' in str(ei.value)


def test_raises_no_special_tokens():
  tok = Tokenizer()
  with pytest.raises(ValueError) as ei:
    tok.train_from_file('not_a_file', 1000, [])
  assert 'at least one special token required' in str(ei.value)


# ----- wiki example as in the regular train --------------
def test_on_wiki_example(tmp_path):
  """
  Full-pipeline integration test on “aaabdaaabac”
  reflecting our tie-breaking rule:
    1) First merge:  "aa" -> Z  token at index 256
    2) Second merge: "Za" -> Y  token at index 257
    3) Third merge:  "Yb" -> X  token at index 258
    Then stop, since we asked for only 3 merges (vocab_size=259).
    see: https://en.wikipedia.org/wiki/Byte_pair_encoding#Example
  """
  text = 'aaabdaaabac'
  file_path = tmp_path / 'tmp.txt'
  file_path.write_text(text, encoding='utf-8')
  tok = Tokenizer()

  # Request exactly 3 merges beyond the 256 base bytes, + 1 for one special token
  # we want to ensure same behavior as slow `train`
  tok.train_from_file(file_path, vocab_size=256 + 4, special_tokens=['tok'])

  # Expect exactly those three merges—in byte‐pair form:
  first = (ord('a'), ord('a'))  # "aa"
  second = (256, ord('a'))  # "Za"
  third = (257, ord('b'))  # "Yb"
  assert tok.merges == [first, second, third]

  # And the new vocab entries should be:
  # index=256 => "aa", index=257 => "Za", index=258 => "Yb"
  assert tok.vocab[256] == b'aa'
  assert tok.vocab[257] == b'aaa'
  assert tok.vocab[258] == b'aaab'


# ----- handles special token + escapes properly ---------------------------
def test_train_strips_and_merges_with_special_token(tmp_path):
  """
  Given text="ab<|tok|>ab" and special_tokens=["<|tok|>"], with vocab_size=258:
   - We reserve 256 base bytes, 1 merge, 1 special token.
   - After stripping <|tok|>, text → "abab", so the single most-frequent pair is ("a","b").
   - We expect one merge: ("a","b") → idx=256, and then special token at idx=257.
  """
  text = 'ab<|tok|>ab'
  fp = tmp_path / 'tmp.txt'
  fp.write_text(text, encoding='utf-8')
  special = ['<|tok|>']
  vocab_size = 256 + 1 + len(special)  # one merge + one special token

  tok = Tokenizer()
  # run training
  tok.train_from_file(fp, vocab_size=vocab_size, special_tokens=special)

  # 1 merge only, on ("a","b")
  assert tok.merges == [(ord('a'), ord('b'))]

  # vocab[256] == b"ab"
  assert tok.vocab[256] == b'ab'
  # special token appended at 257
  assert tok.vocab[257] == b'<|tok|>'


# ------ resets state ----------------------------------------------------
def test_train_resets_previous_state(tmp_path):
  """
  Ensure that train() always clears self.merges and self.vocab back to
  the clean 0–255 byte vocab before doing any new merges.
  """
  tok = Tokenizer()

  # 1) Corrupt the state on purpose
  tok.merges = [(1, 2), (3, 4)]
  tok.vocab = {999: b'XXX'}  # bogus entry

  # 2) Run train on a simple example that will do exactly one merge:
  #    text="abab" => one merge ("a","b") => idx=256
  fp = tmp_path / 'tmp.txt'
  fp.write_text('abab', encoding='utf-8')
  tok.train_from_file(fp, vocab_size=256 + 2, special_tokens=['x'])

  # 3) After train, old merges must be gone
  assert tok.merges == [(ord('a'), ord('b'))]

  # 4) And the bogus vocab entry must have been cleared
  assert 999 not in tok.vocab

  # 5) The vocab must at least contain the 0–255 byte entries
  for i in range(256):
    assert tok.vocab[i] == bytes([i])

  # 6) And the new merge token at index 256
  assert tok.vocab[256] == b'ab'


# ------- train and train_from_file produce same output -------------------

TEXT = r"""u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>


Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
"Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
"No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
"Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
"Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
"We're okay, Mommy. But our toys are broken," Lily said.
"I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
"Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
<|endoftext|>

"""


def test_train_and_train_from_file_same_out(tmp_path):
  """
  We give the piece of TinyStories and want to be sure
  baseline slow `train` and optimized fast `train_from_file`
  produce the same output
  """
  fp = tmp_path / 'tmp.txt'
  fp.write_text(TEXT, encoding='utf-8')
  base_tok = Tokenizer()
  fast_tok = Tokenizer()
  special_tokens = ['<|endoftext|>']
  vocab_size = 256 + 31 + 1  # first bytes + 31 merge + 1 special

  base_tok.train(TEXT, vocab_size, special_tokens)
  fast_tok.train_from_file(fp, vocab_size, special_tokens)

  assert base_tok.merges == fast_tok.merges
  assert base_tok.vocab == fast_tok.vocab
