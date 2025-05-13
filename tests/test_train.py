import pytest

# Adjust these to wherever your code lives
import toksmith.tokenizer as m
from toksmith.tokenizer import Tokenizer

def test_train_raises_on_small_vocab():
    tok = Tokenizer()
    # vocab_size must be at least 256 + len(special_tokens)=0
    with pytest.raises(ValueError) as ei:
        tok.train("whatever", 255, [])
    assert "vocab_size must be >" in str(ei.value)

def test_train_single_merge(monkeypatch):
    """Stub out pretoken & pair counts so exactly one merge occurs."""
    # 1) Prepare a fresh Tokenizer with clean state
    tok = Tokenizer()
    tok.merges = []
    # ensure initial vocab maps 0..255 to bytes([i])
    tok.vocab = {i: bytes([i]) for i in range(256)}

    # 2) Stub _pretoken_count to always return a single pretoken (0,1) count=1
    monkeypatch.setattr(
        Tokenizer,
        "_pretoken_count",
        lambda self, text: {(0, 1): 1}
    )

    # 3) Stub _pairs_count so that only the first call returns {(0,1):1}, then empty
    call = {"n": 0}
    def fake_pairs(self, pret):
        call["n"] += 1
        return {(0, 1): 1} if call["n"] == 1 else {}
    monkeypatch.setattr(Tokenizer, "_pairs_count", fake_pairs)

    # 4) Stub the module‐level _merge to simply collapse (0,1) → (new_ix,)
    monkeypatch.setattr(
        m,
        "_merge",
        lambda pt, pair, new_ix: (new_ix,)
    )

    # 5) Invoke train for exactly one iteration (vocab_size=257)
    tok.train("dummy", vocab_size=257, special_tokens=[])

    # 6) Assertions
    #   - Exactly one merge recorded, and it's the pair we stubbed
    assert tok.merges == [(0, 1)]
    #   - A new vocab entry at index 256 = vocab[0] + vocab[1]
    assert 256 in tok.vocab
    assert tok.vocab[256] == bytes([0]) + bytes([1])
    #   - No further merges
    assert call["n"] == 1

def test_train_on_wiki_example():
    """
    Full-pipeline integration test on “aaabdaaabac” 
    reflecting our tie-breaking rule:
      1) First merge:  "aa" -> Z  token at index 256
      2) Second merge: "Za" -> Y  token at index 257
      3) Third merge:  "Yb" -> X  token at index 258
      Then stop, since we asked for only 3 merges (vocab_size=259).
      see: https://en.wikipedia.org/wiki/Byte_pair_encoding#Example
    """
    text = "aaabdaaabac"
    tok = Tokenizer()

    # Request exactly 3 merges beyond the 256 base bytes
    tok.train(text, vocab_size=256 + 3, special_tokens=[])

    # Expect exactly those three merges—in byte‐pair form:
    first   = (ord("a"), ord("a"))  # "aa"
    second  = (256, ord("a"))       # "Za"
    third   = (257, ord("b"))       # "Yb"
    assert tok.merges == [ first, second, third ]

    # And the new vocab entries should be:
    # index=256 => "aa", index=257 => "Za", index=258 => "Yb"
    assert tok.vocab[256] == b"aa"
    assert tok.vocab[257] == b"aaa"
    assert tok.vocab[258] == b"aaab"

def test_train_strips_and_merges_with_special_token():
    """
    Given text="ab<tok>ab" and special_tokens=["<tok>"], with vocab_size=258:
     - We reserve 256 base bytes, 1 merge, 1 special token.
     - After stripping <tok>, text → "abab", so the single most-frequent pair is ("a","b").
     - We expect one merge: ("a","b") → idx=256, and then special token at idx=257.
    """
    text = "ab<tok>ab"
    special = ["<tok>"]
    vocab_size = 256 + 1 + len(special)  # one merge + one special token

    tok = Tokenizer()
    # run training
    tok.train(text, vocab_size=vocab_size, special_tokens=special)

    # 1 merge only, on ("a","b")
    assert tok.merges == [(ord("a"), ord("b"))]

    # vocab[256] == b"ab"
    assert tok.vocab[256] == b"ab"
    # special token appended at 257
    assert tok.vocab[257] == b"<tok>"

def test_merges_equal_for_clean_and_special_text():
    """
    The merge sequence on "abab" (no special tokens) should match
    the merge sequence on "ab<tok>ab" when stripping <tok>,
    as long as we reserve the same vocab slots.
    """
    clean = "abab"
    noisy = "ab<tok>ab"
    special = ["<tok>"]
    # both will do exactly one merge

    t1 = Tokenizer()
    t1.train(clean,   vocab_size=256 + 1, special_tokens=[])
    t2 = Tokenizer()
    t2.train(noisy,   vocab_size=256 + 1 + len(special), special_tokens=special)

    # The _only_ difference should be that t2 has the special token
    assert t1.merges == t2.merges
    assert t1.vocab[256] == t2.vocab[256]