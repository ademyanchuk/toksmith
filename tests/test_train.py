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

def test_train_raises_on_special_tokens():
    tok = Tokenizer()
    # special_tokens not yet supported
    with pytest.raises(NotImplementedError):
        tok.train("any text", 300, ["<|endoftext|>"])

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
