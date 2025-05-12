import pytest
from toksmith.tokenizer import Tokenizer

@pytest.fixture
def tok():
    # This will compile GPT2_SPLIT_PAT internally
    return Tokenizer()

# test _pretoken_count function
@pytest.mark.parametrize("text, expected", [
    # 1) A single two‐byte letter group "hé"
    (
        "hé",
        { tuple("hé".encode("utf-8")): 1 }
    ),
    # 2) ASCII words + space: "hello" and " world"
    (
        "hello world",
        {
            tuple("hello".encode("utf-8")): 1,
            tuple(" world".encode("utf-8")): 1,
        }
    ),
    # 3) Contraction: splits into ["it", "'s"]
    (
        "it's",
        {
            tuple("it".encode("utf-8")): 1,
            tuple("'s".encode("utf-8")): 1,
        }
    ),
    # 4) Mixed letters + digits
    (
        "test123",
        {
            tuple("test".encode("utf-8")): 1,
            tuple("123".encode("utf-8")): 1,
        }
    ),
    # 5) Emoji + surrounding spaces
    (
        " A🌟B ",
        {
            tuple(" A".encode("utf-8")): 1,
            tuple("🌟".encode("utf-8")): 1,
            tuple("B".encode("utf-8")): 1,
            tuple(" ".encode("utf-8")): 1,
        }
    ),
])
def test_pretoken_count_unicode(tok, text, expected):
    """
    Ensure that _pretoken_count:
      - Groups letters (including accents) as single pretokens
      - Keeps leading spaces when the pattern dictates
      - Splits numbers, punctuation, emojis correctly
    """
    result = tok._pretoken_count(text)
    assert result == expected

# test _pairs_count
def test_pairs_count_unit_manual(tok):
    """
    Unit-test _pairs_count in isolation:
      - pretend we have two pretokens: (1,2,3) occurring twice, and (3,4) once
      - so pairs are:
           (1,2): 2
           (2,3): 2
           (3,4): 1
    """
    manual_pretoks = {
        (1, 2, 3): 2,
        (3, 4):    1,
    }
    expected = {
        (1, 2): 2,
        (2, 3): 2,
        (3, 4): 1,
    }
    result = tok._pairs_count(manual_pretoks)
    assert result == expected

