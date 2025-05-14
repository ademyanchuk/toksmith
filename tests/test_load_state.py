import json
import pytest
from pathlib import Path

from toksmith.tokenizer import Tokenizer

def _write_state(tmp_path: Path, prefix: str, data: dict) -> Path:
    """
    Helper to write `<tmp_path>/<prefix>_tokenizer.json` with the given dict.
    """
    fn = tmp_path / f"{prefix}_tokenizer.json"
    fn.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return fn

def test_load_state_happy_path(tmp_path):
    """
    Given a well-formed state file, load_state should populate merges & vocab correctly.
    """
    prefix = "mytok"
    # prepare JSON payload
    merges = [[1, 2], [3, 4]]
    vocab = {
        "0":   "00",    # b"\x00"
        "255": "ff",    # b"\xff"
        "256": "6162",  # b"ab"
    }
    payload = {"version": 1, "merges": merges, "vocab": vocab}
    _write_state(tmp_path, prefix, payload)

    tok = Tokenizer()
    # ensure we start with non-empty state
    tok.merges = [(9, 9)]
    tok.vocab = {999: b"X"}

    tok.load_state(prefix, tmp_path)

    # merges → list of tuples
    assert tok.merges == [(1, 2), (3, 4)]
    # vocab → dict of ints to bytes
    assert tok.vocab[0]   == b"\x00"
    assert tok.vocab[255] == b"\xff"
    assert tok.vocab[256] == b"ab"
    # no stray keys
    assert set(tok.vocab.keys()) == {0, 255, 256}

def test_load_raises_if_folder_not_dir(tmp_path):
    """
    Passing a file path (not a dir) should raise NotADirectoryError.
    """
    bad = tmp_path / "not_a_dir"
    bad.write_text("hey")  # create file
    tok = Tokenizer()
    with pytest.raises(NotADirectoryError):
        tok.load_state("prefix", bad)

def test_load_raises_if_file_missing(tmp_path):
    """
    If the expected JSON file isn't there, FileNotFoundError is raised.
    """
    tok = Tokenizer()
    with pytest.raises(FileNotFoundError):
        tok.load_state("does_not_exist", tmp_path)

def test_load_raises_on_unsupported_version(tmp_path):
    """
    If the 'version' field isn't 1, we reject it.
    """
    _write_state(tmp_path, "p", {"version": 2, "merges": [], "vocab": {}})
    tok = Tokenizer()
    with pytest.raises(ValueError) as ei:
        tok.load_state("p", tmp_path)
    assert "Unsupported tokenizer state version" in str(ei.value)

def test_load_raises_on_bad_token_id(tmp_path):
    """
    Non-integer token-id keys should trigger ValueError.
    """
    payload = {"version": 1, "merges": [], "vocab": {"not_int": "00"}}
    _write_state(tmp_path, "p", payload)
    tok = Tokenizer()
    with pytest.raises(ValueError) as ei:
        tok.load_state("p", tmp_path)
    assert "Invalid token ID" in str(ei.value)

def test_load_raises_on_bad_hex(tmp_path):
    """
    Invalid hex strings in the vocab should trigger ValueError.
    """
    payload = {"version": 1, "merges": [], "vocab": {"0": "zz"}}
    _write_state(tmp_path, "p", payload)
    tok = Tokenizer()
    with pytest.raises(ValueError) as ei:
        tok.load_state("p", tmp_path)
    assert "Invalid hex for token" in str(ei.value)
