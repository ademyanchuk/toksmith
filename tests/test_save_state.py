import json
import os
import tempfile
import pytest
from pathlib import Path

from toksmith.tokenizer import Tokenizer

@pytest.fixture
def dummy_tok(tmp_path):
    # Make a fresh Tokenizer and override its state
    tok = Tokenizer()
    tok.merges = [(1, 2), (3, 4)]
    # sample vocab: two entries plus the 0–255 seeded in __init__
    tok.vocab = {i: bytes([i]) for i in range(256)}
    tok.vocab.update({
        256: b"ab",
        257: b"cd",
    })
    return tok

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_save_creates_folder_and_file(tmp_path, dummy_tok):
    folder = tmp_path / "output_dir"
    # folder does not exist yet
    assert not folder.exists()

    out = dummy_tok.save_state(prefix="mytok", folder=str(folder))
    # Returns the correct Path
    assert isinstance(out, Path)
    assert out.name == "mytok_tokenizer.json"
    assert out.parent == folder.resolve()

    # Folder is created
    assert folder.is_dir()
    # File is created
    assert out.exists()

    data = load_json(out)
    # version field
    assert data["version"] == 1
    # merges round-tripped
    assert data["merges"] == [[1, 2], [3, 4]]
    # vocab keys are strings, values are hex
    # check a couple entries
    assert data["vocab"]["256"] == "6162"  # b"ab".hex()
    assert data["vocab"]["257"] == "6364"  # b"cd".hex()
    # ensure existing base bytes also present
    assert data["vocab"]["0"] == "00"

def test_save_raises_if_folder_is_file(tmp_path, dummy_tok):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("oops")  # create a file
    with pytest.raises(NotADirectoryError):
        dummy_tok.save_state(prefix="p", folder=str(file_path))

def test_prefix_sanitization(tmp_path, dummy_tok):
    # If prefix contains path components, only final name is used
    messy_prefix = "../etc/passwd"
    out = dummy_tok.save_state(prefix=messy_prefix, folder=tmp_path)
    # Should ignore "../etc" and just use "passwd"
    assert out.name == "passwd_tokenizer.json"

def test_folder_as_path_object(tmp_path, dummy_tok):
    # Using a Path directly for folder
    folder_path = tmp_path / "dir2"
    out = dummy_tok.save_state(prefix="foo", folder=folder_path)
    assert out.parent == folder_path.resolve()
    assert (folder_path / "foo_tokenizer.json").exists()
