import logging
from pathlib import Path

import pytest

# Adjust these imports to where your script lives:
from toksmith.train import configure_logging, validate_paths


@pytest.mark.parametrize(
  'verbosity, quiet, expected_level',
  [
    (0, False, logging.WARNING),
    (1, False, logging.INFO),
    (2, False, logging.DEBUG),
    (5, False, logging.DEBUG),  # anything ≥2 → DEBUG
    (0, True, logging.ERROR),
    (1, True, logging.ERROR),
    (2, True, logging.ERROR),
  ],
)
def test_configure_logging_sets_root_level(verbosity, quiet, expected_level):
  configure_logging(verbosity=verbosity, quiet=quiet)
  root = logging.getLogger()
  assert root.level == expected_level


def test_validate_paths_success(tmp_path):
  # Create a valid UTF-8 file
  infile = tmp_path / 'ok.txt'
  infile.write_text('hello world', encoding='utf-8')

  # And ensure output directory does not yet exist
  outdir = tmp_path / 'outdir'
  assert not outdir.exists()

  # Should not raise or exit
  validate_paths(infile, outdir)

  # Now outdir should have been created
  assert outdir.is_dir()


def test_validate_paths_missing_input(tmp_path, caplog):
  # Missing input file → SystemExit(1)
  missing = tmp_path / 'nope.txt'
  assert not validate_paths(missing, tmp_path / 'out')
  assert 'is not a file' in caplog.text


def test_validate_paths_invalid_utf8(tmp_path, caplog):
  # Create a file with invalid UTF-8 bytes
  bad = tmp_path / 'bad.txt'
  bad.write_bytes(b'\xff\xfe\xfd')

  outdir = tmp_path / 'outdir'
  assert not validate_paths(bad, outdir)
  assert 'Failed to read' in caplog.text


def test_validate_paths_output_exists_as_file(tmp_path, caplog):
  # Input file is good
  infile = tmp_path / 'ok.txt'
  infile.write_text('ok', encoding='utf-8')

  # Create a file at the output-path location
  outpath = tmp_path / 'not_a_dir'
  outpath.write_text('oops')

  assert not validate_paths(infile, outpath)
  assert 'exists and is not a directory' in caplog.text


def test_validate_paths_cant_create_output(tmp_path, monkeypatch, caplog):
  # Input file is good
  infile = tmp_path / 'ok.txt'
  infile.write_text('ok', encoding='utf-8')

  # Simulate a mkdir failure
  outdir = tmp_path / 'will_fail'

  def fake_mkdir(*args, **kwargs):
    raise PermissionError('no perms')

  monkeypatch.setattr(Path, 'mkdir', fake_mkdir)

  assert not validate_paths(infile, outdir)
  assert 'Could not create output directory' in caplog.text
