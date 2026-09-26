import ast
import json
import zipfile
from pathlib import Path

import pytest

from scripts.prepare_colab_data import REQUIRED, prepare


def make_zip(path, overrides=None):
    overrides = overrides or {}
    with zipfile.ZipFile(path, "w") as archive:
        for name in REQUIRED:
            archive.writestr(f"student_resource/dataset/{name}", overrides.get(name, "id\tvalue\n"))


def test_extract_and_restart_replaces_corrupt_local_copy(tmp_path):
    archive = tmp_path / "official.zip"
    make_zip(archive)
    out = tmp_path / "local"
    result = prepare(archive, out)
    assert set(result["members"]) == set(REQUIRED)
    target = out / REQUIRED[0]
    target.write_text("bad data\n")  # Same length, different CRC.
    prepare(archive, out)
    assert target.read_text() == "id\tvalue\n"


def test_missing_official_member_fails(tmp_path):
    archive = tmp_path / "incomplete.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("student_resource/dataset/train/train_source1.tsv", "x")
    with pytest.raises(FileNotFoundError, match="missing required"):
        prepare(archive, tmp_path / "local")


def test_duplicate_dataset_member_fails(tmp_path):
    archive = tmp_path / "ambiguous.zip"
    make_zip(archive)
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr("other/dataset/train/train_source1.tsv", "x")
    with pytest.raises(ValueError, match="Duplicate dataset member"):
        prepare(archive, tmp_path / "local")


def test_colab_notebook_cells_parse():
    path = Path(__file__).resolve().parents[1] / "notebooks/Plan2_Training.ipynb"
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
