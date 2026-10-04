import numpy as np
import pytest

from completion_core.prep import save_outputs
from completion_core.training import SequenceDataset
from completion_core.vocabulary import Vocabulary
from phonebook.prepare_phonebook import load_phonebook_lines


@pytest.mark.parametrize("empty_part", ["train", "val"])
def test_empty_preparation_does_not_modify_existing_artifacts(tmp_path, empty_part):
    names = ["train.bin", "val.bin", "meta.pkl"]
    for name in names:
        (tmp_path / name).write_bytes(b"previous valid artifact")
    before = {name: (tmp_path / name).read_bytes() for name in names}
    arrays = {name: np.array([1, 2], dtype=np.uint16) for name in ["train", "val"]}
    arrays[empty_part] = np.array([], dtype=np.uint16)
    with pytest.raises(ValueError, match="both be non-empty"):
        save_outputs(tmp_path, arrays["train"], arrays["val"], {}, {})
    assert {name: (tmp_path / name).read_bytes() for name in names} == before


def test_default_all_train_split_fails_before_writing(run_script, tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("a=1\nb=2\n", encoding="utf-8")
    output = tmp_path / "data"
    result = run_script("phonebook/prepare_phonebook.py", [str(source), "--out-dir", str(output)])
    assert result.returncode != 0
    assert "both be non-empty" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("empty_part", ["train", "eval"])
def test_explicit_empty_input_is_rejected(run_script, tmp_path, empty_part):
    paths = {name: tmp_path / f"{name}.txt" for name in ["train", "eval"]}
    for name, path in paths.items():
        path.write_text("" if name == empty_part else "a=1\n", encoding="utf-8")
    output = tmp_path / "data"
    result = run_script("phonebook/prepare_phonebook.py", [
        "--train-input", str(paths["train"]), "--eval-input", str(paths["eval"]), "--out-dir", str(output),
    ])
    assert result.returncode != 0
    assert "both be non-empty" in result.stderr
    assert not output.exists()


def test_phonebook_normalizes_final_newline(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("a=1\nb=2", encoding="utf-8")
    assert load_phonebook_lines(source) == ["a=1\n", "b=2\n"]


@pytest.mark.parametrize("text", ["\n", "a\n", "=1\n", "a=\n", "a=1=2\n"])
def test_phonebook_rejects_invalid_records(tmp_path, text):
    source = tmp_path / "source.txt"
    source.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid completion record"):
        load_phonebook_lines(source)


def test_dataset_rejects_unterminated_record_instead_of_dropping_it(tmp_path):
    chars = "a=1\nb2_"
    stoi = {c: i for i, c in enumerate(chars)}
    vocab = Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})
    binary = tmp_path / "data.bin"
    np.array([stoi[c] for c in "a=1\nb=2"], dtype=np.uint16).tofile(binary)
    with pytest.raises(ValueError, match="unterminated final record"):
        SequenceDataset(binary, vocab)


def test_dataset_reports_empty_data_clearly(tmp_path):
    stoi = {c: i for i, c in enumerate("_=\n")}
    vocab = Vocabulary(len(stoi), stoi, {i: c for c, i in stoi.items()})
    binary = tmp_path / "empty.bin"
    binary.write_bytes(b"")
    with pytest.raises(ValueError, match="Dataset is empty"):
        SequenceDataset(binary, vocab)
