import importlib.util
from pathlib import Path
import zipfile
import pytest

SPEC = importlib.util.spec_from_file_location('atomic_output',
    Path(__file__).resolve().parents[1] / 'experiments/atomic_output.py')
atomic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(atomic)


def write_archive(path, data=b'new tensor data'):
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', b'metadata')
        archive.writestr('archive/data/0', data)


def test_interrupted_save_preserves_previous_published_artifact(tmp_path):
    path = tmp_path / 'artifact.pt'
    write_archive(path, b'old tensors')
    original = path.read_bytes()
    def fail(temporary):
        temporary.write_bytes(b'partial write')
        raise OSError('simulated disk exhaustion')
    with pytest.raises(OSError):
        atomic.atomic_zip_save(fail, path, min_free_bytes=0)
    assert path.read_bytes() == original
    assert len(list(tmp_path.glob('*.partial'))) == 1


def test_invalid_archive_never_publishes_and_complete_archive_does(tmp_path):
    path = tmp_path / 'artifact.pt'
    with pytest.raises(zipfile.BadZipFile):
        atomic.atomic_zip_save(lambda temp: temp.write_bytes(b'broken zip'), path, min_free_bytes=0)
    assert not path.exists()
    atomic.atomic_zip_save(write_archive, path, min_free_bytes=0)
    atomic.validate_torch_zip(path)
    with zipfile.ZipFile(path) as archive:
        assert archive.read('archive/data/0') == b'new tensor data'


def test_crc_failure_is_rejected_and_disk_preflight_happens_before_writer(tmp_path):
    path = tmp_path / 'artifact.pt'
    write_archive(path, b'UNIQUE_TENSOR_BYTES')
    content = path.read_bytes().replace(b'UNIQUE_TENSOR_BYTES', b'CORRUPTED_DATA_HERE')
    path.write_bytes(content)
    with pytest.raises(ValueError, match='corrupt Torch ZIP member'):
        atomic.validate_torch_zip(path)
    def forbidden(_):
        raise AssertionError('writer called despite disk preflight')
    with pytest.raises(OSError, match='insufficient free disk'):
        atomic.atomic_zip_save(forbidden, path, min_free_bytes=10 ** 30)


def test_actual_torch_archive_round_trip(tmp_path):
    torch = pytest.importorskip('torch')
    expected = {'regular_attr': {'module': torch.arange(12).reshape(3, 4)}, 'n_tokens_processed': 102400}
    path = tmp_path / 'attributions.pt'
    atomic.atomic_zip_save(lambda p: torch.save(expected, p), path, min_free_bytes=0)
    actual = torch.load(path, weights_only=True)
    assert actual['n_tokens_processed'] == 102400
    assert torch.equal(actual['regular_attr']['module'], expected['regular_attr']['module'])
