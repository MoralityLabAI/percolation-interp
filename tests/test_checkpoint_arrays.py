import io
import pickle

import numpy as np
import pytest
import torch

from percinterp.checkpoint_arrays import CheckpointArrays, _Metadata


def test_matches_torch_save_shared_strided_views(tmp_path):
    a = torch.arange(120, dtype=torch.float32).reshape(10, 12)
    state = {"full": a, "transpose": a.T, "offset": a[2:8:2, 1:11:3],
             "separate": torch.randn(3, 7)}
    path = tmp_path / "state.pth"
    torch.save(state, path)
    reader = CheckpointArrays(path)
    for name, value in state.items():
        got = reader.array(name)
        assert np.array_equal(got, value.numpy())
        assert not got.flags.writeable


def test_rejects_arbitrary_pickle_globals_without_executing(tmp_path):
    class Unsupported:
        def __reduce__(self):
            return (eval, ("1+1",))
    with pytest.raises(pickle.UnpicklingError):
        _Metadata(io.BytesIO(pickle.dumps(Unsupported()))).load()
    path = tmp_path / "double.pth"
    torch.save({"x": torch.ones(2, dtype=torch.float64)}, path)
    with pytest.raises(pickle.UnpicklingError):
        CheckpointArrays(path)
