"""Read selected float32 arrays from an uncompressed torch.save state archive.

No PyTorch import, model construction, CUDA context, or arbitrary pickle
globals. Unsupported formats are rejected. Views are read-only file mappings.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
from pathlib import Path
import pickle
import re
import struct
import zipfile

import numpy as np


@dataclass(frozen=True)
class _Storage:
    key: str
    count: int


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    shape: tuple
    stride: tuple


def _rebuild(storage, offset, shape, stride, *unused):
    if not isinstance(storage, _Storage):
        raise ValueError("unsupported tensor storage")
    return _Tensor(storage, offset, tuple(shape), tuple(stride))


class _Metadata(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch", "FloatStorage"):
            return "float32"
        if (module, name) in (("torch._utils", "_rebuild_tensor_v2"),
                              ("torch._utils", "_rebuild_tensor")):
            return _rebuild
        raise pickle.UnpicklingError(f"unsupported metadata global {module}.{name}")

    def persistent_load(self, saved):
        if (not isinstance(saved, tuple) or len(saved) != 5 or saved[0] != "storage"
                or saved[1] != "float32" or not re.fullmatch(r"[0-9]+", str(saved[2]))
                or not isinstance(saved[4], int) or saved[4] < 0):
            raise pickle.UnpicklingError("unsupported storage descriptor")
        return _Storage(str(saved[2]), saved[4])


class CheckpointArrays:
    def __init__(self, path):
        self.path = Path(path)
        with zipfile.ZipFile(self.path) as archive:
            metadata = [n for n in archive.namelist() if n.endswith("/data.pkl")]
            if len(metadata) != 1:
                raise ValueError("expected one torch.save metadata record")
            self.prefix = metadata[0][:-len("data.pkl")]
            self.state = _Metadata(io.BytesIO(archive.read(metadata[0]))).load()
            if not isinstance(self.state, (dict, OrderedDict)):
                raise ValueError("expected state dictionary")
            self.infos = {i.filename: i for i in archive.infolist()}
            order_path = self.prefix + "byteorder"
            order = archive.read(order_path).decode("ascii") if order_path in self.infos else "little"
            if order not in ("little", "big"):
                raise ValueError("unsupported byte order")
            self.dtype = np.dtype("<f4" if order == "little" else ">f4")

    def array(self, key):
        ref = self.state[key]
        if not isinstance(ref, _Tensor):
            raise ValueError("requested value is not a supported tensor")
        if (len(ref.shape) != len(ref.stride) or not isinstance(ref.offset, int) or ref.offset < 0
                or any(not isinstance(x, int) or x < 0 for x in (*ref.shape, *ref.stride))):
            raise ValueError("invalid tensor view")
        last = ref.offset + sum((n - 1) * s for n, s in zip(ref.shape, ref.stride))
        if any(n == 0 for n in ref.shape) or last >= ref.storage.count:
            raise ValueError("empty/out-of-bounds tensor view unsupported")
        info = self.infos[self.prefix + "data/" + ref.storage.key]
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError("compressed/encrypted storage cannot be memory-mapped")
        if info.file_size != ref.storage.count * self.dtype.itemsize:
            raise ValueError("storage byte count mismatch")
        with self.path.open("rb") as f:
            f.seek(info.header_offset)
            header = f.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise ValueError("invalid ZIP local header")
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        payload = info.header_offset + 30 + name_length + extra_length
        storage = np.memmap(self.path, mode="r", dtype=self.dtype,
                            offset=payload, shape=(ref.storage.count,))
        result = np.ndarray(ref.shape, dtype=self.dtype, buffer=storage,
                            offset=ref.offset * self.dtype.itemsize,
                            strides=tuple(s * self.dtype.itemsize for s in ref.stride))
        result.flags.writeable = False
        return result
