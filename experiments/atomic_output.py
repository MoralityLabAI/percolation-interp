"""Publish large Torch ZIP artifacts only after a complete validated write."""
from pathlib import Path
import os
import shutil
import tempfile
import zipfile


def validate_torch_zip(path):
    """Stream CRC checks without loading GB-sized tensors into host memory."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len([n for n in names if n.endswith('/data.pkl')]) != 1:
            raise ValueError('expected one Torch metadata record')
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f'incomplete/corrupt Torch ZIP member: {bad}')


def atomic_zip_save(writer, path, min_free_bytes=2 * 1024 ** 3):
    """Keep the old published file intact if serialization or validation fails.

    A failed temporary is preserved for diagnosis, but never gets the final
    filename on which existing pipeline stage-skip checks depend.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(path.parent).free < min_free_bytes:
        raise OSError('insufficient free disk before attribution serialization')
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    print(f'Writing temporary attribution archive: {temporary}', flush=True)
    writer(temporary)
    validate_torch_zip(temporary)
    with temporary.open('r+b') as f:
        os.fsync(f.fileno())
    temporary.replace(path)
    return path
