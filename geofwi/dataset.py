"""Memory-mapped inspection of GeoFWI NumPy model collections."""
from pathlib import Path
import numpy as np
from .download import data_dir


def find_arrays(path=None) -> list[Path]:
    root = Path(path).expanduser() if path else data_dir()
    if root.is_file():
        return [root] if root.suffix == '.npy' else []
    return sorted(root.rglob('*.npy')) if root.exists() else []


def _select(path=None) -> Path:
    root = Path(path).expanduser() if path else data_dir()
    if root.is_file():
        return root
    arrays = find_arrays(root)
    matches = [p for p in arrays if p.name.lower() == 'geofwi.npy']
    if len(matches) == 1:
        return matches[0]
    if len(arrays) == 1:
        return arrays[0]
    if not arrays:
        raise FileNotFoundError(f'No .npy files found in {root}. The Zenodo archive may not contain velocity models.')
    raise ValueError('Multiple NumPy arrays found; pass an explicit .npy path: ' + ', '.join(str(p) for p in arrays[:15]))


def load(path=None, *, mmap=True):
    """Load a NumPy array without copying it into RAM by default."""
    return np.load(_select(path), mmap_mode='r' if mmap else None, allow_pickle=False)


def info(path=None) -> dict:
    p = _select(path)
    arr = load(p)
    result = {'path': str(p), 'shape': tuple(int(x) for x in arr.shape),
              'dtype': str(arr.dtype), 'size_bytes': p.stat().st_size,
              'memory_mapped': isinstance(arr, np.memmap)}
    print('GeoFWI array information')
    for key, value in result.items():
        print(f'  {key}: {value}')
    return result


def get_models(indices, path=None):
    arr = load(path)
    if arr.ndim != 3:
        raise ValueError(f'Expected (N, nz, nx), received {arr.shape}')
    return np.stack([arr[int(i)] for i in indices])
