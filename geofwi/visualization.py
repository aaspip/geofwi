"""Simple reproducible visualizations for 2D velocity model datasets."""
import math
import numpy as np
from .dataset import load


def visualize(path=None, *, indices=None, n=9, seed=42, cmap='turbo',
              vmin=None, vmax=None, save=None, show=True):
    """Plot a small subset of models with a shared velocity color scale."""
    import matplotlib.pyplot as plt
    arr = load(path)
    if arr.ndim != 3:
        raise ValueError(f'Expected (N, nz, nx), got {arr.shape}')
    if indices is None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(arr), size=min(n, len(arr)), replace=False).tolist()
    indices = [int(i) for i in indices]
    if not indices:
        raise ValueError('No indices selected')
    models = np.stack([arr[i] for i in indices])
    if vmin is None:
        vmin = float(np.nanmin(models))
    if vmax is None:
        vmax = float(np.nanmax(models))
    cols = min(3, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False, constrained_layout=True)
    im = None
    for ax, index, model in zip(axes.flat, indices, models):
        im = ax.imshow(model, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='upper')
        ax.set(title=f'Model #{index}', xlabel='Horizontal sample', ylabel='Depth sample')
    for ax in axes.flat[len(indices):]:
        ax.axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Velocity (m/s)')
    if save:
        fig.savefig(save, dpi=180, bbox_inches='tight')
    if show:
        plt.show()
    return fig
