"""
QKernel-Lite – Data Loading & Stratified Subsampling
=====================================================
Stage 1: PneumoniaMNIST loader with deterministic,
stratified 50/50 subset sampler.
"""

import numpy as np
import torch
from torch.utils.data import Subset
from pathlib import Path
from typing import Tuple, Optional
import matplotlib.pyplot as plt

import config


def load_pneumoniamnist(
    size: int = config.IMAGE_SIZE,
    download: bool = True,
) -> Tuple:
    """
    Load PneumoniaMNIST train/val/test splits via medmnist.

    Returns
    -------
    train_dataset, val_dataset, test_dataset
        Each is a medmnist dataset object with .imgs and .labels attributes.
    info : dict
        Dataset metadata (class names, n_channels, etc.).
    """
    import medmnist
    from medmnist import INFO

    info = INFO[config.DATASET_NAME]
    DataClass = getattr(medmnist, info["python_class"])

    # Try size=224 first; fall back to default if not supported
    kwargs = dict(split="train", download=download, root=str(config.CACHE_DIR))
    try:
        train_ds = DataClass(size=size, **kwargs)
    except TypeError:
        # older medmnist versions don't support `size`
        train_ds = DataClass(**kwargs)
        size = 28  # actual size

    val_ds = DataClass(
        split="val", download=download, root=str(config.CACHE_DIR),
        **({"size": size} if size != 28 else {})
    )
    test_ds = DataClass(
        split="test", download=download, root=str(config.CACHE_DIR),
        **({"size": size} if size != 28 else {})
    )

    return train_ds, val_ds, test_ds, info


def get_class_counts(dataset) -> dict:
    """Return {class_label: count} for a medmnist dataset."""
    labels = dataset.labels.flatten()
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def stratified_subset_indices(
    dataset,
    n: int,
    seed: int,
) -> np.ndarray:
    """
    Draw a stratified 50/50 balanced subset of size n from dataset.

    Parameters
    ----------
    dataset : medmnist dataset with .labels
    n : int
        Total subset size (must be even).
    seed : int
        Random seed for deterministic sampling.

    Returns
    -------
    indices : np.ndarray of shape (n,)
        Indices into the dataset, sorted.
    """
    assert n % 2 == 0, f"N must be even for 50/50 balance, got {n}"
    half = n // 2

    labels = dataset.labels.flatten()
    rng = np.random.RandomState(seed)

    class_0_idx = np.where(labels == 0)[0]
    class_1_idx = np.where(labels == 1)[0]

    assert len(class_0_idx) >= half, \
        f"Not enough class-0 samples: need {half}, have {len(class_0_idx)}"
    assert len(class_1_idx) >= half, \
        f"Not enough class-1 samples: need {half}, have {len(class_1_idx)}"

    chosen_0 = rng.choice(class_0_idx, size=half, replace=False)
    chosen_1 = rng.choice(class_1_idx, size=half, replace=False)

    indices = np.sort(np.concatenate([chosen_0, chosen_1]))
    return indices


def get_subset_data(
    dataset,
    indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract images and labels for given indices.

    Returns
    -------
    images : np.ndarray of shape (n, H, W) or (n, H, W, C)
    labels : np.ndarray of shape (n,)
    """
    images = dataset.imgs[indices]
    labels = dataset.labels[indices].flatten()
    return images, labels


def save_sample_grid(
    dataset,
    indices: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = "Sample Grid",
) -> Path:
    """Save an 8-image grid for visual inspection."""
    if save_path is None:
        save_path = config.RESULTS_DIR / "sample_grid.png"

    images, labels = get_subset_data(dataset, indices[:8])
    class_names = ["Normal", "Pneumonia"]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            img = images[i]
            if img.ndim == 3:
                img = img[:, :, 0]  # take first channel if multi-channel
            ax.imshow(img, cmap="gray")
            ax.set_title(f"{class_names[labels[i]]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


def print_split_info(train_ds, val_ds, test_ds):
    """Print sizes and class counts for each split."""
    for name, ds in [("Train", train_ds), ("Val", val_ds), ("Test", test_ds)]:
        counts = get_class_counts(ds)
        total = len(ds.labels)
        print(f"{name}: {total} images | "
              f"Normal={counts.get(0, 0)}, Pneumonia={counts.get(1, 0)} | "
              f"Pneumonia fraction={counts.get(1, 0)/total:.2%}")


if __name__ == "__main__":
    # Quick sanity check
    train_ds, val_ds, test_ds, info = load_pneumoniamnist()
    print_split_info(train_ds, val_ds, test_ds)

    # Test determinism
    idx1 = stratified_subset_indices(train_ds, n=20, seed=42)
    idx2 = stratified_subset_indices(train_ds, n=20, seed=42)
    assert np.array_equal(idx1, idx2), "Subset sampler is not deterministic!"
    print(f"\nSubset (N=20, seed=42): indices = {idx1}")

    _, labels = get_subset_data(train_ds, idx1)
    print(f"Subset class counts: Normal={np.sum(labels==0)}, Pneumonia={np.sum(labels==1)}")

    path = save_sample_grid(train_ds, idx1, title="N=20, seed=42")
    print(f"Sample grid saved to: {path}")
