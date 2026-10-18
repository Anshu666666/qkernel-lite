"""
QKernel-Lite – ResNet-18 Feature Extraction
============================================
Stage 2: Frozen ResNet-18 (ImageNet weights) → 512-d GAP vectors.
Features are cached to disk per split.
"""

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from pathlib import Path
from tqdm import tqdm

import config


def get_resnet18_extractor(device: str = "cpu") -> torch.nn.Module:
    """
    Load ResNet-18 with ImageNet weights, remove the final FC layer,
    set to eval mode, freeze all parameters.

    Returns the model (on device) that maps (B, 3, 224, 224) → (B, 512).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Remove final FC → output is after adaptive avg pool = (B, 512, 1, 1)
    model.fc = torch.nn.Identity()

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


def get_transform(image_size: int = config.IMAGE_SIZE) -> T.Compose:
    """
    ImageNet-standard preprocessing:
    1. Resize to image_size
    2. Convert grayscale to 3-channel
    3. ToTensor → [0, 1]
    4. Normalize with ImageNet mean/std
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=3),  # repeat gray → 3ch
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


@torch.no_grad()
def extract_features(
    dataset,
    model: torch.nn.Module,
    transform: T.Compose,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract 512-d features for every image in the dataset.

    Parameters
    ----------
    dataset : medmnist dataset with .imgs attribute
    model : frozen ResNet-18 extractor
    transform : preprocessing pipeline

    Returns
    -------
    features : np.ndarray of shape (n_images, 512)
    """
    images = dataset.imgs  # (N, H, W) or (N, H, W, C)
    n = len(images)
    all_features = []

    for start in tqdm(range(0, n, batch_size), desc="Extracting features"):
        end = min(start + batch_size, n)
        batch_imgs = images[start:end]

        # Apply transform to each image
        tensors = []
        for img in batch_imgs:
            if img.ndim == 2:
                img = img  # grayscale
            elif img.ndim == 3 and img.shape[-1] == 1:
                img = img[:, :, 0]  # squeeze channel dim
            elif img.ndim == 3 and img.shape[-1] == 3:
                pass  # already RGB
            t = transform(img)
            tensors.append(t)

        batch_tensor = torch.stack(tensors).to(device)  # (B, 3, 224, 224)
        feats = model(batch_tensor)  # (B, 512)

        # Flatten just in case shape is (B, 512, 1, 1)
        feats = feats.view(feats.size(0), -1)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def get_cache_path(split: str) -> Path:
    """Return cache file path for a split."""
    return config.CACHE_DIR / f"resnet18_features_{split}.npy"


def extract_and_cache_features(
    dataset,
    split: str,
    device: str = "cpu",
    force: bool = False,
) -> np.ndarray:
    """
    Extract features and cache to disk.
    Loads from cache on subsequent calls.
    """
    cache_path = get_cache_path(split)

    if cache_path.exists() and not force:
        print(f"  Loading cached features from {cache_path}")
        features = np.load(cache_path)
    else:
        print(f"  Extracting ResNet-18 features for '{split}'...")
        model = get_resnet18_extractor(device)
        transform = get_transform()
        features = extract_features(dataset, model, transform, device)
        np.save(cache_path, features)
        print(f"  Cached to {cache_path} | shape={features.shape}")

    return features


def extract_all_splits(
    train_ds, val_ds, test_ds,
    device: str = "cpu",
    force: bool = False,
) -> dict:
    """Extract and cache features for all three splits."""
    results = {}
    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        results[name] = extract_and_cache_features(ds, name, device, force)
    return results


if __name__ == "__main__":
    from data import load_pneumoniamnist

    train_ds, val_ds, test_ds, info = load_pneumoniamnist()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    feats = extract_all_splits(train_ds, val_ds, test_ds, device)

    for name, f in feats.items():
        assert f.ndim == 2 and f.shape[1] == 512, f"Bad shape: {f.shape}"
        assert not np.any(np.isnan(f)), f"NaN in {name} features!"
        print(f"  {name}: shape={f.shape}, min={f.min():.4f}, max={f.max():.4f}")

    # Quick logistic regression sanity check
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Use 200 training images
    n_check = min(200, len(feats["train"]))
    X_train = feats["train"][:n_check]
    y_train = train_ds.labels.flatten()[:n_check]
    X_test = feats["test"]
    y_test = test_ds.labels.flatten()

    lr = LogisticRegression(max_iter=1000, random_state=0)
    lr.fit(X_train, y_train)
    y_prob = lr.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n  Logistic Regression sanity check (N={n_check}): AUC = {auc:.4f}")
    assert auc > 0.6, f"AUC too low ({auc:.4f}), features may be bad!"
    print("  ✓ Features look good!")
