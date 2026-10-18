"""
QKernel-Lite – Central Configuration
=====================================
All frozen experiment specifications live here.
Do NOT scatter magic numbers elsewhere.
"""

import numpy as np
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
CACHE_DIR = ROOT_DIR / "cache"
RESULTS_DIR = ROOT_DIR / "results"
NOTES_DIR = ROOT_DIR / "notes"

CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Dataset ───────────────────────────────────────────────────────────
DATASET_NAME = "pneumoniamnist"
IMAGE_SIZE = 224          # try 224 first; fall back to 64 + upsample
INPUT_CHANNELS = 3        # grayscale repeated to 3-ch for ResNet

# ─── Experiment grid ──────────────────────────────────────────────────
N_VALUES = [10, 20, 50, 100]
SEEDS = list(range(15))            # 15 seeds (0..14) for main comparison
MAIN_SEEDS = list(range(15))
ABLATION_SEEDS = list(range(15))   # 15 seeds (0..14) for 2x2 ablation grid

# ─── Feature extraction ───────────────────────────────────────────────
FEATURE_DIM = 512                 # ResNet-18 GAP output

# ─── PCA ───────────────────────────────────────────────────────────────
PCA_COMPONENTS = 8                # main run
PCA_FIT_ON = "subset"             # "subset" (default) or "pool"

# ─── Angle scaling ────────────────────────────────────────────────────
ANGLE_MIN = 0.0
ANGLE_MAX = np.pi

# ─── M1: Small CNN from scratch ──────────────────────────────────────
M1_LR = 1e-3
M1_WEIGHT_DECAY = 1e-4
M1_EPOCHS = 60
M1_BATCH_SIZE_CAP = 16            # batch = min(16, N)

# ─── M2 / M3: SVM tuning grid ────────────────────────────────────────
GAMMA_MULTIPLIERS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
C_VALUES = [0.1, 1.0, 10.0, 100.0]

def get_cv_folds(n_train: int) -> int:
    """Stratified k-fold: k=3 for N≤20, else 5."""
    return 3 if n_train <= 20 else 5

# ─── Quantum circuit ──────────────────────────────────────────────────
DEFAULT_QUBITS = 8
DEFAULT_LAYERS = 2
DEFAULT_ENTANGLEMENT = True

# Ablation grid
QUBIT_ABLATION = [4, 6, 8]
ENTANGLEMENT_ABLATION = [True, False]
LAYER_ABLATION = [1, 2]
ENCODING_SCALES = [0.0625, 0.125, 0.25, 0.5, 1.0]

# Kernel diagnostics alarm
KERNEL_STD_ALARM = 0.01           # off-diagonal std below this is a red flag

# ─── Results CSV ──────────────────────────────────────────────────────
RESULTS_CSV = RESULTS_DIR / "results.csv"
RESULTS_COLUMNS = [
    "model", "N", "seed", "qubits", "entanglement", "layers",
    "auc", "balanced_accuracy", "f1",
    "auc_notuned",
    "best_scale", "scale_edge",
    "auc_s00625", "auc_s0125", "auc_s025", "auc_s050", "auc_s100",
    "best_gamma", "best_C", "gamma_edge", "C_edge",
    "feat_variance", "dist_std_over_median", "dist_spearman_rank_corr",
    "feat_variance_s100", "dist_std_over_median_s100", "dist_spearman_rank_corr_s100",
    "feat_variance_s025", "dist_std_over_median_s025", "dist_spearman_rank_corr_s025",
    "kernel_offdiag_mean", "kernel_offdiag_std", "kernel_min_eigenvalue",
    "runtime_seconds",
]
