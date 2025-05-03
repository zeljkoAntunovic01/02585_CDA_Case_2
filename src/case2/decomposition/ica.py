# src/ica.py

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from typing import List, Tuple, Optional
from sklearn.exceptions import ConvergenceWarning
import warnings
from pathlib import Path
from matplotlib import cm, lines

import seaborn as sns

sns.set_theme(style="darkgrid")

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser().parent.parent.parent.parent / 'docs' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_for_ica(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute missing values, then center data for ICA.
    """
    # Impute missing values with column medians
    X_imputed = X.fillna(X.median())
    # Center the data
    Xc = X_imputed - X_imputed.mean()
    return Xc

# def preprocess_for_ica(X: pd.DataFrame) -> pd.DataFrame:
#     X_imputed = X.fillna(X.median())
#     Xc = X_imputed - X_imputed.mean()
#     # drop constant columns
#     zero_var = Xc.std() <= 0
#     if zero_var.any():
#         print("Dropping zero-variance features:", list(Xc.columns[zero_var]))
#         Xc = Xc.loc[:, ~zero_var]
#     return Xc

def compute_ica(
    X: np.ndarray,
    n_components: int,
    random_state: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit ICA to X, returning sources S and mixing matrix A, with increased iterations
    and tolerance to address convergence issues.

    S has shape (n_samples, n_components); A is (n_components, n_features).
    """
    ica = FastICA(
        n_components=n_components,algorithm='parallel'
        # n_components=n_components,
        # whiten='unit-variance',
        # random_state=random_state,
        # max_iter=max_iter,
        # tol=tol
    )
    # Suppress convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        S = ica.fit_transform(X)
    A = ica.mixing_.T  # shape (n_components, n_features)
    return S, A


def evaluate_ica(
    X: pd.DataFrame,
    k_list: List[int]
) -> List[float]:
    """
    Compute average absolute kurtosis of independent components as measure of non-Gaussianity.
    """
    Xc = preprocess_for_ica(X)
    vals = []
    for k in k_list:
        S, _ = compute_ica(Xc.values, k)
        # Compute kurtosis per component, fisher=True
        kurts = kurtosis(S, axis=0, fisher=True, bias=False)
        vals.append(np.mean(np.abs(kurts)))
    return vals


def plot_metric(
    k_list: List[int],
    metric: List[float],
    ylabel: str,
    title: str,
    fname: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Save a line plot of metric vs k.
    """
    fig, ax = plt.subplots()
    ax.plot(k_list, metric, marker='o')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / fname
    fig.savefig(save_path)
    plt.close(fig)


def plot_mixing_matrix_grid(
    A: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save bar-chart grid of each ICA mixing vector (rows of A).
    """
    n_components = A.shape[0]
    cols = 2
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    indices = np.arange(len(feature_names))
    for i, mix in enumerate(A):
        ax = axes[i]
        ax.bar(indices, mix)
        ax.set_xticks(indices)
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_title(f'Mixing {i+1}')
    for j in range(n_components, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / "ica" / 'ica_mixing_grid.png'
    fig.savefig(save_path)
    plt.close(fig)

def plot_ica_subplots(
    S: np.ndarray,
    y: pd.DataFrame,
    save_path: Optional[Path] = None,
    method_name: str = "ICA"
) -> None:
    """
    Plot ICA embedding (first two components) in subplots for:
      - 'Phase'
      - 'Puzzler'
      - each emotion column in y
    Uses the same style/layout as the PCA subplots.
    """
    # Build titles dictionary just like PCA
    phases = np.sort(y.index.get_level_values("Phase").unique())
    titles = {"Phase": phases}
    for col in y.columns:
        if col not in titles:
            titles[col] = np.sort(y[col].unique())

    # Grid size
    n_plots = len(titles)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(7 * ncols, 7 * nrows))
    axes = axes.flatten()

    valid = np.isfinite(S[:, 0]) & np.isfinite(S[:, 1])

    # For each label type, plot S[:,0] vs S[:,1]
    for ax, (title, unique_values) in zip(axes, titles.items()):

        for val in unique_values:
            # original mask for this category
            try:
                mask = y.index.get_level_values(title) == val
            except KeyError:
                mask = y[title] == val

            # combine with “valid” mask to drop NaNs
            combined = mask & valid
            idx = np.where(combined)[0]
            if len(idx) == 0:
                continue

            ax.scatter(
                S[idx, 0],
                S[idx, 1],
                label=str(val),
                alpha=0.7
            )
        
        ax.set_xlabel("IC 1")
        ax.set_ylabel("IC 2")
        # dashed zero‐lines
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.legend()
        ax.grid(True)
        ax.set_title(f"{method_name} - {title}")

    # hide unused axes
    for ax in axes[len(titles):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURE_DIR / "ica" / "ica_subplots.png"
    # ensure subfolder exists
    (save_path.parent).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close(fig)

def run_ica_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k_list: List[int] = list(range(2, 16)),
    select_k: Optional[int] = None
) -> List[float]:
    """
    Execute ICA analysis: explained variance vs k, mixing grid, scatter embedding.
    Returns list of explained variance per k.
    """

    print("Running ICA pipeline...")
    
    # 1) Preprocess data
    Xc = preprocess_for_ica(X)
    X_np = Xc.values
    # 2) Evaluate explained variance metric
    total_ss = np.linalg.norm(X_np)**2
    evs: List[float] = []
    for k in k_list:
        S, A = compute_ica(X_np, k)
        Xhat = np.dot(S, A)
        sse = np.linalg.norm(X_np - Xhat)**2
        evs.append(1 - sse/total_ss)
        # 3) Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(k_list, evs, marker='o', label='Explained Variance')

    # Determine chosen k as the smallest k with maximal explained variance
    # max_ev = 1.0
    max_ev = max(evs)
    eps = 1e-4
    chosen = None
    for k_val, ev_val in zip(k_list, evs):
        if ev_val >= max_ev - eps:
            chosen = k_val
            break

    ax.axvline(chosen, color='grey', linestyle='--', label=f'Selected k={chosen}')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Explained Variance')
    ax.set_title('ICA Explained Variance vs k')
    ax.legend()
    fig.tight_layout()
    save_path = FIGURE_DIR / "ica" / 'ica_explained_variance.png'
    fig.savefig(save_path)
    plt.close(fig)

    # Print selected k
    print(f"Selected k = {chosen} with VE = {max_ev:.4f}")

    # 4) Fit final ICA and visualize mixing matrix
    S_sel, A_sel = compute_ica(X_np, chosen)
    plot_mixing_matrix_grid(A_sel, list(X.columns))

    # 5+6) Single figure with Phase, Puzzler, and all emotions
    print("Plotting ICA scatter for Phase, Puzzler, and emotions...")
    plot_ica_subplots(S_sel, y)

    return evs
