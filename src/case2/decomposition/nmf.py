# src/nmf.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from typing import List, Tuple, Optional
from pathlib import Path

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser().parent.parent.parent.parent / 'docs' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_for_nmf(
    X: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute missing values and shift data to be non-negative.
    """
    X_nonneg = X.copy().astype(float)
    X_nonneg = X_nonneg.fillna(X_nonneg.median())
    shifts = pd.Series(index=X_nonneg.columns, dtype=float)
    for col in X_nonneg.columns:
        min_val = X_nonneg[col].min()
        shifts[col] = min_val
        if min_val < 0:
            X_nonneg[col] -= min_val
    return X_nonneg, shifts


def compute_nmf(
    X: np.ndarray,
    n_components: int,
    init: str = 'nndsvda',
    solver: str = 'cd',
    random_state: int = 0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit NMF with given init and solver, returning W, H, and reconstruction error.
    """
    model = NMF(n_components=n_components, init=init,
                solver=solver, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    err = model.reconstruction_err_
    return W, H, err


def evaluate_nmf(
    X: pd.DataFrame,
    k_list: List[int],
    init: str = 'nndsvda',
    solver: str = 'cd'
) -> List[float]:
    """
    Compute reconstruction errors for each k in k_list.
    """
    X_np = X.values
    return [compute_nmf(X_np, k, init=init, solver=solver)[2] for k in k_list]


def plot_explained_variance(
    k_list: List[int],
    explained_var: List[float],
    chosen_k: Optional[int] = None,
    save_path: Optional[Path] = None
) -> None:
    """
    Save explained variance vs k plot, with optional vertical line at chosen_k.
    """
    fig, ax = plt.subplots()
    ax.plot(k_list, explained_var, marker='o', label='Explained Variance')
    if chosen_k is not None:
        ax.axvline(chosen_k, color='grey', linestyle='--', label=f'Selected k={chosen_k}')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Explained Variance')
    ax.set_title('NMF Explained Variance vs k')
    if chosen_k is not None:
        ax.legend()
    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / 'explained_variance.png'
    fig.savefig(save_path)
    plt.close(fig)


def plot_components_grid(
    H: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save bar-chart grid of each NMF component.
    """
    n_components = H.shape[0]
    cols = 3
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    indices = np.arange(len(feature_names))
    for i, comp in enumerate(H):
        ax = axes[i]
        ax.bar(indices, comp)
        ax.set_xticks(indices)
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_title(f'Component {i+1}')
    for j in range(n_components, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / 'nmf_components_grid.png'
    fig.savefig(save_path)
    plt.close(fig)


def run_nmf_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k_list: List[int] = list(range(2, 16)),
    select_k: Optional[int] = None
) -> Tuple[None, None, List[float]]:
    """
    Execute the NMF analysis: plot explained variance curve with vline at chosen k,
    then component bars.
    """

    print("Running NMF pipeline...")
    
    # 1) Preprocess and evaluate
    X_nonneg, _ = preprocess_for_nmf(X)
    errs = evaluate_nmf(X_nonneg, k_list)

    # 2) Compute explained variance
    total_ss = np.linalg.norm(X_nonneg.values)**2
    explained_var = [1 - (err**2)/total_ss for err in errs]

    # 3) Determine chosen k
    chosen = select_k or k_list[int(np.argmax(explained_var))]
    print(f"Selected k = {chosen} with VE = {explained_var[k_list.index(chosen)]:.4f}")

    # 4) Plot explained variance with vertical marker
    plot_explained_variance(k_list, explained_var, chosen_k=chosen)

    # 5) Bar-chart grid of components
    _, H_sel, _ = compute_nmf(X_nonneg.values, chosen)
    plot_components_grid(H_sel, list(X.columns))

    return None, None, errs

