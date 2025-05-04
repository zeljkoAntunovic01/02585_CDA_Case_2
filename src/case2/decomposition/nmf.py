# src/nmf.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from typing import List, Tuple, Optional
from pathlib import Path
import seaborn as sns
import math

sns.set_theme(style="darkgrid")


# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser().parent.parent.parent.parent / 'docs' / 'figures' / 'nmf' 
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_for_nmf(
    X: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Impute missing values, shift data to be non-negative, and drop any zero-variance features.
    
    Parameters
    ----------
    X : pd.DataFrame
        Original feature matrix.

    Returns
    -------
    X_nonneg : pd.DataFrame
        The non-negative, imputed feature matrix with zero-variance columns removed.
    shifts : pd.Series
        The amount each original column was shifted (min value).
    """
    # 1) Copy & impute
    X_nonneg = X.copy().astype(float)
    X_nonneg = X_nonneg.fillna(X_nonneg.median())

    # 2) Record shifts & make non-negative
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
    init: str = 'random',
    solver: str = 'mu',
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

def plot_nmf_subplots(
    W: np.ndarray,
    y: pd.DataFrame,
    save_path: Optional[Path] = None,
    method_name: str = "NMF"
) -> None:
    """
    Plot NMF embedding (first two components) in subplots for:
      - 'Phase'
      - 'Puzzler'
      - each emotion column in y
    Uses the same style/layout as the PCA subplots,
    but skips any points where W[:,0] or W[:,1] is NaN.
    """
    # 1) Build titles dict
    phases = np.sort(y.index.get_level_values("Phase").unique())
    titles = {"Phase": phases}
    for col in y.columns:
        if col not in titles and col not in ["Individual", "Round"]:
            titles[col] = np.sort(y[col].unique())

    # 2) Compute valid‐point mask once
    valid = np.isfinite(W[:, 0]) & np.isfinite(W[:, 1])

    # 3) Grid dimensions
    n_plots = len(titles)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 7 * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    # 4) Scatter by each label, filtering out NaNs
    for ax, (title, unique_vals) in zip(axes, titles.items()):
        for val in unique_vals:
            try:
                mask = y.index.get_level_values(title) == val
            except KeyError:
                mask = y[title] == val

            # combine with valid mask
            combined = mask & valid
            idx = np.where(combined)[0]
            if idx.size == 0:
                continue

            ax.scatter(
                W[idx, 0], W[idx, 1],
                label=str(val),
                alpha=0.7
            )

        # PCA‐style axes and grid
        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.legend()
        ax.grid(True)
        ax.set_title(f"{method_name} - {title}")

    # 5) Disable any extra axes
    for ax in axes[len(titles):]:
        ax.set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURE_DIR / f"{method_name.lower()}_subplots.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
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

    # after computing W_sel, H_sel in run_nmf_pipeline...
    W_sel, H_sel, _ = compute_nmf(X_nonneg.values, chosen)

    # existing bar‐grid
    plot_components_grid(H_sel, list(X.columns))    

    # new NMF scatter‐subplots
    print("Plotting NMF scatter subplots for Phase, Puzzler, and emotions...")
    plot_nmf_subplots(
        W_sel,
        y,
        save_path=FIGURE_DIR / "nmf_subplots.png"
    )

    return None, None, errs

