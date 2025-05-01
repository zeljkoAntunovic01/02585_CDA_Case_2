import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from typing import List, Tuple, Optional
from sklearn.exceptions import ConvergenceWarning
import warnings
from pathlib import Path
import py_pcha

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser(
).parent.parent.parent.parent / 'docs' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_for_aa(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute missing values, then center data for ICA.
    """
    # Impute missing values with column medians
    X_imputed = X.fillna(X.median())
    return X_imputed


def compute_aa(
    X: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray]:
    '''
        Output
        ------
        XC : numpy.2darray
            I x noc feature matrix (i.e. XC=X[:,I]*C forming the archetypes)

        S : numpy.2darray
            noc X n matrix, S>=0 |S_j|_1=1

        C : numpy.2darray
            x x noc matrix, C>=0 |c_j|_1=1

        SSE : float
            Sum of Squared Errors
    '''
    XC, S, C, SSE, varexpl = py_pcha.PCHA(X.T, noc=n_components, delta=0.1)
    X_hat = X.T @ C @ S
    L = 0.5*np.linalg.norm(X.T-X_hat)**2
    return XC, varexpl


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


def plot_archetypes(
    XC: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save bar-chart grid of each AA archetype.
    """
    n_components = XC.shape[1]
    assert XC.shape[0] == len(
        feature_names), "Mismatch: XC rows must match number of feature names"
    cols = 2
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    indices = np.arange(len(feature_names))
    for i in range(n_components):
        mix = np.asarray(XC[:, i]).astype(float).flatten()
        ax = axes[i]
        ax.bar(indices, mix)
        ax.set_xticks(indices)
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_title(f'Archetype {i+1}')
    for j in range(n_components, len(axes)):
        axes[j].axis('off')
    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / 'aa_archetypes.png'
    fig.savefig(save_path)
    plt.close(fig)


def run_aa_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k_list: List[int] = list(range(2, 10)),
    select_k: Optional[int] = None
) -> List[float]:
    """
    Execute AA analysis: explained variance vs noc, archetypes.
    Returns list of explained variance per noc.
    """

    print("Running AA pipeline...")

    # 1) Preprocess data
    Xc = preprocess_for_aa(X)
    X_np = Xc.values
    # 2) Evaluate explained variance metric
    evs = []
    for k in k_list:
        XC, varexp = compute_aa(X_np, k)
        evs.append(varexp)
        # 3) Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(k_list, evs, marker='o', label='Explained Variance')

    # Determine chosen k as the smallest k with maximal explained variance
    max_ev = 1.0
    eps = 5e-4
    chosen = None
    print(evs)
    for k_val, ev_val in zip(k_list, evs):
        if ev_val >= max_ev - eps:
            chosen = k_val
            break
    print(f"Chosen k: {chosen}")
    if chosen is None:
        raise ValueError("No suitable k found.")

    ax.axvline(chosen, color='grey', linestyle='--',
               label=f'Selected k={chosen}')
    ax.set_xlabel('Number of archetypes')
    ax.set_ylabel('Explained Variance')
    ax.set_title('AA Explained Variance vs Number of Archetypes')
    ax.legend()
    fig.tight_layout()
    save_path = FIGURE_DIR / 'aa_explained_variance.png'
    fig.savefig(save_path)
    plt.close(fig)

    # Print selected k
    print(f"Selected k = {chosen} with VE = {max_ev:.4f}")

    # 4) Fit final ICA and visualize mixing matrix
    XC_sel, varexp_sel = compute_aa(X_np, chosen)
    plot_archetypes(XC_sel, list(X.columns))

    return evs
