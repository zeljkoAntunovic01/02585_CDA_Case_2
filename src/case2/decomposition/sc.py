import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from scipy.stats import kurtosis
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import KFold
import warnings
from pathlib import Path
import py_pcha

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser(
).parent.parent.parent.parent / 'docs' / 'figures'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_for_sc(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute missing values.
    """
    # Impute missing values with column medians
    X_imputed = X.fillna(X.median())
    return X_imputed


def compute_sc(
    X: np.ndarray,
    n_components: int,
    lambda_: float
) -> Tuple[np.ndarray, np.ndarray]:
    '''

    '''
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = DictionaryLearning(n_components=n_components, alpha=lambda_, transform_alpha=lambda_,
                               max_iter=1000, transform_max_iter=1000, fit_algorithm='cd', transform_algorithm='lasso_cd')
    model.fit(X_scaled)
    return scaler, model


def evaluate_sc(
    X: np.ndarray,
    n_components: int,
    lambda_: float,
    X_val: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate SC model and return explained variance.
    """
    scaler, model = compute_sc(X, n_components, lambda_)
    X_scaled = scaler.transform(X_val)
    X_transformed = model.transform(X_scaled)
    X_val_recon = X_transformed @ model.components_

    ev = explained_variance_score(X_scaled, X_val_recon)
    return ev, X_transformed


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
    cols = 2
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    indices = np.arange(len(feature_names))
    for i in range(n_components):
        mix = XC[:, i]
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


def run_sc_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    select_k: Optional[int] = None
) -> List[float]:
    """
    Execute SC analysis: explained variance vs noc, archetypes.
    Returns list of explained variance per noc.
    """

    print("Running SC pipeline...")

    # 1) Preprocess data
    Xc = preprocess_for_sc(X)
    X_np = Xc.values
    # 2) Evaluate explained variance metric
    evs = []
    k_list = [5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50]
    alpha = 0.0001
    for k in k_list:
        scaler, model = compute_sc(X_np, k, alpha)
        X_scaled = scaler.transform(X_np)
        X_transformed = model.transform(X_scaled)
        X_val_recon = X_transformed @ model.components_
        ev = explained_variance_score(X_scaled, X_val_recon)
        evs.append(ev)
        print(f"Evaluated k={k}, alpha={alpha}, EV={ev:.8f}")

    # 3) Plot explained variance

    # Plot
    fig, ax = plt.subplots()
    ax.plot(k_list, evs, marker='o', label='Explained Variance')

    # Determine chosen k as the smallest k with maximal explained variance
    print(evs)
    max_ev = 1.0
    eps = 0.0005
    chosen = None
    for k_val, ev_val in zip(k_list, evs):
        if ev_val >= max_ev - eps:
            chosen = k_val
            break
    print(f"Chosen k: {chosen}")
    if chosen is None:
        raise ValueError("No suitable k found.")

    ax.axvline(chosen, color='grey', linestyle='--',
               label=f'Selected k={chosen}')
    ax.set_xlabel('Number of Components (k)')
    ax.set_ylabel('Explained Variance')
    ax.set_title('SC Explained Variance vs Number of Components')
    ax.legend()
    fig.tight_layout()
    save_path = FIGURE_DIR / 'sc_nb_features_explained_variance.png'
    fig.savefig(save_path)
    plt.close(fig)

    # Print selected k
    print(f"Selected k = {chosen}")

    # 4) Fit final SC and visualize mixing matrix
    scaler, model = compute_sc(X_np, chosen, alpha)

    return evs
