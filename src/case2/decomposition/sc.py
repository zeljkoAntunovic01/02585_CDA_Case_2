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
from matplotlib import cm, lines

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser(
).parent.parent.parent.parent / 'docs' / 'figures' / 'sc'
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
    H_SC = model.fit_transform(X_scaled)
    return scaler, model, H_SC


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


def plot_sc_scatter_emotions(
    S1: np.ndarray,
    S2: np.ndarray,
    y: pd.DataFrame,
    emotions: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save a grid of SC 2-D scatter plots, one per emotion label in `emotions`,
    coloring the points by the intensity of that emotion.
    """
    n = len(emotions)
    cols = 3
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4, rows * 4),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    for i, emo in enumerate(emotions):
        ax = axes[i]
        if emo not in y.columns:
            ax.set_visible(False)
            continue

        # scatter colored by this emotion’s ratings
        sc = ax.scatter(
            S1, S2,
            c=y[emo].values,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_title(emo)
        ax.set_xlabel('SC 1')
        ax.set_ylabel('SC 2')

        # individual colorbar per subplot
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
        cbar.set_label(emo)

    # turn off any extra axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / 'sp_scatter_emotions.png'
    fig.savefig(save_path)
    plt.close(fig)


def plot_sc_h_elements(H_SC: np.ndarray, lambda_: float) -> None:
    # Correct way to get both fig and ax
    fig, ax2 = plt.subplots(figsize=(15, 15))

    # Plot histogram of H
    ax2.hist(H_SC.flatten(), bins=100, stacked=True)
    ax2.set_ylabel('Number of elements in H')
    ax2.set_xlabel('Bin value')
    ax2.set_title(
        f'Histogram of the values in H for sparse coding, lambda={lambda_}')

    save_path = FIGURE_DIR / 'sp_h_elements.png'
    fig.savefig(save_path)
    plt.close(fig)


def plot_sc_scatter_phase(
    S1: np.ndarray,
    S2: np.ndarray,
    phases: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save scatter of the first two independent components,
    colouring points by the 'Phase' category with a matching legend.
    """
    # Wrap phases in a pandas Index so factorize gets a supported type
    phases_idx = pd.Index(phases)

    # factorize without warning
    codes, uniques = pd.factorize(phases_idx)

    n_phases = len(uniques)
    cmap = cm.get_cmap('viridis', n_phases)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        S1,
        S2,
        c=codes,
        cmap=cmap,
        alpha=0.7
    )

    ax.set_xlabel('SC 1')
    ax.set_ylabel('SC 2')
    ax.set_title('SC Embedding by Phase')

    # Build a legend that matches the scatter colors
    handles = []
    for i, phase in enumerate(uniques):
        handles.append(
            lines.Line2D(
                [], [],
                marker='o',
                linestyle='',
                color=cmap(i),
                label=phase,
                alpha=0.7
            )
        )
    ax.legend(
        handles=handles,
        title='Phase',
        loc='best',
        frameon=False
    )

    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / 'sc_embedding_phase.png'
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
    # evs = []
    # k_list = [5, 10, 15, 20, 25, 30, 35, 38, 40, 42, 45, 50]
    # alpha = 0.0001
    # for k in k_list:
    #     scaler, model = compute_sc(X_np, k, alpha)
    #     X_scaled = scaler.transform(X_np)
    #     X_transformed = model.transform(X_scaled)
    #     X_val_recon = X_transformed @ model.components_
    #     ev = explained_variance_score(X_scaled, X_val_recon)
    #     evs.append(ev)
    #     print(f"Evaluated k={k}, alpha={alpha}, EV={ev:.8f}")

    # 3) Plot explained variance

    # Plot
    # fig, ax = plt.subplots()
    # ax.plot(k_list, evs, marker='o', label='Explained Variance')

    # # Determine chosen k as the smallest k with maximal explained variance
    # print(evs)
    # max_ev = 1.0
    # eps = 0.0005
    # chosen = None
    # for k_val, ev_val in zip(k_list, evs):
    #     if ev_val >= max_ev - eps:
    #         chosen = k_val
    #         break
    # print(f"Chosen k: {chosen}")
    # if chosen is None:
    #     raise ValueError("No suitable k found.")

    # ax.axvline(chosen, color='grey', linestyle='--',
    #            label=f'Selected k={chosen}')
    # ax.set_xlabel('Number of Components (k)')
    # ax.set_ylabel('Explained Variance')
    # ax.set_title('SC Explained Variance vs Number of Components')
    # ax.legend()
    # fig.tight_layout()
    # save_path = FIGURE_DIR / 'sc_nb_features_explained_variance.png'
    # fig.savefig(save_path)
    # plt.close(fig)

    # # Print selected k
    # print(f"Selected k = {chosen}")

    chosen = 40
    alpha = 0.006
    # 4) Fit final SC and visualize mixing matrix
    scaler, model, H_SC = compute_sc(X_np, chosen, alpha)
    # 5) Plot mixing matrix
    plot_sc_h_elements(H_SC.T, alpha)

    # 6) Plot scatter of SC components colored by emotion labels
    mean_usage = np.mean(np.abs(H_SC), axis=0)
    top_atoms = np.argsort(mean_usage)[::-1][:2]
    emotions = ['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined']
    plot_sc_scatter_emotions(
        S1=H_SC[:, top_atoms[0]],
        S2=H_SC[:, top_atoms[1]],
        y=y,
        emotions=emotions
    )
    # 7) Plot scatter of SC components colored by phase labels
    phases = y.index.get_level_values('Phase').astype(str).tolist()
    plot_sc_scatter_phase(
        S1=H_SC[:, top_atoms[0]],
        S2=H_SC[:, top_atoms[1]],
        phases=phases
    )
    # return evs
