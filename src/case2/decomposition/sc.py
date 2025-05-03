import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from scipy.stats import kurtosis
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import warnings
from pathlib import Path
import py_pcha
from matplotlib import cm, lines
import seaborn as sns
sns.set_theme(style="darkgrid")

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
                               max_iter=1500, transform_max_iter=1500, fit_algorithm='cd', transform_algorithm='lasso_cd')
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

    # Grid of hyperparameters
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    k_list = [5, 10, 20, 30, 40, 50]
    results = []

    # Split for validation
    X_train, X_val = train_test_split(X_np, test_size=0.2, random_state=42)

    for alpha in alpha_list:
        for k in k_list:
            scaler, model, _ = compute_sc(X_train, k, alpha)
            X_val_scaled = scaler.transform(X_val)

            X_transformed = model.transform(X_val_scaled)
            X_recon = X_transformed @ model.components_
            ev = explained_variance_score(X_val_scaled, X_recon)
            results.append((k, alpha, ev))
            print(f"Alpha={alpha}, k={k}, EV={ev:.6f}")

    # Format results
    results_df = pd.DataFrame(
        results, columns=['noc', 'lambda', 'EV'])
    results_df['lambda'] = results_df['lambda'].astype(float)
    results_df['noc'] = results_df['noc'].astype(int)
    results_df['EV'] = results_df['EV'].astype(float)

    # Select best: smallest k, largest alpha that achieves near-max EV
    eps = 1e-3
    max_ev = results_df['EV'].max()
    valid = results_df[results_df['EV'] >= max_ev - eps]
    best_row = valid.sort_values(
        ['noc', 'lambda'], ascending=[True, False]).iloc[0]
    print(
        f"Selected alpha={best_row['lambda']}, k={best_row.noc}, EV={best_row.EV:.5f}")

    # Optional: Plot heatmap
    fig, ax = plt.subplots()
    heatmap_data = results_df.pivot(index='lambda', columns='noc', values='EV')
    sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap='viridis')
    # Mark selected point
    best_k = int(best_row.noc)
    best_alpha = float(best_row['lambda'])
    ax.plot(
        k_list.index(best_k) + 0.5,  # x-coord (column index + 0.5 for center)
        alpha_list.index(best_alpha) + 0.5,  # y-coord
        marker='o', color='red', markersize=10, markeredgecolor='black'
    )
    ax.set_title("Sparse Coding EV with lambda and number of components (noc)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "sc_elementwise_cv.png", dpi=300)
    plt.close()

    # 4) Fit final SC and visualize mixing matrix
    scaler, model, H_SC = compute_sc(X_np, best_k, best_alpha)
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
