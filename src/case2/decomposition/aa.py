import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from typing import List, Tuple, Optional
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
import py_pcha
import math
import seaborn as sns
sns.set_theme(style="darkgrid")

# Directory to save figures
FIGURE_DIR = Path(__file__).expanduser(
).parent.parent.parent.parent / 'docs' / 'figures' / 'aa'
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_for_aa(
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Impute missing values, then center data for ICA.
    """
    # Impute missing values with column medians
    X_imputed = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled+10


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
    closest_archetypes = np.argmax(C, axis=1)
    closest_archetypes = np.asarray(closest_archetypes).squeeze()
    if closest_archetypes.ndim != 1:
        raise ValueError(
            f"Expected 1D array, got shape {closest_archetypes.shape}")
    return XC, varexpl, closest_archetypes

def plot_closest_archetypes(
    closest_archetypes: np.ndarray,
    feature_names: List[str],
    y: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Strip‐plot of each emotion score by the closest archetype,
    with explicit axis labels on each facet.
    """
    # Add closest archetype column
    df_plot = y.copy()
    df_plot["Archetype"] = closest_archetypes.astype(str)

    # Reshape to long format
    df_long = df_plot.melt(
        id_vars="Archetype",
        value_vars=feature_names,
        var_name="Emotion",
        value_name="Score"
    )

    # Create FacetGrid
    g = sns.FacetGrid(
        df_long,
        col="Emotion",
        col_wrap=3,
        height=4,
        sharey=False
    )
    g.map(
        sns.stripplot,
        "Archetype",
        "Score",
        jitter=True,
        alpha=0.7
    )
    # Add axis labels
    g.set_axis_labels("Archetype", "Score")

    # Rotate x‐ticks
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=45)

    g.fig.suptitle("Emotion Scores by Closest Archetype", y=1.02)
    g.tight_layout()

    save_path = save_path or FIGURE_DIR / "emotion_scores_by_archetype_strip.png"
    g.savefig(save_path, dpi=300)


def plot_phases_archetypes(
    closest_archetypes: np.ndarray,
    y: pd.DataFrame,
    save_path: Optional[Path] = None
) -> None:
    """
    Strip‐plot of Phase vs. Archetype, with explicit axis labels.
    """
    df_plot = y.copy()
    df_plot["Archetype"] = closest_archetypes.astype(str)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.stripplot(
        data=df_plot,
        x="Phase",
        y="Archetype",
        jitter=True,
        alpha=0.7,
        ax=ax
    )

    ax.set_xlabel("Phase")
    ax.set_ylabel("Archetype")
    ax.set_title("Phase by Closest Archetype", y=1.02)
    ax.grid(True)

    save_path = save_path or FIGURE_DIR / "phase_by_archetype.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_archetypes(
    XC: np.ndarray,
    feature_names: List[str],
    save_path: Optional[Path] = None
) -> None:
    """
    Save bar‐chart grid of each AA archetype, with axis labels on each subplot.
    """
    n_components = XC.shape[1]
    cols = 2
    rows = int(np.ceil(n_components / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    indices = np.arange(len(feature_names))

    for i in range(n_components):
        mix = np.asarray(XC[:, i]).astype(float).flatten()
        ax = axes[i]
        ax.bar(indices, mix)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        ax.set_xticks(indices)
        ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
        ax.set_title(f"Archetype {i+1}")

    for j in range(n_components, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    save_path = save_path or FIGURE_DIR / "aa_archetypes.png"
    fig.savefig(save_path)
    plt.close(fig)


def plot_aa_explained_variance(
    k_list: List[int],
    evs: List[float],
    chosen: int,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot AA explained variance vs number of archetypes in the same style as ICA/NMF.
    """
    # match the default subplot size and font setup
    fig, ax = plt.subplots()
    ax.plot(k_list, evs, marker='o', label='Explained Variance')
    ax.axvline(chosen, color='grey', linestyle='--', label=f'Selected k={chosen}')
    ax.set_xlabel('Number of components (k)')
    ax.set_ylabel('Explained Variance')
    ax.set_title('AA Explained Variance vs Number of Archetypes')
    ax.legend()
    fig.tight_layout()

    save_path = save_path or FIGURE_DIR / 'aa_explained_variance.png'
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_aa_subplots(
    closest_archetypes: np.ndarray,
    y: pd.DataFrame,
    save_path: Optional[Path] = None,
    method_name: str = "AA"
) -> None:
    """
    Plot archetype assignments in subplots for:
      - 'Phase'
      - 'Puzzler'
      - each emotion column in y
    in the same PCA‐style grid (7×7" per subplot, dashed zero‐lines, grid),
    with explicit x‐tick labels on every subplot and no legends.
    """
    # 1) Build titles dict
    phases = np.sort(y.index.get_level_values("Phase").unique())
    titles = {"Phase": phases}
    for col in y.columns:
        if col not in titles:
            titles[col] = np.sort(y[col].unique())

    # 2) Grid dims
    n_plots = len(titles)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7 * ncols, 7 * nrows),
        sharex=True
    )
    axes = axes.flatten()

    # 3) Precompute jitter
    jitter = (np.random.rand(len(closest_archetypes)) - 0.5) * 0.2

    # 4) Determine unique archetypes for x‐ticks
    unique_arch = np.sort(np.unique(closest_archetypes))

    # 5) For each label, scatter and set ticks
    for ax, (title, unique_vals) in zip(axes, titles.items()):
        # scatter points
        for val in unique_vals:
            try:
                mask = y.index.get_level_values(title) == val
            except KeyError:
                mask = y[title] == val

            idx = np.where(mask)[0]
            if idx.size:
                ax.scatter(
                    closest_archetypes[idx] + jitter[idx],
                    y.iloc[idx][title] if title in y.columns else pd.Series(val, index=idx),
                    alpha=0.7
                )

        # enforce x‐tick labels on every subplot
        ax.set_xticks(unique_arch)
        ax.set_xticklabels([str(int(a)) for a in unique_arch], rotation=0)
        ax.tick_params(labelbottom=True)

        # PCA‐style zero‐lines & grid
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axvline(-0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("Archetype")
        ax.set_ylabel(str(title))
        ax.set_title(f"{method_name} – {title}")
        ax.grid(True)

    # 6) Hide any unused axes
    for extra_ax in axes[len(titles):]:
        extra_ax.set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = FIGURE_DIR / f"{method_name.lower()}_subplots.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def run_aa_pipeline(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k_list: List[int] = list(range(2, 21)),
    select_k: Optional[int] = None
) -> List[float]:
    """
    Execute AA analysis: explained variance vs noc, archetypes.
    Returns list of explained variance per noc.
    """

    print("Running AA pipeline...")

    # 1) Preprocess data
    X_np = preprocess_for_aa(X)
    # 2) Evaluate explained variance metric
    evs = []
    for k in k_list:
        XC, varexp, _ = compute_aa(X_np, k)
        evs.append(varexp)
    # 3) Plot explained variance

    df = pd.DataFrame({
        'Number of archetypes': k_list,
        'Explained Variance': evs
    })
    # print(evs)
    # Determine chosen k
    max_ev = 1.0
    eps = 5e-3
    chosen = None
    for k_val, ev_val in zip(k_list, evs):
        if ev_val >= max_ev - eps:
            chosen = k_val
            break
    if chosen is None:
        raise ValueError("No suitable k found.")
    print(f"Chosen k: {chosen}")

    # Plot EV using our new helper
    plot_aa_explained_variance(
        k_list=k_list,
        evs=evs,
        chosen=chosen,
        save_path=FIGURE_DIR / 'aa_explained_variance.png'
    )
    # Print selected k
    print(f"Selected k = {chosen} with VE = {max_ev:.4f}")

    # 4) Fit final ICA and visualize mixing matrix
    XC_sel, varexp_sel, closest_archetype = compute_aa(X_np, chosen)
    plot_closest_archetypes(closest_archetypes=closest_archetype,
                            feature_names=['Frustrated', 'upset', 'hostile', 'alert', 'ashamed',
                                           'inspired', 'nervous', 'attentive', 'afraid', 'active', 'determined'],
                            y=y)
    plot_archetypes(XC=XC_sel, feature_names=list(X.columns))

    plot_phases_archetypes(closest_archetypes=closest_archetype,
                           y=y)
    
    print("Plotting AA subplots for Phase, Puzzler, and emotions...")
    plot_aa_subplots(closest_archetype, y)


    return evs
