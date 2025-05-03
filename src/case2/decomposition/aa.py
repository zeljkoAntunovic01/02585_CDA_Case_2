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
import seaborn as sns

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
    # Add closest archetype column
    df_plot = y.copy()
    df_plot["Archetype"] = closest_archetypes

    # Reshape to long format
    df_long = df_plot.melt(id_vars="Archetype", value_vars=feature_names,
                           var_name="Emotion", value_name="Score")

    # Convert archetype to string for nicer x-axis ticks
    df_long["Archetype"] = df_long["Archetype"].astype(str)

    # Create FacetGrid
    g = sns.FacetGrid(df_long, col="Emotion",
                      col_wrap=3, height=4, sharey=False)
    g.map(sns.stripplot, "Archetype", "Score",
          jitter=True, alpha=0.7)

    # Improve layout
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45)
    g.fig.suptitle("Emotion Scores by Closest Archetype", y=1.02)
    g.tight_layout()

    # Show or save
    # plt.show()
    g.savefig(FIGURE_DIR / "emotion_scores_by_archetype_strip.png", dpi=300)


def plot_phases_archetypes(
    closest_archetypes: np.ndarray,
    y: pd.DataFrame,
) -> None:
    # Add closest archetype column
    df_plot = y.copy()
    df_plot["Archetype"] = closest_archetypes

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=df_plot, x="Phase", y="Archetype",
                  jitter=True, alpha=0.7, ax=ax)

    # Set title and layout
    ax.set_title("Phase by Closest Archetype", y=1.02)
    fig.tight_layout()

    # Save
    fig.savefig(FIGURE_DIR / "phase_by_archetype.png", dpi=300)
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
    k_list: List[int] = list(range(2, 16)),
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
    print(evs)
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

    # Plot with seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x='Number of archetypes',
                 y='Explained Variance', marker='o', label='Explained Variance')

    # Add vertical line for selected k
    plt.axvline(chosen, color='grey', linestyle='--',
                label=f'Selected number of archetypes = {chosen}')

    # Decorations
    plt.xlabel('Number of Archetypes')
    plt.ylabel('Explained Variance')
    plt.title('AA Explained Variance vs Number of Archetypes')
    plt.legend()
    plt.tight_layout()

    # Save and close
    save_path = FIGURE_DIR / 'aa_explained_variance.png'
    plt.savefig(save_path, dpi=300)
    plt.close()

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

    return evs
