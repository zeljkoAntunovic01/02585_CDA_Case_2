import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, SparsePCA
import pandas as pd
import seaborn as sns

from data_loader import load_data

sns.set(style="darkgrid")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
FIGURE_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"


def preprocess_pca_data(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Preprocess the data for PCA by centering and imputing missing values.

    Parameters
    ----------
    X : np.ndarray
        Input data for PCA.
    y : np.ndarray
        Labels/responses matrix.

    Returns
    -------
    X_preprocessed : np.ndarray
        Preprocessed input data for PCA.
    y_preprocessed : np.ndarray
        Preprocessed labels/responses matrix.
    """

    X_centered = X - np.mean(X, axis=0)

    # Impute missing values with the median of each column
    X_imputed = X_centered.fillna(X_centered.median())

    S = np.linalg.norm(X_imputed, axis=0)
    X_preprocessed = X_imputed / S

    cohort_mapping = {"D1_1": 1, "D1_2": 2, "D1_3": 3, "D1_4": 4, "D1_5": 5, "D1_6": 6}

    y = pd.DataFrame(y)
    # Map cohort names to numbers
    y_mapped = y.copy()
    y_mapped["Cohort"] = y_mapped["Cohort"].map(cohort_mapping)

    y_preprocessed = y_mapped.fillna(-1)

    return X_preprocessed, y_preprocessed


def plot_explained_variance(pca: PCA) -> None:
    """
    Plot the explained variance ratio.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    """
    n_components = pca.n_components_

    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, n_components + 1), pca.explained_variance_ratio_.cumsum(), marker="o"
    )
    plt.ylim([0, 1])
    plt.title("Explained Variance Ratio by PCA Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.xticks(range(1, n_components + 1))
    plt.savefig(
        FIGURE_DIR / "pca" /"explained_variance_ratio.png")


def scree_plot(X: np.ndarray, pca: PCA) -> None:
    """
    Compare the cumulative explained variance from PCA performed on the original data
    with that from PCA performed on randomized data.

    Parameters
    ----------
    X : np.ndarray
        Input data for PCA.
    n_components : int, optional
        Number of PCA components to compute, by default 5.
    """
    # PCA on original data
    n_components = pca.n_components_

    # Create a randomized version of the data by permuting each column
    X_random = np.copy(X)
    for i in range(X_random.shape[1]):
        X_random[:, i] = np.random.permutation(X_random[:, i])

    # PCA on randomized data
    pca_rand = PCA(n_components=n_components)
    pca_rand.fit(X_random)

    # Plot comparison
    plt.figure(figsize=(8, 6))
    plt.plot(
        range(1, n_components + 1),
        pca.explained_variance_,
        marker="o",
        label="Original Data",
    )
    plt.plot(
        range(1, n_components + 1),
        pca_rand.explained_variance_,
        marker="o",
        label="Randomized Data",
    )
    plt.title("Scree plot of PCA Components")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue")
    plt.xticks(range(1, n_components + 1))
    plt.legend()
    plt.savefig(
        FIGURE_DIR / "pca" /"scree_plot.png",
    )


def pca_correlation_matrix(X: np.ndarray, pca: PCA) -> None:
    """
    Plot the correlation matrix of the PCA scores and loadings.

    Parameters
    ----------
    X_pca : np.ndarray
        PCA-transformed data.
    pca : PCA
        Fitted PCA object.
    """
    # Get PCA scores and loadings
    scores = pca.transform(X)
    loadings = pca.components_

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].matshow(np.abs(np.corrcoef(scores, rowvar=False)), cmap="viridis")
    ax[0].set_title("Correlation Matrix of PCA Scores")

    ax[1].matshow(np.abs(np.corrcoef(loadings.T, rowvar=False)), cmap="viridis")
    ax[1].set_title("Correlation Matrix of PCA Loadings")

    if isinstance(pca, SparsePCA):
        plt.suptitle("Sparse PCA Correlation Matrices")
        plt.savefig(
            FIGURE_DIR / "sparse_pca" /"sparse_pca_correlation_matrix.png",
        )
    else:
        plt.suptitle("PCA Correlation Matrices")
        plt.savefig(
            FIGURE_DIR / "pca" / "pca_correlation_matrix.png"
        )


def plot_pca(X: np.ndarray, pca: PCA) -> None:
    """
    Plot the PCA results, coloring points by 'Phase'.

    Parameters
    ----------
    X : np.ndarray
        Input data for PCA.
    pca : PCA
        Fitted PCA object.
    """
    # Transform the data
    X_pca = pca.transform(X)

    # Extract 'Phase' from the index.
    phases = X.index.get_level_values("Phase")
    unique_phases = phases.unique()
    colors = ["red", "blue", "green"]  # Define colors for each phase

    # Create a color mapping for each phase
    color_mapping = {
        phase: colors[i % len(colors)] for i, phase in enumerate(unique_phases)
    }

    plt.figure(figsize=(8, 6))
    for phase in unique_phases:
        mask = phases == phase
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=phase,
            alpha=0.5,
            color=color_mapping[phase],
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Phase")

    if isinstance(pca, SparsePCA):
        plt.title("Sparse PCA of HR Data by Phase")
        plt.savefig(
            FIGURE_DIR / "sparse_pca" /"sparse_pca_plot.png",
        )
    else:
        plt.title("PCA of HR Data by Phase")
        plt.savefig(
            FIGURE_DIR / "pca" /"pca_plot.png",
        )


def plot_features_against_loadings(data: np.ndarray, features: list, pca: PCA) -> None:
    """
    Plot the features against the loadings of the PCA components.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    """

    loadings = pca.components_.T
    loadings = loadings[:, :2]  # Take only the first two components
    feature_names = data.columns

    # find index for features
    feature_indices = [feature_names.get_loc(feature) for feature in features]
    feature_names = [feature_names[i] for i in feature_indices]


    plt.figure(figsize=(8, 6))
    plt.axhline(0, color='grey', linestyle='--', linewidth=1)
    plt.axvline(0, color='grey', linestyle='--', linewidth=1)

    # Plot the loadings as vectors
    for idx, varname in zip(feature_indices, feature_names):
        plt.plot(loadings[idx, 0], loadings[idx, 1], 'o', color='r')
        plt.text(loadings[idx, 0], loadings[idx, 1] + 0.025, varname, color='b', ha='center', va='center')

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA Loadings Plot")
    plt.grid(True)
    plt.tight_layout()
    if isinstance(pca, SparsePCA):
        plt.savefig(
            FIGURE_DIR / "sparse_pca" /"sparse_pca_loadings.png",
        )
    else:
        plt.savefig(
            FIGURE_DIR / "pca" /"pca_loadings.png",
        )
    

def pca_pipeline(pca: PCA) -> None:
    """
    Run the PCA pipeline: load data, preprocess, fit PCA, and plot results.

    Parameters
    ----------
    pca : SparsePCA or PCA
        PCA object to use for the analysis.
    """
    # Load data
    _, X, y = load_data()

    # Preprocess data
    X_preprocessed, y_preprocessed = preprocess_pca_data(X, y)

    hr_preprocessed = pd.concat([X_preprocessed, y_preprocessed], axis=1)

    # Fit PCA
    pca.fit(hr_preprocessed)

    if isinstance(pca, SparsePCA):
        print("Sparse PCA")

    else:
        print("PCA")

        # Plot explained variance
        plot_explained_variance(pca)

        # Scree plot
        scree_plot(hr_preprocessed, pca)

    # Correlation matrix
    pca_correlation_matrix(hr_preprocessed, pca)

    # Plot PCA
    plot_pca(hr_preprocessed, pca)

    # Plot PCA loadings
    features = y.columns.drop(["Cohort", "Puzzler"])
    plot_features_against_loadings(hr_preprocessed, features, pca)


if __name__ == "__main__":
    # Example usage
    pca = PCA(n_components=10)
    pca_pipeline(pca)
    sparse_pca = SparsePCA(n_components=10)
    pca_pipeline(sparse_pca)
