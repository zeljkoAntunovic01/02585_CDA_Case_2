import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, SparsePCA
import pandas as pd
import seaborn as sns
import math
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


def plot_pca(X: np.ndarray, y: np.ndarray, pca: PCA) -> None:
    """
    Plot the PCA results, coloring points by 'Phase'.

    Parameters
    ----------
    X : np.ndarray
        Input data for PCA.
    y : np.ndarray
        Labels/responses matrix.
    pca : PCA
        Fitted PCA object.
    """
    # Transform the data
    X_pca = pca.transform(X)
    data = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)

    # Extract unique values for different categories
    phases = np.sort(data.index.get_level_values("Phase").unique())
    titles = {"Phase": phases}
    for column in y.columns:
        if column not in titles:
            titles[column] = np.sort(data[column].unique())

    # Determine grid dimensions using a near-square layout
    n_plots = len(titles)
    ncols = math.ceil(math.sqrt(n_plots))
    nrows = math.ceil(n_plots / ncols)

    # Create subplots with each subplot sized ~7x7 inches
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 7 * nrows))
    # Flatten the axes array for easy iteration and in case some subplots remain unused
    axes = axes.flatten()

    # Iterate over each title and plot
    for ax, (title, unique_values) in zip(axes, titles.items()):
        for value in unique_values:
            try:
                mask = data.index.get_level_values(title) == value
            except KeyError:
                mask = data[title] == value
            indices = np.where(mask)[0]
            ax.scatter(
                X_pca[indices, 0],
                X_pca[indices, 1],
                label=value,
                alpha=0.7,
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle="--", linewidth=1)
        ax.legend()
        ax.grid(True)
        if isinstance(pca, SparsePCA):
            ax.set_title(f"Sparse PCA - {title}")
        else:
            ax.set_title(f"PCA - {title}")

    # Turn off any extra axes
    for ax in axes[len(titles):]:
        ax.set_visible(False)

    plt.tight_layout()
    if isinstance(pca, SparsePCA):
        plt.savefig(FIGURE_DIR / "sparse_pca" / "pca_subplots.png")
    else:
        plt.savefig(FIGURE_DIR / "pca" / "pca_subplots.png")



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
def print_top_features(pca: PCA, features: list, n_features: int = 3) -> None:
    """
    Print the top features for each PCA component.

    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    n_features : int, optional
        Number of top features to print for each component, by default 10.
    """
    loadings = pca.components_
    for i in range(loadings.shape[0]):
        top_indices = np.argsort(loadings[i])[-n_features:]
        print(f"Top {n_features} features for component {i + 1}:")
        print(", ".join([features[idx] for idx in top_indices]))
        print()
        

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

    

    # Fit PCA
    pca.fit(X_preprocessed)

    if isinstance(pca, SparsePCA):
        print("Sparse PCA")

    else:
        print("PCA")
        # Plot explained variance
        plot_explained_variance(pca)

        # Scree plot
        scree_plot(X_preprocessed, pca)

    # Correlation matrix
    pca_correlation_matrix(X_preprocessed, pca)

    # Plot PCA
    plot_pca(X_preprocessed, y_preprocessed, pca)

    # # Plot PCA loadings
    features = X.columns[::8]
    plot_features_against_loadings(X_preprocessed, features, pca)

    # Print top features
    print_top_features(pca, X.columns, n_features=3)


if __name__ == "__main__":
    # Example usage
    pca = PCA(n_components=5)
    pca_pipeline(pca)
    sparse_pca = SparsePCA(n_components=5, alpha=0.1, ridge_alpha=0.1)
    pca_pipeline(sparse_pca)
