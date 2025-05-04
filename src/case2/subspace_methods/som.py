import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from pathlib import Path
from collections import defaultdict
import seaborn as sns
from matplotlib.cm import get_cmap
import os

sns.set_theme(style="darkgrid")
FIG_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "figures" / "som"

def preprocess_som_data(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the data for SOM by centering, imputing, and normalizing.

    Parameters
    ----------
    X : pd.DataFrame
        Input data for SOM.
    y : pd.DataFrame
        Labels/responses matrix.

    Returns
    -------
    X_preprocessed : pd.DataFrame
        Preprocessed input data for SOM.
    y_preprocessed : pd.DataFrame
        Preprocessed labels/responses matrix.
    """
    X_centered = X - X.mean(axis=0)
    X_imputed = X_centered.fillna(X_centered.median())
    S = np.linalg.norm(X_imputed, axis=0)
    X_preprocessed = (X_imputed / S)

    cohort_mapping = {"D1_1": 1, "D1_2": 2, "D1_3": 3, "D1_4": 4, "D1_5": 5, "D1_6": 6}
    y_mapped = y.copy()
    if "Cohort" in y.columns:
        y_mapped["Cohort"] = y_mapped["Cohort"].map(cohort_mapping)

    y_preprocessed = y_mapped.fillna(-1)
    return X_preprocessed, y_preprocessed

def get_phase_specific_data(X_df: pd.DataFrame, y_df: pd.DataFrame, full_df: pd.DataFrame, phase_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filters feature and label data for a given phase.

    Parameters
    ----------
    X_df : pd.DataFrame
        Feature matrix.
    y_df : pd.DataFrame
        Labels/responses matrix.
    full_df : pd.DataFrame
        Full DataFrame containing phase information.
    phase_name : str
        One of 'phase1', 'phase2', 'phase3'
    
    Returns
    -------
    X_phase : pd.DataFrame
    y_phase : pd.DataFrame
    """
    df_phase = full_df[full_df.index.get_level_values('Phase') == phase_name]
    X_phase = X_df.loc[df_phase.index]
    y_phase = y_df.loc[df_phase.index]
    return X_phase, y_phase

def train_som(X: np.ndarray, x_dim=10, y_dim=10, sigma=1.0, learning_rate=0.5, num_iter=5000) -> MiniSom:
    """
    Train a Self-Organizing Map (SOM) on the input data.

    Parameters
    ----------
    X : np.ndarray
        Input data for SOM.
    x_dim : int
        Number of neurons in the x dimension.
    y_dim : int
        Number of neurons in the y dimension.
    sigma : float
        Width of the neighborhood function.
    learning_rate : float
        Learning rate for the SOM.
    num_iter : int
        Number of iterations for training.
    -----------
    Returns
    som : MiniSom
        Trained SOM object.
    """
    som = MiniSom(x=x_dim, y=y_dim, input_len=X.shape[1], sigma=sigma,
                  learning_rate=learning_rate, neighborhood_function='gaussian', random_seed=42)
    som.pca_weights_init(X)
    print("Training SOM...")
    som.train(X, num_iter, verbose=True)
    return som

def plot_u_matrix(som: MiniSom):
    """
    Plot the U-Matrix of the trained SOM.
    The U-Matrix visualizes the distances between the neurons in the SOM grid.
    It helps to identify clusters and the topology of the data.
    """
    plt.figure(figsize=(8, 8))
    plt.title("U-Matrix")
    u_matrix = som.distance_map()
    plt.imshow(u_matrix.T, cmap='bone_r', origin='lower')
    plt.colorbar(label='Distance')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "som_u_matrix.png")
    plt.close()

def plot_label_heatmaps(som: MiniSom, X: np.ndarray, y: pd.DataFrame, emotion: str):
    """
    Plot a heatmap of the average label values for each neuron in the SOM.
    Each neuron is represented by a grid cell, and the color intensity indicates the average value of the label for that neuron.
    """
    assert emotion in y.columns, f"{emotion} not found in label columns."

    label_map = defaultdict(list)
    for i, x in enumerate(X):
        val = int(y.iloc[i][emotion])
        if val in [-1, 0]:  # skip invalid labels
            continue
        winner = som.winner(x)
        label_map[winner].append(val)

    x_dim, y_dim, _ = som.get_weights().shape
    averaged_map = np.zeros((x_dim, y_dim))

    for x in range(x_dim):
        for y_ in range(y_dim):
            values = label_map.get((x, y_), [])
            if values:
                averaged_map[x, y_] = np.mean(values)

    plt.figure(figsize=(8, 8))
    plt.title(f"Heatmap for Emotion: {emotion}")
    plt.imshow(averaged_map.T, origin='lower', cmap='coolwarm')
    plt.colorbar(label=emotion)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_heatmaps" /f"som_label_heatmap_{emotion}.png")
    plt.close()

def plot_emotion_scatter(som: MiniSom, X: np.ndarray, y: pd.Series, emotion: str):
    """
    Plot emotion values on SOM as scattered text at BMU locations.
    Each text is colored based on the emotion value.
    The color intensity indicates the value of the emotion for that neuron.
    """
    plt.figure(figsize=(10, 10))

    # Filter out invalid emotion values
    y = y.astype(float)  # Optional: ensure consistency
    mask = ~((y == 0.0) | (y == -1.0))
    X_filtered = X[mask.values]
    y_filtered = y[mask]

    unique_vals = sorted(y.unique())
    val_min, val_max = min(unique_vals), max(unique_vals)
    cmap = get_cmap("tab10" if val_max <= 10 else "tab20")

    # Track added labels to prevent duplicate legends
    added_labels = set()

    for x_vec, val in zip(X_filtered, y_filtered):
        bmu = som.winner(x_vec)
        jitter_x = bmu[0] + 0.5 + 0.6 * np.random.rand() - 0.3
        jitter_y = bmu[1] + 0.5 + 0.6 * np.random.rand() - 0.3
        color = cmap((val - val_min) / (val_max - val_min))
        label = str(val) if val not in added_labels else None
        plt.plot(jitter_x, jitter_y, 'o', color=color, alpha=0.7, label=label)
        added_labels.add(val)

    plt.title(f"SOM Emotion Scatter: {emotion}")
    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
    plt.gca().invert_yaxis()

    # Sort legend numerically
    handles, labels = plt.gca().get_legend_handles_labels()
    label_handle_pairs = sorted(zip(labels, handles), key=lambda x: float(x[0]))
    sorted_labels, sorted_handles = zip(*label_handle_pairs)
    plt.legend(sorted_handles, sorted_labels, title="Emotion Scale")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_scatter" / f"som_label_scatter_{emotion}.png")
    plt.close()

def plot_bmu_scatter_by_phase(som: MiniSom, X_df: pd.DataFrame, y_df: pd.DataFrame, df: pd.DataFrame):
    """
    Scatterplot of BMU locations colored by Phase.
    Each phase is represented by a different color.
    The scatterplot shows the distribution of BMUs for each phase on the SOM grid.
    """
    palette = sns.color_palette("colorblind", 3)  # 3 distinct, colorblind-friendly colors
    phase_labels = ['phase1', 'phase2', 'phase3']
    phase_colors = dict(zip(phase_labels, palette))

    plt.figure(figsize=(10, 10))
    plotted_labels = set()

    for phase in phase_labels:
        color = phase_colors[phase]
        X_phase_df, _ = get_phase_specific_data(X_df, y_df, df, phase)
        X_phase_scaled, _ = preprocess_som_data(X_phase_df, y_df.loc[X_phase_df.index])
        xs, ys = [], []
        for x in X_phase_scaled.to_numpy():
            bmu = som.winner(x)
            jitter_x = bmu[0] + 0.5 + 0.6 * np.random.rand() - 0.3
            jitter_y = bmu[1] + 0.5 + 0.6 * np.random.rand() - 0.3
            xs.append(jitter_x)
            ys.append(jitter_y)
        plt.scatter(xs, ys, color=color, alpha=0.7, label=phase)

    plt.title("BMU Locations Colored by Phase")
    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
    plt.gca().invert_yaxis()

    # Sort legend alphabetically by phase name
    handles, labels = plt.gca().get_legend_handles_labels()
    label_handle_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
    sorted_labels, sorted_handles = zip(*label_handle_pairs)
    plt.legend(sorted_handles, sorted_labels, title="Phase")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "som_bmu_scatter_by_phase.png")
    plt.close()

def plot_phase_hitmap(som: MiniSom, X_phase: np.ndarray, phase_label: str):
    """
    Plots a hit map of BMU activations for a given phase.
    Each cell in the heatmap represents the number of samples that activated that neuron.
    The color intensity indicates the number of samples for that neuron.
    """
    activation_map = np.zeros(som.get_weights().shape[:2])
    for x in X_phase:
        bmu = som.winner(x)
        activation_map[bmu] += 1

    plt.figure(figsize=(8, 8))
    plt.title(f"Hit Map for {phase_label}")
    plt.imshow(activation_map.T, cmap='Reds', origin='lower')
    plt.colorbar(label="Number of Samples")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "phase_hitmaps" / f"som_hit_map_{phase_label}.png")
    plt.close()

def plot_phase_trajectories(som: MiniSom, X_scaled: pd.DataFrame, df: pd.DataFrame, 
                            max_individuals: int = None, random_seed: int = 42):
    """
    Plots BMU trajectories (Phase 1 → 2 → 3) for each participant and round.
    Each line represents the path taken by a participant through the SOM grid across phases.
    """
    unique_ids = df.index.get_level_values("Individual").unique()
    if max_individuals is not None and max_individuals < len(unique_ids):
        np.random.seed(random_seed)
        unique_ids = np.random.choice(unique_ids, size=max_individuals, replace=False)

    plt.figure(figsize=(10, 10))

    arrow_colors = ['red', 'green']  # P1→P2: red, P2→P3: green

    for person_id in unique_ids:
        person_rounds = df.loc[person_id].index.get_level_values("Round").unique()
        for round_id in person_rounds:
            bmu_sequence = []
            try:
                for phase in ['phase1', 'phase2', 'phase3']:
                    x_vec = X_scaled.loc[(person_id, round_id, phase)]
                    bmu_sequence.append(som.winner(x_vec.to_numpy()))
                
                for i in range(2):  # phase1→2 and phase2→3
                    x0, y0 = bmu_sequence[i]
                    x1, y1 = bmu_sequence[i+1]
                    dx, dy = x1 - x0, y1 - y0
                    plt.arrow(x0 + 0.5, y0 + 0.5, dx, dy,
                              head_width=0.3, head_length=0.3, fc=arrow_colors[i], ec=arrow_colors[i],
                              length_includes_head=True, alpha=0.6)
            except KeyError:
                continue

    plt.title("SOM Phase Trajectories with Direction (P1 → P2 → P3)")
    plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "som_phase_trajectories.png")
    plt.close()

def som_pipeline(X_df, y_df):
    """
    Main pipeline for SOM analysis.
    1. Load and preprocess data.
    2. Train SOM.
    3. Generate visualizations:
         - U-Matrix
         - Label heatmaps and scatter maps for each emotion label
         - Phase scatter: Where do phases land on the SOM?
         - Phase-wise hit maps
         - Plot trajectories for individuals
     """
    # Preprocess data
    X_scaled, y_processed = preprocess_som_data(X_df, y_df)

    # Train SOM
    som = train_som(X_scaled.to_numpy(), x_dim=12, y_dim=12, num_iter=5000)

    # Visualizations
    # Create directories for saving figures
    os.makedirs(FIG_DIR / "label_heatmaps", exist_ok=True)
    os.makedirs(FIG_DIR / "label_scatter", exist_ok=True)
    os.makedirs(FIG_DIR / "phase_hitmaps", exist_ok=True)
    
    # U-Matrix
    #plot_u_matrix(som)

    # Visualize label heatmaps and scatter maps for each emotion label
    emotions = [
        'Frustrated', 'alert', 'ashamed', 'inspired', 'nervous',
        'attentive', 'afraid', 'active', 'determined'
    ]
    for emotion in emotions:
        if emotion in y_processed.columns:
            plot_label_heatmaps(som, X_scaled.to_numpy(), y_processed, emotion)
            plot_emotion_scatter(som, X_scaled.to_numpy(), y_processed[emotion], emotion)

    """  # Phase scatter: Where do phases land on the SOM?
    plot_bmu_scatter_by_phase(som, X_df, y_df, df)

    # Phase-wise hit maps
    for phase in ['phase1', 'phase2', 'phase3']:
        X_phase_df, _ = get_phase_specific_data(X_df, y_df, df, phase)
        X_phase_scaled, _ = preprocess_som_data(X_phase_df, y_df.loc[X_phase_df.index])
        plot_phase_hitmap(som, X_phase_scaled.to_numpy(), phase)

    # Plot trajectories for individuals
    plot_phase_trajectories(som, X_scaled, df, max_individuals=3)
    """