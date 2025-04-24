# src/data_loader.py

import pandas as pd
from pathlib import Path
from typing import Dict, Any


SignalData = Dict[
    str,  # dataset name, e.g. "D1_1"
    Dict[
        str,  # participant id, e.g. "ID_1"
        Dict[
            str,  # round, e.g. "round_1"
            Dict[
                str,  # phase, e.g. "phase1"
                Dict[
                    str,  # signal type, e.g. "BVP", "EDA", "HR", "TEMP", or "response"
                    pd.DataFrame
                ]
            ]
        ]
    ]
]


MetaData = Dict[
    str,  # dataset name, e.g. "D1_1"
    Dict[str, pd.DataFrame]  # keys "scores" and "team_info"
]


def load_phase_signals(
    phase_dir: Path
) -> Dict[str, pd.DataFrame]:
    """
    Load all five CSVs in one phase directory.

    Parameters
    ----------
    phase_dir : Path
        Path to the folder containing BVP.csv, EDA.csv, HR.csv, TEMP.csv, response.csv.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping {"BVP": ..., "EDA": ..., "HR": ..., "TEMP": ..., "response": ...}.
    """
    signals = {}
    for fname in ["BVP", "EDA", "HR", "TEMP", "response"]:
        fpath = phase_dir / f"{fname}.csv"
        if fpath.exists():
            signals[fname] = pd.read_csv(fpath)
    return signals


def load_dataset(
    dataset_dir: Path
) -> tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """
    Load one D1_* dataset: all participant/round/phase signals plus its metadata.

    Parameters
    ----------
    dataset_dir : Path
        e.g. /path/to/data/D1_1

    Returns
    -------
    data_hierarchy : nested dict
        data_hierarchy[participant][round][phase][signal] → DataFrame
    meta         : dict
        {"scores": DataFrame, "team_info": DataFrame}
    """
    # 1) dataset‐level files
    meta = {}
    # find scores file (scores_D?_2.csv)
    scores = next(dataset_dir.glob("scores_*.csv"), None)
    if scores:
        meta["scores"] = pd.read_csv(scores)
    # team_info.csv
    team_info = dataset_dir / "team_info.csv"
    if team_info.exists():
        meta["team_info"] = pd.read_csv(team_info)

    # 2) hierarchical signals
    data_hierarchy: Dict[str, Any] = {}
    for part_dir in sorted(dataset_dir.glob("ID_*")):
        pid = part_dir.name
        data_hierarchy[pid] = {}
        # each round_1 … round_4
        for round_dir in sorted(part_dir.glob("round_*")):
            rnd = round_dir.name
            data_hierarchy[pid][rnd] = {}
            # each phase1 … phase3
            for phase_dir in sorted(round_dir.glob("phase*")):
                ph = phase_dir.name
                data_hierarchy[pid][rnd][ph] = load_phase_signals(phase_dir)

    return data_hierarchy, meta


def load_all_datasets(
    root_dir: Path
) -> tuple[SignalData, MetaData]:
    """
    Load all D1_1 ... D1_6 under a root folder.

    Parameters
    ----------
    root_dir : Path
        e.g. /path/to/data

    Returns
    -------
    all_data : nested dict for signals
    all_meta : dict of metadata DataFrames
    """
    all_data: SignalData = {}
    all_meta: MetaData = {}

    for ds_dir in sorted(root_dir.glob("D1_*")):
        ds_name = ds_dir.name
        print(f"Loading dataset {ds_name}...")
        data_hierarchy, meta = load_dataset(ds_dir)
        all_data[ds_name] = data_hierarchy
        all_meta[ds_name] = meta

    return all_data, all_meta


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent / "data" / "raw" / "dataset"
    print("Loading datasets from:", base)
    signals, metadata = load_all_datasets(base)

    # quick sanity checks
    print("Datasets loaded:", list(signals.keys()))
    for ds in signals:
        num_parts = len(signals[ds])
        print(f"  {ds}: {num_parts} participants")
        print("    Meta keys:", list(metadata[ds].keys()))
