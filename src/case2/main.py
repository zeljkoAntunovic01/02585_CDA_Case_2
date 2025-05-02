#!/usr/bin/env python3
from data_loader import run_data_loading
from decomposition.nmf import run_nmf_pipeline
from decomposition.ica import run_ica_pipeline
from decomposition.aa import run_aa_pipeline
from decomposition.sc import run_sc_pipeline


def main():
    # Loading the data
    hr_df, X, y = run_data_loading()

    # Decomposition methods
    # run_nmf_pipeline(X, y)
    # run_ica_pipeline(X, y)
    run_aa_pipeline(X, y)
    # run_sc_pipeline(X, y)


if __name__ == "__main__":
    main()
