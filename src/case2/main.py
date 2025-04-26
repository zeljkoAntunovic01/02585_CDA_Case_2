#!/usr/bin/env python3
from data_loader import run_data_loading
from decomposition.nmf import run_nmf_pipeline

if __name__ == "__main__":
    # Loading the data
    hr_df, X, y = run_data_loading()

    # Decomposition methods 
    run_nmf_pipeline(X, y)