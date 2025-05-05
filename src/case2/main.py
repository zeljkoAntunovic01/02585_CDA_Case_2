#!/usr/bin/env python3
from data_loader import run_data_loading, preprocess_data_for_subspace_methods
from decomposition.nmf import run_nmf_pipeline
from decomposition.ica import run_ica_pipeline
from decomposition.aa import run_aa_pipeline
from decomposition.sc import run_sc_pipeline
from subspace_methods.som import som_pipeline
from subspace_methods.pca import run_pca_pipeline

def main():
    # Loading the data
    hr_df, X, y = run_data_loading()
   

    # Decomposition methods
    run_nmf_pipeline(X, y)
    run_ica_pipeline(X, y)
    # run_aa_pipeline(X, y)
    # run_sc_pipeline(X, y)
    
    # Subspace methods
    X_processed, y_processed = preprocess_data_for_subspace_methods(X, y)
    som_pipeline(hr_df, X_processed, y_processed)
    run_pca_pipeline(X, X_processed, y_processed)


if __name__ == "__main__":
    main()