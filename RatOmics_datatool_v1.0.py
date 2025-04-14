#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:12:02 2025

@author: galen2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage

def run_dimension_reduction_analysis(csv_path):
    """
    Perform PCA, PLS-DA, and hierarchical clustering on metabolomics data.
    The CSV is expected to contain:
      - 'Sample_ID', 'Group', 'Time' columns (as metadata).
      - All other columns are metabolites (already scaled/normalized).
    """

    # -----------------------------
    # 1) Load Data
    # -----------------------------
    df = pd.read_csv(csv_path)

    # Separate metadata from data
    meta_cols = ['Sample_ID', 'Group', 'Time']
    # Assumes all other columns are metabolite concentrations
    X = df.drop(columns=meta_cols)
    # Group (for color-coding, PLS-DA)
    group_labels = df['Group'].astype(str)

    # If you want to keep Time numeric or factor, you can do so here
    time_labels = df['Time']

    # -----------------------------
    # 2) PCA
    # -----------------------------
    pca = PCA(n_components=2)
    pca_scores = pca.fit_transform(X.values)
    
    print("PCA - Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)

    # Plot PCA Score Plot
    plt.figure()
    unique_groups = group_labels.unique()
    for g in unique_groups:
        mask = (group_labels == g)
        plt.scatter(pca_scores[mask, 0],
                    pca_scores[mask, 1],
                    label=g)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Score Plot (by Group)')
    plt.legend()
    plt.show()

    # -----------------------------
    # 3) PLS-DA
    # -----------------------------
    # Encode group labels as numeric for PLS
    le = LabelEncoder()
    y_encoded = le.fit_transform(group_labels)

    # Build a PLS model with 2 components
    pls = PLSRegression(n_components=2)
    pls.fit(X.values, y_encoded)

    # Extract the scores for the X-space
    pls_scores = pls.x_scores_

    # Plot PLS-DA Score Plot
    plt.figure()
    for g in unique_groups:
        mask = (group_labels == g)
        plt.scatter(pls_scores[mask, 0],
                    pls_scores[mask, 1],
                    label=g)
    plt.xlabel('PLS1')
    plt.ylabel('PLS2')
    plt.title('PLS-DA Score Plot (by Group)')
    plt.legend()
    plt.show()

    # -----------------------------
    # 4) Hierarchical Clustering
    # -----------------------------
    # We use the metabolite matrix X directly for clustering.
    # 'method="ward"' is common for omics data.
    linkage_matrix = linkage(X.values, method='ward')

    # Plot the dendrogram
    plt.figure()
    dendrogram(linkage_matrix,
               labels=df['Sample_ID'].values,
               leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

    # -----------------------------
    # End of Analysis
    # -----------------------------
    print("Analysis complete.")

if __name__ == "__main__":
    # Update the path to your CSV as needed:
    csv_file_path = "RatOmics_FINAL_processed_data_TSFormat.csv"
    run_dimension_reduction_analysis(csv_file_path)

