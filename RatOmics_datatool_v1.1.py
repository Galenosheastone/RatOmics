#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 17:12:02 2025

@author: galen2
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from scipy.cluster.hierarchy import dendrogram, linkage

def run_dimension_reduction_analysis(csv_path, output_folder="output_plots"):
    """
    Perform PCA, PLS-DA (with cross-validated validation metrics), and hierarchical clustering
    on metabolomics data. The CSV is expected to contain:
      - 'Sample_ID', 'Group', 'Time' columns (as metadata).
      - All other columns are metabolite concentrations (pre-processed, scaled, normalized).
    Saves the plots in the output_folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # 1) Load Data
    # -----------------------------
    df = pd.read_csv(csv_path)
    meta_cols = ['Sample_ID', 'Group', 'Time']
    X = df.drop(columns=meta_cols)
    
    # For plotting colors and PLS-DA we extract Group and Time
    group_labels = df['Group'].astype(str)
    time_labels = df['Time']  # This remains available for additional analysis if needed

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
    pca_plot_path = os.path.join(output_folder, "PCA_Score_Plot.png")
    plt.savefig(pca_plot_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    # -----------------------------
    # 3) PLS-DA
    # -----------------------------
    # Encode group labels numerically for PLS
    le = LabelEncoder()
    y_encoded = le.fit_transform(group_labels)
    
    # Build a PLS model with 2 components
    pls = PLSRegression(n_components=2)
    pls.fit(X.values, y_encoded)

    # Extract X-scores for plotting
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
    pls_plot_path = os.path.join(output_folder, "PLSDA_Score_Plot.png")
    plt.savefig(pls_plot_path, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 4) PLS-DA Model Validation
    # -----------------------------
    # Use cross_val_predict to obtain predictions from the PLS model
    y_pred_cv = cross_val_predict(pls, X.values, y_encoded, cv=5)
    
    # Since predictions are continuous, round them to the nearest valid class integer.
    y_pred_round = np.clip(np.rint(y_pred_cv), y_encoded.min(), y_encoded.max()).astype(int)
    
    # Calculate classification accuracy and R^2 (explained variance) for regression-like predictions.
    accuracy = accuracy_score(y_encoded, y_pred_round)
    r2 = r2_score(y_encoded, y_pred_cv)
    cm = confusion_matrix(y_encoded, y_pred_round)
    
    print("PLS-DA Cross-Validated Accuracy:", accuracy)
    print("PLS-DA Cross-Validated R^2 Score:", r2)
    print("PLS-DA Confusion Matrix:\n", cm)

    # Save the validation metrics into a text file.
    metrics_path = os.path.join(output_folder, "PLSDA_Validation_Metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("PLS-DA Cross-Validated Metrics\n")
        f.write("------------------------------\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"R^2 Score: {r2:.3f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    
    # -----------------------------
    # 5) Hierarchical Clustering
    # -----------------------------
    # Compute the linkage matrix using Ward's method on the metabolites data matrix.
    linkage_matrix = linkage(X.values, method='ward')
    
    # Plot and save the dendrogram.
    plt.figure()
    dendrogram(linkage_matrix,
               labels=df['Sample_ID'].values,
               leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    dendro_plot_path = os.path.join(output_folder, "Dendrogram.png")
    plt.savefig(dendro_plot_path, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # End of Analysis
    # -----------------------------
    print("Analysis complete. Plots and metrics saved in:", output_folder)

if __name__ == "__main__":
    # Update the path to your CSV file as needed
    csv_file_path = "RatOmics_FINAL_processed_data_TSFormat.csv"
    run_dimension_reduction_analysis(csv_file_path)