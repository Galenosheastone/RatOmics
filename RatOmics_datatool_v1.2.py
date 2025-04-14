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
import matplotlib.patches as mpatches

def run_dimension_reduction_analysis(csv_path, output_folder="output_plots"):
    """
    Perform PCA, PLS-DA (with cross-validated validation metrics including Q2),
    and hierarchical clustering with a legend on the dendrogram.

    The CSV is expected to contain:
      - 'Sample_ID', 'Group', 'Time' columns (as metadata)
      - All other columns are metabolite concentrations (pre-processed, scaled, normalized)

    Plots are saved into the output_folder.
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # -----------------------------
    # 1) Load Data
    # -----------------------------
    df = pd.read_csv(csv_path)
    meta_cols = ['Sample_ID', 'Group', 'Time']
    X = df.drop(columns=meta_cols)
    
    # For plotting and PLS-DA: extract Group and Time
    group_labels = df['Group'].astype(str)
    time_labels = df['Time']  # Can be used for additional analyses

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
    plt.close()

    # -----------------------------
    # 3) PLS-DA and Model Validation
    # -----------------------------
    # Encode group labels numerically for PLS-DA
    le = LabelEncoder()
    y_encoded = le.fit_transform(group_labels)
    
    # Fit a PLS model with 2 components
    pls = PLSRegression(n_components=2)
    pls.fit(X.values, y_encoded)
    
    # Extract X-scores from the fitted model for plotting
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

    # Cross-validate the PLS model (5-fold)
    y_pred_cv = cross_val_predict(pls, X.values, y_encoded, cv=5)
    
    # Since predictions are continuous, round them to the nearest valid class integer.
    y_pred_round = np.clip(np.rint(y_pred_cv), y_encoded.min(), y_encoded.max()).astype(int)
    
    # Compute accuracy and R² (explained variance)
    accuracy = accuracy_score(y_encoded, y_pred_round)
    r2 = r2_score(y_encoded, y_pred_cv)
    
    # Compute Q2 as: 1 - (PRESS / TSS)
    press = np.sum((y_encoded - y_pred_cv)**2)
    tss = np.sum((y_encoded - np.mean(y_encoded))**2)
    Q2 = 1 - press/tss
    
    cm = confusion_matrix(y_encoded, y_pred_round)
    
    print("PLS-DA Cross-Validated Accuracy:", accuracy)
    print("PLS-DA Cross-Validated R^2 Score:", r2)
    print("PLS-DA Q2:", Q2)
    print("PLS-DA Confusion Matrix:\n", cm)
    
    # Save the validation metrics to a text file.
    metrics_path = os.path.join(output_folder, "PLSDA_Validation_Metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("PLS-DA Cross-Validated Metrics\n")
        f.write("------------------------------\n")
        f.write(f"Accuracy: {accuracy:.3f}\n")
        f.write(f"R^2 Score: {r2:.3f}\n")
        f.write(f"Q2: {Q2:.3f}\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
    
    # -----------------------------
    # 4) Hierarchical Clustering with Legend
    # -----------------------------
    # Compute the linkage matrix using Ward's method
    linkage_matrix = linkage(X.values, method='ward')
    
    # Create a mapping from Sample_ID to Group for coloring the labels
    group_dict = dict(zip(df['Sample_ID'], df['Group']))
    
    plt.figure(figsize=(10, 8))
    dendro_data = dendrogram(linkage_matrix,
                             labels=df['Sample_ID'].values,
                             leaf_rotation=90,
                             color_threshold=0)
    
    # Access current axis for modifying tick labels
    ax = plt.gca()
    
    # Determine unique groups and assign colors using a colormap
    unique_groups_sorted = sorted(df['Group'].unique())
    cmap = plt.cm.get_cmap('tab10', len(unique_groups_sorted))
    group_colors = {group: cmap(i) for i, group in enumerate(unique_groups_sorted)}
    
    # Color tick labels based on sample's group
    xticklabels = ax.get_xmajorticklabels()
    for label in xticklabels:
        sample_id = label.get_text()
        group = group_dict.get(sample_id, "Unknown")
        label.set_color(group_colors.get(group, "black"))
    
    # Create legend handles for the groups
    legend_handles = [mpatches.Patch(color=group_colors[group], label=group)
                      for group in unique_groups_sorted]
    plt.legend(handles=legend_handles, title="Group", loc="upper right", bbox_to_anchor=(1.15, 1))
    
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
    # Update the path to your CSV file as needed:
    csv_file_path = "RatOmics_FINAL_processed_data_TSFormat.csv"
    run_dimension_reduction_analysis(csv_file_path)