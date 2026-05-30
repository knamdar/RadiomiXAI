#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:15:40 2025

@author: Ernest Namdar
"""

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import multiprocessing
from Dataset_nonormalization import RadiomicsDataset
import bnlearn as bn
import networkx as nx
from sklearn.metrics import mutual_info_score
from pyvis.network import Network

if __name__ == "__main__":
    # === Load data ===
    data_dir = "../data/"  
    standard_columns_file = "../data/Features.xlsx"  
    radiomics_dataset = RadiomicsDataset(data_dir, standard_columns_file)
    print("The dataset was loaded successfully")

    # Convert to cuDF DataFrame
    data_array = radiomics_dataset.data.numpy()  
    column_names = radiomics_dataset.standard_columns

    if data_array.shape[1] != len(column_names):
        raise ValueError("Mismatch: Number of columns in data does not match column_names")

    data_gpu = cudf.DataFrame(data_array, columns=column_names)
    target_column = "original_shape_VoxelVolume"

    # === Correlation Analysis ===
    correlation_matrix = data_gpu.corr()
    correlations = correlation_matrix.loc[target_column].abs()
    sorted_correlations = correlations[correlations > 0.8].sort_values(ascending=False)

    print("Sorted Correlation Coefficients with Voxel Volume:")
    print(sorted_correlations)

    high_corr_features = sorted_correlations.index.to_arrow().to_pylist()
    print("Highly correlated columns (potential confounders):")
    print(high_corr_features)

    # === Preprocessing for BN ===
    total_cores = multiprocessing.cpu_count()
    dedicated_cores = max(1, total_cores - 2)

    sampled_data_gpu = data_gpu[high_corr_features]

    for col in sampled_data_gpu.columns:
        col_data = sampled_data_gpu[col]
        bins = cp.linspace(col_data.min(), col_data.max(), num=11)
        sampled_data_gpu[col] = cp.digitize(col_data.values, bins) - 1

    non_constant_columns = sampled_data_gpu.nunique().to_pandas() > 1
    sampled_data_gpu = sampled_data_gpu.loc[:, non_constant_columns.index[non_constant_columns].tolist()]
    sampled_data = sampled_data_gpu.to_pandas()

    # === Bayesian Network ===
    model = bn.structure_learning.fit(sampled_data, methodtype='hc', scoretype='aic', n_jobs=dedicated_cores)
    bn_model = bn.parameter_learning.fit(model, sampled_data)
    edges = model["model_edges"]

    # === Mutual Information Calculation ===
    edge_weights = []
    for src, dst in edges:
        mi = mutual_info_score(sampled_data[src], sampled_data[dst])
        edge_weights.append((src, dst, mi))

    # === Save weighted edges to CSV ===
    edges_weighted_df = pd.DataFrame(edge_weights, columns=["source", "target", "mutual_information"])
    edges_weighted_df.to_csv("bayesian_network_edges_weighted.csv", index=False)

    # === Interactive Visualization with Pyvis ===
    G = nx.DiGraph()
    G.add_edges_from([(src, dst) for src, dst, _ in edge_weights])

    target_node1 = "original_shape_VoxelVolume"
    target_node2 = "original_shape_MeshVolume"
    neighbors = list(G.successors(target_node1)) + list(G.predecessors(target_node1)) + list(G.successors(target_node2)) + list(G.predecessors(target_node2))

    net = Network(height="1000px", width="100%", directed=True, notebook=False)
    net.barnes_hut()

    for node in G.nodes():
        if node == target_node1 or node == target_node2:
            color = "red"
            font_size = 300
        elif node in neighbors:
            color = "blue"
            font_size = 200
        else:
            color = "lightgray"
            font_size = 80
        net.add_node(node, label=node, color=color, size=150, font={'size': font_size})

    for src, dst, mi in edge_weights:
        net.add_edge(src, dst, width=mi * 10, title=f"Mutual Information: {mi:.3f}")

    net.toggle_physics(True)
    net.show("bayesian_network.html", notebook=False)

    print("Bayesian network saved and visualized.")

    # Save the full trained BN model for future reuse
    bn.save(bn_model, filepath="bayesian_network_model.json")