#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ernest Namdar
"""

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cuml.linear_model import LinearRegression
from cuml.metrics.regression import r2_score
from Dataset_nonormalization import RadiomicsDataset

# Standardization: Z-score (Mean 0, Std 1)
def zscore_standardization(data_gpu, features):
    means = data_gpu[features].mean()
    stds = data_gpu[features].std()
    return (data_gpu[features] - means) / stds

# Min-Max Scaling: Rescale to [0,1]
def minmax_scaling(data_gpu, features):
    mins = data_gpu[features].min()
    maxs = data_gpu[features].max()
    return (data_gpu[features] - mins) / (maxs - mins)

# Log Transformation: Apply log1p (log(x+1) to avoid log(0))
def log_transformation(data_gpu, features):
    log_df = cudf.DataFrame()
    for col in features:
        log_df[col] = cp.log1p(data_gpu[col].to_cupy())  # Convert to CuPy and apply log1p
    return log_df

# Divide by Volume: Normalize features by `original_shape_VoxelVolume`
def divide_by_volume(data_gpu, features, volume_column='original_shape_VoxelVolume'):
    div_df = cudf.DataFrame()
    for col in features:
        div_df[col] = data_gpu[col] / data_gpu[volume_column]  # Element-wise division
    return div_df

# Compute VIF using cuML Linear Regression
def compute_vif_cuml(data_gpu, features, normalization='none'):
    if normalization == 'zscore':
        X_gpu = zscore_standardization(data_gpu, features)
    elif normalization == 'minmax':
        X_gpu = minmax_scaling(data_gpu, features)
    elif normalization == 'log':
        X_gpu = log_transformation(data_gpu, features)
    elif normalization == 'divide_by_volume':
        X_gpu = divide_by_volume(data_gpu, features)
    else:
        X_gpu = data_gpu[features]  # No normalization

    X_gpu = X_gpu.astype(cp.float32).dropna()
    vif_dict = {}

    for feature in features:
        X_sub = X_gpu.drop(columns=[feature])
        y = X_gpu[feature]

        model = LinearRegression(fit_intercept=True)
        model.fit(X_sub, y)

        y_hat = model.predict(X_sub)
        r2 = r2_score(y, y_hat, convert_dtype=True)
        vif = 1 / (1 - r2) if 0 < r2 < 1 else np.inf
        vif_dict[feature] = vif

    return vif_dict

if __name__ == "__main__":
    data_dir = '../data/'  
    standard_columns_file = '../data/Features.xlsx'  

    radiomics_dataset = RadiomicsDataset(data_dir, standard_columns_file)
    print("The dataset was loaded successfully")

    data_gpu = cudf.DataFrame.from_records(radiomics_dataset.data.numpy())
    column_names = radiomics_dataset.standard_columns
    data_gpu.columns = column_names
    target_column = 'original_shape_VoxelVolume'

    are_same = data_gpu["original_shape_VoxelVolume"].equals(data_gpu["lbp-2D_gldm_GrayLevelNonUniformity"])
    print(f"⚠️ Are the two columns identical? {are_same}")
    # conclusion: ⚠️ Are the two columns identical? True
    
    # Print a few examples from the target column
    print("\n🔍 Sample values from target column:", target_column)
    print(data_gpu[target_column].head(10))  # Print first 10 values
    
    # Check if 'lbp-2D_gldm_GrayLevelNonUniformity' exists in the dataset and print samples
    if "lbp-2D_gldm_GrayLevelNonUniformity" in data_gpu.columns:
        print("\n🔍 Sample values from 'lbp-2D_gldm_GrayLevelNonUniformity':")
        print(data_gpu["lbp-2D_gldm_GrayLevelNonUniformity"].head(10))  # Print first 10 values
    else:
        print("\n⚠️ Column 'lbp-2D_gldm_GrayLevelNonUniformity' not found in dataset!")

    # Drop specific features that should be excluded from analysis
    features_to_drop = ['original_shape_MeshVolume', 'lbp-2D_gldm_GrayLevelNonUniformity']  # Add more if needed
    data_gpu = data_gpu.drop(columns=features_to_drop)

    correlation_matrix = data_gpu.corr()
    correlations = correlation_matrix.loc[target_column].abs()

    # Print sorted correlation coefficients
    sorted_correlations = correlations[correlations > 0.8].sort_values(ascending=False)
    print("\n🔍 Sorted Correlation Coefficients with Voxel Volume:")
    print(sorted_correlations)

    high_corr_features = sorted_correlations.index.to_arrow().to_pylist()
    if target_column in high_corr_features:
        high_corr_features.remove(target_column)

    print("Highly correlated columns (potential confounders):")
    print(high_corr_features)

    normalization_methods = ['zscore', 'minmax', 'log', 'divide_by_volume']
    vif_results_dict = {}

    for norm_method in normalization_methods:
        print(f"\n Testing normalization: {norm_method}")
        vif_results = compute_vif_cuml(data_gpu, [target_column] + high_corr_features, normalization=norm_method)
        vif_results_dict[norm_method] = vif_results

    vif_df = pd.DataFrame(vif_results_dict)
    vif_df.insert(0, "Feature", vif_df.index)
    vif_df["Correlation (Before Normalization)"] = vif_df["Feature"].map(sorted_correlations.to_dict())

    vif_df_gpu = cudf.DataFrame.from_pandas(vif_df)
    print("\n📊 VIF Comparison Across Normalization Methods:")
    print(vif_df_gpu)

    output_file = "vif_results.csv"
    vif_df_gpu.to_csv(output_file, index=False)
    print(f"\n📁 VIF results saved to: {output_file}")
