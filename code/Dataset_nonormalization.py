#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ernest Namdar
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Load standard columns and metadata
def load_standard_columns(excel_file):
    df = pd.read_excel(excel_file)
    standard_columns = df.apply(lambda row: f"{row['Source']}_{row['Feature category']}_{row['Feature']}", axis=1).tolist()
    return standard_columns

def load_metadata(excel_file):
    df = pd.read_excel(excel_file)
    metadata = {
        'Source': {},
        'Feature category': {},
        'Feature': {}
    }
    for index, row in df.iterrows():
        source = row['Source']
        feature_category = row['Feature category']
        feature = row['Feature']
        full_feature_name = f"{source}_{feature_category}_{feature}"
        
        if source not in metadata['Source']:
            metadata['Source'][source] = []
        metadata['Source'][source].append(full_feature_name)
        
        if feature_category not in metadata['Feature category']:
            metadata['Feature category'][feature_category] = []
        metadata['Feature category'][feature_category].append(full_feature_name)
        
        metadata['Feature'][full_feature_name] = full_feature_name
    return metadata

# Radiomics Dataset
class RadiomicsDataset(Dataset):
    def __init__(self, data_dir, standard_columns_file):
        self.standard_columns = load_standard_columns(standard_columns_file)
        self.data = self._load_data(data_dir)

    def _load_data(self, data_dir):
        data_points = []

        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, file))
                    aligned_df = df.reindex(columns=self.standard_columns).fillna(0)
                    # for column in self.standard_columns:
                    #     col_mean = aligned_df[column].mean()
                    #     col_std = aligned_df[column].std()
                    #     if col_std != 0:
                    #         aligned_df[column] = (aligned_df[column] - col_mean) / col_std
                    #     else:
                    #         aligned_df[column] = 0

                    data_points.extend(aligned_df.values.tolist())

        data_array = np.array(data_points)
        return torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # Define paths
    data_dir = '../data/'  # Path the data directory
    standard_columns_file = '../data/Features.xlsx'  # Path to the standard columns Excel file
    
    # Create the dataset
    radiomics_dataset = RadiomicsDataset(data_dir, standard_columns_file)
    
    # Create DataLoader
    dataloader = DataLoader(radiomics_dataset, batch_size=4, shuffle=True)
    
    # Iterate over the DataLoader
    for i, data in enumerate(dataloader):
        print(f"Batch {i + 1}")
        print(data)
        if i == 1:  # Just to limit the output for demonstration
            break
