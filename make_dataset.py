#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

def load_and_preprocess_data(file_path):
    df = pd.read_parquet(file_path)

    # this is to fix some annoyings name in the dataset
    mat_mapping = {
        "(Jungle Wood, Bottom)": "jungle wood",
        "(Jungle Wood, Upper)": "jungle wood",
        "(Acacia Wood, Upper)": "acacia wood",
        "(Acacia Wood, Bottom)": "acacia wood",
        "(Birch Wood, Bottom)": "birch wood",
        "(Birch Wood, Upper)": "birch wood",
        "(Stone) Wooden Slab (Upper)": "wooden slab"
    }
    df['mat'] = df['mat'].replace(mat_mapping).str.lower()
    df['simplified_mat'] = df['mat'].apply(simplify_mat)
    return df

def simplify_mat(mat):
    wood_types = [
        "spruce", "birch", "jungle", "acacia", "dark oak",
        "mangrove", "cherry", "bamboo", "crimson", "warped",
    ]
    directions = ['west', 'south', 'east', 'north']

    pos = None
    if 'stairs' in mat:
        for direction in directions:
            if direction in mat:
                pos = direction
                break
        for nou in ['normal', 'upside']:
            if nou in mat:
                pos += f" {nou}" if pos is not None else nou
                break

    if '(' in mat:
        mat = mat[:mat.index('(')].strip()

    for wood_type in wood_types:
        if wood_type.lower() in mat:
            return mat.replace(wood_type, 'oak') + (f" ({pos})" if pos is not None else "")

    return mat if pos is None else mat + f" ({pos})"

def create_mappings(df):
    sorted_simplified_mats = sorted(df['simplified_mat'].unique())
    simplified_mat_to_id = {mat: i for i, mat in enumerate(sorted_simplified_mats)}
    df['simplified_mat_id'] = df['simplified_mat'].map(simplified_mat_to_id)

    sorted_structures = sorted(df['structure'].unique())
    structure_to_id = {structure: i for i, structure in enumerate(sorted_structures)}
    df['structure_id'] = df['structure'].map(structure_to_id)

    return df, simplified_mat_to_id

def save_processed_data(df, mat_to_id_mapping):
    df.to_parquet('cleaned.parquet')

    mat_to_id_serializable = {str(k): int(v) for k, v in mat_to_id_mapping.items()}
    with open('mat_to_id_mapping.json', 'w') as f:
        json.dump(mat_to_id_serializable, f, indent=2)

def process_structures(df, crop_size=32):
    df = df.sort_values(by=['structure_id', 'y', 'x', 'z'])
    grouped = df.groupby('structure')
    structures_blocks = []
    structures_to_idx = {}

    for _, group in tqdm(grouped, desc="Processing structures"):
        max_x, max_y, max_z = group['x'].max(), group['y'].max(), group['z'].max()
        if max_x >= crop_size or max_y >= crop_size or max_z >= crop_size:
            continue
        grid = np.zeros((crop_size, crop_size, crop_size), dtype=np.int16)
        for _, row in group.iterrows():
            x, y, z = int(row['x']), int(row['y']), int(row['z'])
            grid[x, y, z] = row['simplified_mat_id']
        structures_blocks.append(grid)
        structures_to_idx[group['structure'].iloc[0]] = len(structures_blocks) - 1

    return structures_blocks, structures_to_idx

def main():
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data('minecraft_structures.parquet')
    print("Creating mappings...")
    df, simplified_mat_to_id = create_mappings(df)
    print("Saving processed data...")
    save_processed_data(df, simplified_mat_to_id)
    print("Processing structures...")
    structures_blocks, structures_to_idx = process_structures(df)
    np.savez_compressed('builds.npz', np.array(structures_blocks))
    with open('structures_to_idx.json', 'w') as f:
        json.dump(structures_to_idx, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    main()
