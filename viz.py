#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json
import argparse

def visualize_minecraft_build(blocks, ax):
    x_dim, y_dim, z_dim = blocks.shape

    with open('mat_to_id_mapping.json', 'r') as f:
        mat_to_id_mapping = json.load(f)
    id_to_mat_mapping = {v: k for k, v in mat_to_id_mapping.items()}

    color_map = {
        'stone': '#808080',
        'dirt': '#8B4513',
        'grass': '#228B22',
        'wood': '#DEB887',
        'leaves': '#228B22',
        'glass': '#87CEEB',
        'water': '#1E90FF',
        'lava': '#FF4500',
        'sand': '#F4A460',
        'wool': '#FFFFFF',
        'default': '#FFA500'
    }

    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                if blocks[i, j, k] != 0:
                    material = id_to_mat_mapping.get(blocks[i, j, k], 'default')
                    base_material = material.split()[0].lower()
                    for mat in color_map:
                        if mat in material.lower():
                            color = color_map[mat]
                            break
                    else:
                        color = color_map['default']

                    if 'stairs' in material.lower():
                        if 'east' in material.lower():
                            vertices = np.array([[i, k, j], [i, k+1, j], [i+1, k+1, j+1], [i+1, k, j+1]])
                        elif 'west' in material.lower():
                            vertices = np.array([[i, k, j+1], [i, k+1, j+1], [i+1, k+1, j], [i+1, k, j]])
                        elif 'north' in material.lower():
                            vertices = np.array([[i, k, j+1], [i+1, k, j+1], [i+1, k+1, j], [i, k+1, j]])
                        elif 'south' in material.lower():
                            vertices = np.array([[i, k, j], [i+1, k, j], [i+1, k+1, j+1], [i, k+1, j+1]])
                        else:
                            continue

                        if 'upside' in material.lower():
                            vertices[:, 1] = k + 1 - (vertices[:, 1] - k)

                        poly = Poly3DCollection([vertices], alpha=0.8)
                        poly.set_facecolor(color)
                        ax.add_collection3d(poly)
                    else:
                        ax.bar3d(i, k, j, 1, 1, 1, color=color, alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    ax.set_xlim(0, x_dim)
    ax.set_ylim(0, z_dim)
    ax.set_zlim(0, y_dim)

    ax.view_init(elev=20, azim=45)

def load_and_visualize(file_paths):
    num_files = len(file_paths)
    if num_files == 0:
        raise ValueError("Please provide at least one file path.")

    fig = plt.figure(figsize=(6*num_files, 6))

    for i, file_path in enumerate(file_paths):
        print(i, file_path)
        ax = fig.add_subplot(1, num_files, i+1, projection='3d')
        build = np.load(file_path)
        print(build.shape)
        visualize_minecraft_build(build, ax)
        ax.set_title(f'Build: {file_path}')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize multiple Minecraft builds from .npy files")
    parser.add_argument("file_paths", type=str, nargs='+', help="Path(s) to the .npy file(s) containing the Minecraft build(s)")
    args = parser.parse_args()

    load_and_visualize(args.file_paths)

if __name__ == "__main__":
    main()
