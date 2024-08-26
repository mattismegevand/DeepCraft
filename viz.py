import numpy as np
import matplotlib.pyplot as plt

# TODO:
# - allow to enable or not interactivity
# - add stairs viz
# - add textures
def visualize_minecraft_build(blocks):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    x_dim, y_dim, z_dim = blocks.shape
    unique_materials = np.unique(blocks)
    unique_materials = unique_materials[unique_materials != -1]
    color_map = plt.colormaps['tab20']

    max_dim = max(x_dim, y_dim, z_dim)
    x_scale = max_dim / x_dim
    y_scale = max_dim / y_dim
    z_scale = max_dim / z_dim

    for x in range(x_dim):
        for y in range(y_dim):
            for z in range(z_dim):
                if blocks[x, y, z] != 0:
                    color = color_map(np.where(unique_materials == blocks[x, y, z])[0][0] / len(unique_materials))
                    ax.bar3d(x * x_scale, y * y_scale, z * z_scale, 
                             x_scale, y_scale, z_scale, 
                             color=color, shade=True, edgecolor='black', alpha=0.8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Minecraft Build Visualization')

    ax.set_box_aspect((x_dim, y_dim, z_dim))

    sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=len(unique_materials)-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks(range(len(unique_materials)))
    cbar.set_ticklabels(unique_materials)

    ax.mouse_init()

    plt.show()
