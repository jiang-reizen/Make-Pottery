import numpy as np
import plotly.graph_objects as go
from . import utils

# --- Wrapper functions for backward compatibility or simple usage ---

def __read_vox__(path: str):
    '''Wrapper to read voxels using utils with fixed 64 dim.'''
    return utils.read_vox_clean(path)

def __read_vox_frag__(path: str, fragment_idx: int):
    '''Wrapper to read a specific fragment.'''
    full_vox = utils.read_vox_clean(path)
    return utils.filter_by_fragments(full_vox, [fragment_idx])

# --- Visualization Logic ---

def plot(vox, save_dir=None):
    '''
    Plot the whole voxel matrix (binary view, ignoring fragment labels).
    
    :param vox: np.array (np.uint8), voxel data
    :param save_dir: str or None, directory to save the plot HTML
    '''
    # Get coordinates of non-zero voxels
    voxels = np.array(np.where(vox > 0)).T
    
    if len(voxels) == 0:
        print("Warning: Voxel is empty, nothing to plot.")
        return

    x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
    
    fig = go.Figure(
        data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name='Pottery',
            marker=dict(
                size=5,
                symbol='square',
                color='#ceabb2',
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
    )
    fig.update_layout(title='Full Voxel Visualization (64x64x64)')
    
    if save_dir:
        fig.write_html(save_dir)
    else:
        fig.show()

def plot_frag(vox_pottery, save_dir=None):
    '''
    Plot the voxel with different colors for different fragment labels.
    
    :param vox_pottery: np.array (np.uint8), voxel data with labels
    :param save_dir: str or None, directory to save the plot HTML
    '''
    # Define a color palette
    colors= ['#ceabb2', '#d05d86', '#7e1b2f', '#c1375b', '#cdc1c3',
             '#4C4C73', '#FFD700', '#2992BC', '#FF69B4', '#7B0BE3',
             '#2E8B57', '#8B4513']
    
    data = []
    
    # Use utils to get IDs present in the object
    unique_ids = utils.get_all_fragment_ids(vox_pottery)

    if len(unique_ids) == 0:
        print("Warning: No fragments found in voxel data.")
        return

    for i, uid in enumerate(unique_ids):
        # Extract coordinates for this specific fragment ID
        x, y, z = np.where(vox_pottery == uid)

        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            name = f'Fragment {uid}',
            marker=dict(
                size=5,
                symbol='square',
                color=colors[i % len(colors)], # Cycle colors
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        data.append(scatter)

    fig = go.Figure(data=data)
    fig.update_layout(title='Pottery Fragments Visualization')
    
    if save_dir:
        fig.write_html(save_dir)
    else:
        fig.show()

def plot_join(vox_input, vox_missing, save_dir=None):
    '''
    Visualize Input parts vs. Missing parts (or Generated parts).
    Input is shown in light gray, Missing/Generated in red.
    
    :param vox_input: np.array, the input fragments provided to the network
    :param vox_missing: np.array, the missing parts (or the network output)
    '''
    data = []
    
    # 1. Plot Input (Gray)
    x1, y1, z1 = np.where(vox_input > 0)
    if len(x1) > 0:
        data.append(go.Scatter3d(
            x=x1, y=y1, z=z1,
            mode='markers',
            name='Input Fragments',
            marker=dict(
                size=4,
                symbol='square',
                color='lightgray',
                opacity=0.4,
                line=dict(width=1, color='gray')
            )
        ))

    # 2. Plot Missing/Generated (Red)
    x2, y2, z2 = np.where(vox_missing > 0)
    if len(x2) > 0:
        data.append(go.Scatter3d(
            x=x2, y=y2, z=z2,
            mode='markers',
            name='Reconstructed/Missing',
            marker=dict(
                size=5,
                symbol='square',
                color='#7e1b2f',
                line=dict(width=1, color='DarkSlateGrey')
            )
        ))

    fig = go.Figure(data=data)
    fig.update_layout(title='Input vs Reconstruction Visualization')
    
    if save_dir:
        fig.write_html(save_dir)
    else:
        fig.show()