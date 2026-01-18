import numpy as np
import pyvox.parser
import scipy.ndimage as ndimage

# Global constant for voxel resolution
DIM_SIZE = 64

def read_vox_clean(path: str):
    '''
    Read a .vox file, center the object, and crop/pad it to a fixed 64x64x64 resolution.
    '''
    vox = pyvox.parser.VoxParser(path).parse()
    dense = vox.to_dense()
    
    s_x, s_y, s_z = dense.shape
    
    off_x = (DIM_SIZE - s_x) // 2
    off_y = (DIM_SIZE - s_y) // 2
    off_z = (DIM_SIZE - s_z) // 2
    
    dst_x_start = max(0, off_x)
    dst_y_start = max(0, off_y)
    dst_z_start = max(0, off_z)
    
    dst_x_end = min(DIM_SIZE, off_x + s_x)
    dst_y_end = min(DIM_SIZE, off_y + s_y)
    dst_z_end = min(DIM_SIZE, off_z + s_z)
    
    src_x_start = max(0, -off_x)
    src_y_start = max(0, -off_y)
    src_z_start = max(0, -off_z)
    
    src_x_end = src_x_start + (dst_x_end - dst_x_start)
    src_y_end = src_y_start + (dst_y_end - dst_y_start)
    src_z_end = src_z_start + (dst_z_end - dst_z_start)

    adjusted_vox = np.zeros((DIM_SIZE, DIM_SIZE, DIM_SIZE), dtype=np.uint8)
    
    if (dst_x_end > dst_x_start) and (dst_y_end > dst_y_start) and (dst_z_end > dst_z_start):
        adjusted_vox[dst_x_start:dst_x_end, dst_y_start:dst_y_end, dst_z_start:dst_z_end] = \
            dense[src_x_start:src_x_end, src_y_start:src_y_end, src_z_start:src_z_end]

    return adjusted_vox

def compute_normals(voxel_grid):
    '''
    Input: (64, 64, 64) binary voxel or float voxel
    Output: (3, 64, 64, 64) normal vectors (nx, ny, nz)
    '''
    voxel_grid = voxel_grid.astype(np.float32)
    
    # 使用 Sobel 算子计算梯度
    dx = ndimage.sobel(voxel_grid, axis=0)
    dy = ndimage.sobel(voxel_grid, axis=1)
    dz = ndimage.sobel(voxel_grid, axis=2)
    
    # 堆叠
    normals = np.stack([dx, dy, dz], axis=0) # (3, 64, 64, 64)
    
    # 归一化
    norm = np.linalg.norm(normals, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        normals = normals / (norm + 1e-8)
        
    normals[np.isnan(normals)] = 0.0
    return normals.astype(np.float32)

def get_all_fragment_ids(voxel):
    unique_ids = np.unique(voxel)
    return unique_ids[unique_ids != 0]

def filter_by_fragments(voxel, keep_ids):
    mask = np.isin(voxel, keep_ids)
    return voxel * mask