import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from . import utils

class FragmentDataset(Dataset):
    def __init__(self, vox_path: str, vox_type: str, transform=None):
        self.vox_path = vox_path
        self.vox_type = vox_type
        self.transform = transform
        self.dim_size = 64
        search_path = os.path.join(vox_path, vox_type, "*", "*.vox")
        self.vox_files = sorted(glob.glob(search_path))
        print(f"[{self.vox_type.upper()}] Dataset initialized. Found {len(self.vox_files)} files.")

    def __len__(self):
        return len(self.vox_files)

    def __read_vox__(self, path):
        return utils.read_vox_clean(path)

    def __select_fragment__(self, voxel):
        all_ids = utils.get_all_fragment_ids(voxel)
        if len(all_ids) <= 1:
            return voxel
        
        # 50% chance Hard Mode (1 fragment), 50% random
        if np.random.random() < 0.5:
            num_to_keep = 1
        else:
            num_to_keep = np.random.randint(1, len(all_ids) + 1)
        
        select_ids = np.random.choice(all_ids, size=num_to_keep, replace=False)
        return utils.filter_by_fragments(voxel, select_ids)

    def __select_fragment_specific__(self, voxel, select_frag_id):
        if isinstance(select_frag_id, (int, np.integer)):
            select_ids = np.array([select_frag_id])
        else:
            select_ids = np.array(select_frag_id)
        return utils.filter_by_fragments(voxel, select_ids)

    def __getitem__(self, idx):
        vox_path = self.vox_files[idx]
        vox_gt = self.__read_vox__(vox_path)
        
        # get Class Label
        #  data/train/1/xxx.vox, data/train/2/xxx.vox
        try:
            parent_dir = os.path.basename(os.path.dirname(vox_path))
            label = int(parent_dir) - 1
        except ValueError:
            label = 0 # Default

        if self.vox_type == 'train':
            frag_input = self.__select_fragment__(vox_gt)
        else:
            all_ids = utils.get_all_fragment_ids(vox_gt)
            if len(all_ids) > 0:
                target_id = all_ids[0]
                frag_input = self.__select_fragment_specific__(vox_gt, target_id)
            else:
                frag_input = vox_gt
        
        if self.transform:
            # Transform return 4-channel tensor (1 Occ + 3 Norm)
            frag_input = self.transform(frag_input)
            
            # GT only need 1-channel Occupancy
            vox_gt_tensor = torch.from_numpy((vox_gt > 0).astype(np.float32)).unsqueeze(0)

        return frag_input, vox_gt_tensor, label, vox_path

class ToTensor3D(object):
    '''
    Converts (D, H, W) -> (4, D, H, W) 
    Channel 0: Occupancy
    Channel 1-3: Normals
    '''
    def __init__(self, binary=True):
        self.binary = binary

    def __call__(self, vox):
        # Normals (based on original uint8 data)
        normals = utils.compute_normals(vox) # (3, 64, 64, 64)

        # Process Occupancy
        if self.binary:
            vox_occ = (vox > 0).astype(np.float32)
        else:
            vox_occ = vox.astype(np.float32)
        
        vox_occ = vox_occ[np.newaxis, ...] # (1, 64, 64, 64)
        
        # Concatenate
        combined = np.concatenate([vox_occ, normals], axis=0) # (4, 64, 64, 64)

        return torch.from_numpy(combined)