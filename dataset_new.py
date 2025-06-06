import os
import sys
import glob

import cv2
import numpy as np
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2
from utils import sobel_transform

class CustomDataset(Dataset):
    def __init__(self, root_path, img_transforms=None):
        self.image_paths = (
            glob.glob(os.path.join(root_path, 'images', '*.jpg')) +
            glob.glob(os.path.join(root_path, 'images', '*.tif')) +
            glob.glob(os.path.join(root_path, 'images', '*.ppm'))
        )
        self.mask_paths = (
            glob.glob(os.path.join(root_path, 'mask', '*.png')) +
            glob.glob(os.path.join(root_path, 'mask', '*.tif')) +
            glob.glob(os.path.join(root_path, 'mask', '*.ppm'))
        )

        self.image_transforms = img_transforms
        self.name = os.path.basename(os.path.dirname(root_path))

    def get_name(self): 
        return self.name
    
    def __len__(self): 
        return len(self.image_paths) 
    
    def __getitem__(self, index): 
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        image = image[:, :, 2]
        image = image.repeat(image, 3)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        mask = np.ceil(mask / 255.0).astype(np.uint8)
        
        if self.image_transforms: 
            transformed = self.image_transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

            edge_np = sobel_transform(image.clone().detach().cpu().numpy().transpose(1, 2, 0))
            edge = ToTensorV2()(image=edge_np)['image']
        else: 
            raise Exception("Transformation is required")
        
        return {
            'image': image, 
            'mask': mask, 
            'edge': edge
        }
