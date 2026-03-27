import os
import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes=None, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # Load image
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        if img is None:
            raise FileNotFoundError(f"Image {img_id + self.img_ext} not found in {self.img_dir}")
        
        origin_h, origin_w = img.shape[:2]

        # Load mask (check if mask exists)
        mask_path = os.path.join(self.mask_dir, img_id + self.mask_ext)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask {mask_path} not found in {self.mask_dir}")

        mask = mask[..., None]  # Add channel dimension

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Handle tensor or numpy array
        if isinstance(img, torch.Tensor):
            # Already normalized and converted by albumentations ToTensorV2
            pass
        else:
            # Normalize image and mask for numpy arrays
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)  # Change shape to (C, H, W)
            img = torch.from_numpy(img)
        
        if isinstance(mask, torch.Tensor):
            # Already converted by albumentations ToTensorV2
            mask = mask.float() / 255.0
        else:
            mask = mask.astype('float32') / 255
            mask = mask.transpose(2, 0, 1)  # Change shape to (C, H, W)
            mask = torch.from_numpy(mask)
        
        return img, mask, {'img_id': img_id, 'origin_h': origin_h, 'origin_w': origin_w}
