import os
from glob import glob
import torch.utils.data as data
import numpy as np
import cv2 
from PIL import Image
import torch

# Path of Images and there corresponding Masks
path = r"C:\Users\manta\OneDrive\Desktop\CT Scan Reconstruction\Attention U-Net\data\train"

image_list = sorted(glob(os.path.join(path, "images", "*")))
mask_list = sorted(glob(os.path.join(path, "masks", "*")))

def get_Training_Set():
    return DatasetFromFolder(image_list, mask_list)

class DatasetFromFolder(data.Dataset):

    def __init__(self, image_list, mask_list):
        super(DatasetFromFolder, self).__init__()
        self.image_list = image_list
        self.mask_list = mask_list

    def __getitem__(self, index):
        image = self.image_list[index]
        mask = self.mask_list[index]

        # Normalizing the Images and the Masks
        img = cv2.imread(image)/255.0
        msk = cv2.imread(mask)/255.0

        # Transpose the dimensions from (512, 512, 3) to (3, 512, 512)
        img = np.transpose(img, (2, 0, 1))
        msk = np.transpose(msk, (2, 0, 1))

        return {"image": torch.Tensor(img), "mask": torch.Tensor(msk[0].reshape(1, 512, 512))}
    
    def __len__(self):
        return len(self.image_list)
    
a = get_Training_Set()
print((a.__getitem__(15))["mask"].shape)
print(a.__len__())
print(type(a.__getitem__(3)["image"][0][0][0]))
