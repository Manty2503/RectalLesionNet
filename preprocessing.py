import os 
import cv2 
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob


# Images are already Normalized between 0 and 1
# Masks are also already Normalied between 0 and 1
# But while Saving i am Unnormalizing the photo so as to appear as images

def load_dataset(path):
    raw_data = sorted(glob(os.path.join(path, "raw_files", "*")))

    return raw_data

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_dataset(files, save_dir):
    for file_path in tqdm(files, total=len(files)):
        file_name = file_path.split("\\")[-1].split(".")[0]
        name = f"{file_name}.png"

        data = np.load(file_path)

        image = data["image"]
        mask = data["label"]

        img = Image.fromarray(image*255)
        masks = Image.fromarray(mask*255)

        img = np.array(img)
        masks = np.array(masks)
        
        save_image_path = os.path.join(save_dir, "images", name)
        save_mask_path = os.path.join(save_dir, "masks", name)

        cv2.imwrite(save_image_path, img)
        cv2.imwrite(save_mask_path, masks)


dataset_path = "Attention U-Net/data"
dataset = load_dataset(os.path.join(dataset_path, "train"))

save_dir = os.path.join(dataset_path, "train")
for items in ["images", "masks"]:
    create_dir(os.path.join(save_dir, items))

save_dataset(dataset, save_dir)





