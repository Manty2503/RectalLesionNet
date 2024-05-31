import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

path1 = r"C:\Users\manta\OneDrive\Desktop\CT Scan Reconstruction\Attention U-Net\data\prediction"
path2 = r"C:\Users\manta\OneDrive\Desktop\CT Scan Reconstruction\Attention U-Net\data\test\masks"

pred_list = sorted(glob(os.path.join(path1, "*")))
mask_list = sorted(glob(os.path.join(path2, "*")))

score = []

for pred_y, true_y in tqdm(zip(pred_list, mask_list), total=len(pred_list)):
    img = cv2.imread(pred_y)/255.0
    img = np.transpose(img, (2, 0, 1))
    pred_y = img[0]
    pred_y = pred_y.astype(np.int32)
    pred_y = pred_y.flatten()

    img = cv2.imread(true_y)/255.0
    img = np.transpose(img, (2, 0, 1))
    true_y = img[0]
    true_y = true_y.astype(np.int32)
    true_y = true_y.flatten()

    acc_value = accuracy_score(pred_y, true_y)
    f1_value = f1_score(pred_y, true_y, labels=[0, 1], average="binary")
    jac_value = jaccard_score(pred_y, true_y, labels=[0, 1], average="binary")
    recall_value = recall_score(pred_y, true_y, labels=[0, 1], average="binary")
    precision_value = precision_score(pred_y, true_y, labels=[0, 1], average="binary")

    score.append([acc_value, f1_value, jac_value, recall_value, precision_value])

mean_score = np.mean(score, axis=0)

print(f"Accuracy: {mean_score[0]:0.5f}")
print(f"F1: {mean_score[1]:0.5f}")
print(f"Jaccard: {mean_score[2]:0.5f}")
print(f"Recall: {mean_score[3]:0.5f}")
print(f"Precision: {mean_score[4]:0.5f}")