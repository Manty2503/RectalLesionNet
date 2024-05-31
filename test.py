import argparse
import os
import numpy as np
import sys
import pandas as pd
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from model import *
from DataLoader_test import get_Testing_Set
from loss import dice_coeff, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=18, help="the name of the trained model")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

criterion = FocalLoss()

model = AttentionUNet()

if cuda:
    model = model.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])


print('===> Loading Datasets')
Test_dataset = get_Testing_Set()
testing_data_loader = DataLoader(dataset=Test_dataset, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

model_path = r"/scratch/ravihm.scee.iitmandi/MantyMourya/AttentionUNet/result/saved_models/model_%d.pth" % opt.epoch
model.load_state_dict(torch.load(model_path))

index = 0
fieldnames = ['index', 'testing_dice_coeff']

df = pd.DataFrame(columns=['index', 'testing_dice_coeff'])

save_dir = r"/scratch/ravihm.scee.iitmandi/MantyMourya/AttentionUNet/result/prediction"

for i, sample in enumerate(tqdm(testing_data_loader)):

    batchsummary = {a: [0] for a in fieldnames}
    batchsummary["index"] = i

    inputs = Variable(sample['image']).float()
    masks = Variable(sample['mask']).float()

    if cuda:
        inputs = inputs.cuda()
        masks = masks.cuda()

    with torch.no_grad():   
        outputs = model(inputs)
        loss = criterion(outputs, masks)

    y_pred = outputs.data.cpu().numpy().ravel()
    y_true = masks.data.cpu().numpy().ravel()
    batchsummary['testing_dice_coeff'] = dice_coeff(y_pred, y_true)

    # Convert batchsummary to DataFrame
    # Append the batch summary DataFrame to the main DataFrame using pd.concat
    batchsummary_df = pd.DataFrame([batchsummary])
    df = pd.concat([df, batchsummary_df], ignore_index=True)

    # Save the DataFrame to a CSV file, overwriting it each time
    csv_file_path = r"/scratch/ravihm.scee.iitmandi/MantyMourya/AttentionUNet/result/saved_models/addi_info/testing_log.csv"
    df.to_csv(csv_file_path, index=False)

    img = (outputs.cpu().numpy()*255).astype(np.uint8)

    img_save_path = os.path.join(save_dir, f"prediction_{i}.png")
    cv2.imwrite(img_save_path, img)




