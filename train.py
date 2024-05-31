import argparse
import os
import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
from model import *
from DataLoader_train import get_Training_Set
from DataLoader_val import get_Val_Set
from loss import dice_coeff, FocalLoss

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=60, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=2, help="size of batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--threads", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss Function
criterion = FocalLoss()

# Model
model = AttentionUNet()

if cuda:
    model = model.cuda()

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])

# Optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor Type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

print('===> Loading datasets')
train_dataset = get_Training_Set()
training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
val_dataset = get_Val_Set()
validation_data_loader = DataLoader(dataset=val_dataset, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)


fieldnames = ['epoch', 'training_loss', 'val_loss', 'training_dice_coeff', 'val_dice_coeff']
train_epoch_losses = []
val_epoch_losses = []

df = pd.DataFrame(columns=['epoch', 'training_loss', 'val_loss', 'training_dice_coeff', 'val_dice_coeff'])

for epoch in range(1, opt.n_epochs+1):
    print(f'Epoch {epoch}/{opt.n_epochs}')
    print('-' * 20)

    # Each epoch has a training and Validation Phase
    # Initialize Batch Summary
    batchsummary = {a: [0] for a in fieldnames}
    batch_train_loss = 0.0
    batch_val_loss = 0.0
    batchsummary["epoch"] = epoch

    model.train()

    for i, sample in enumerate(tqdm(training_data_loader)):
        inputs = Variable(sample['image']).float()
        masks = Variable(sample['mask']).float()

        if cuda:
            inputs = inputs.cuda()
            masks = masks.cuda()
        
        # Make Gradients to Zero
        optimizer.zero_grad()

        outputs = Variable(model(inputs))
        loss = criterion(outputs, masks)

        y_pred = outputs.data.cpu().numpy().ravel()
        y_true = masks.data.cpu().numpy().ravel()

        batchsummary["training_dice_coeff"].append(dice_coeff(y_pred, y_true))

        loss.backward()
        optimizer.step()

        batch_train_loss += loss.item() *sample["image"].size(0)

    epoch_train_loss = batch_train_loss / len(training_data_loader)
    batchsummary['training_loss'] = epoch_train_loss
    train_epoch_losses.append(epoch_train_loss)
    print(f'Training Loss: {epoch_train_loss:.4f}')


    model.eval()

    for i, sample in enumerate(tqdm(validation_data_loader)):
        inputs = Variable(sample['image']).float()
        masks = Variable(sample['mask']).float()

        if cuda:
            inputs = inputs.cuda()
            masks = masks.cuda()

        with torch.no_grad():   
            outputs = Variable(model(inputs))
            loss = criterion(outputs, masks)

        y_pred = outputs.data.cpu().numpy().ravel()
        y_true = masks.data.cpu().numpy().ravel()
        batchsummary['val_dice_coeff'].append(dice_coeff(y_pred, y_true))

        # Accumulate batch loss
        batch_val_loss += loss.item() * sample['image'].size(0)

    epoch_val_loss = batch_val_loss / len(validation_data_loader)
    batchsummary['val_loss'] = epoch_val_loss
    val_epoch_losses.append(epoch_val_loss)

    print(f'Validation Loss: {epoch_val_loss:.4f}')


    if epoch >= 0:
        torch.save(model.state_dict(), r"/scratch/ravihm.scee.iitmandi/MantyMourya/AttentionUNet/result/saved_models/model_%d.pth" % epoch)

    for field in fieldnames[3:]:
        batchsummary[field] = np.mean(batchsummary[field])
    print(f'\tTraining Dice Coefficient: {batchsummary["training_dice_coeff"]:.4f}, Val Dice Coefficient: {batchsummary["val_dice_coeff"]:.4f}')


    # Convert batchsummary to DataFrame
    # Append the batch summary DataFrame to the main DataFrame using pd.concat
    batchsummary_df = pd.DataFrame([batchsummary])
    df = pd.concat([df, batchsummary_df], ignore_index=True)

    # Save the DataFrame to a CSV file, overwriting it each time
    csv_file_path = r"/scratch/ravihm.scee.iitmandi/MantyMourya/AttentionUNet/result/saved_models/addi_info/training_log.csv"
    df.to_csv(csv_file_path, index=False)




