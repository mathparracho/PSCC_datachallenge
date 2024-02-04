from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
    DivisiblePadd,
    RandAffined,
    RandRotated,
    RandGaussianNoised,
    Resized
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet, SwinUNETR
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from datetime import datetime
import shutil
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

seed = 24
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_dir = "/tsi/data_education/data_challenge"
train_images = sorted(glob.glob(os.path.join(data_dir, "train/volume", "*.nii*")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "train/seg", "*.nii*")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[0], data_dicts[:] # before was 20
print(train_images)


####################################################################

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        DivisiblePadd(["image", "label"], 16),
        Resized(keys=["image", "label"], spatial_size=(192,192,192))
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        DivisiblePadd(["image", "label"], 16),
        Resized(keys=["image", "label"],spatial_size=(192,192,192))
    ]
)

augm_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        RandAffined(keys=['image', 'label'], prob=0.5, translate_range=10),
        RandRotated(keys=['image', 'label'], prob=0.5, range_x=10.0),
        RandGaussianNoised(keys='image', prob=0.5),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        DivisiblePadd(["image", "label"], 16),
        Resized(keys=["image", "label"],spatial_size=(192,192,192))
    ]
)


################################################################

# 16 images -> transform
train_ds = Dataset(data=train_files, transform=train_transforms)

# 16 images -> transform + data augmentation
#augm_ds= Dataset(data=train_files, transform=augm_transforms)

# Not declararing the batch size -> whole dataset -> 16(data) + 16(data augmentation) = 32 images per epoch
# this means 1 batch = 32 images
#train_ds = ConcatDataset([train_ds, augm_ds])

train_loader = DataLoader(train_ds,num_workers=4,batch_size=1)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds,num_workers=4,batch_size=1)


print("\n\n\n\n\n\n")
print(train_loader)
print("\n\n\n\n\n\n")

################################################################
import numpy as np
import cv2

def mask2rle(img):
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    """
    if np.array_equal(img, np.zeros(img.shape)):
      return 0

    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle: str, shape,label=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1] * shape[2], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction

def find_largest_containing_circle(segmentation, pixdim):
    largest_circle = None
    largest_slice = -1
    max_radius = -1

    segmentation8 = segmentation.astype(np.float32).astype('uint8')
    for i in range(segmentation8.shape[-1]):
        # Find the contours in the segmentation
        contours, _ = cv2.findContours(image = segmentation8[:,:,i], mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Fit the smallest circle around the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius > max_radius:
                max_radius = radius
                largest_circle = ((int(x), int(y)), int(radius))
                largest_slice = i
    recist = max_radius * 2 * pixdim[0]
#     print(max_radius)
    predicted_volume = np.round(np.sum(segmentation.flatten())*pixdim[0]*pixdim[1]*pixdim[2]*0.001,2)
    return recist, predicted_volume, largest_circle, largest_slice

###############################################################
class SegEnsemble(nn.Module):
    def __init__(self, modelA, modelB, modelC):
        super(SegEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.modelC = modelC
        self.classifier = nn.Conv3d(4, 2, kernel_size=3, padding=1)

    def forward(self, inputData):
        x1 = self.modelA(inputData)
        x2 = self.modelB(inputData)
        x2 = torch.argmax(x2, dim=1, keepdim=True).float()
        x3 = self.modelC(x2)
        x = torch.cat((x1, x3), dim=1)
        x = self.classifier(x)
        return x


modelA = torch.load("./models/final/model24SWINFINAL.h5").to(device)
modelC = torch.load("./models/final/model5SwinSEGLUNG.h5").to(device)
modelB = torch.load("./models/final/modelSwinVOLUME-SEG.h5").to(device)
model = SegEnsemble(modelA, modelB, modelC).to(device)

##############################################################
"""
#visualize
for val_data in train_loader:
    val_inputs, val_labels = (
        val_data["image"].to(device),
        val_data["label"].to(device),
    )
    break

print(val_inputs[0][0].shape)
print(val_labels[0][0].shape)

for i in range (20):
    plt.imsave(f"./images/slice{i}.png",val_inputs[0][0].cpu()[:, :, 90+i], cmap="gray")

##############################################################


# implementing the early stopping
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    torch.save(state, checkpoint_dir)
"""
###################################################################
model = torch.load("./models/final/model24SWINFINAL.h5").to(device)
loss_function = DiceCELoss(to_onehot_y=True,softmax = True, include_background=False)

"""
model = UNet(
    spatial_dims=3,
    in_channels=1, # hard labeling
    out_channels=2, # soft labeling
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    dropout =0.2,
    norm=Norm.BATCH,
).to(device)


model = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
#loss_function = DiceCELoss()
loss_function = DiceCELoss(to_onehot_y=True,softmax = True, include_background=False)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4,amsgrad=True)
early_stopper = EarlyStopper(patience=3, min_delta=0.3)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

model = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4,amsgrad=True)

model, optimizer, epochs = load_ckp("./models/model20SWINFINALCKP.h5",model,optimizer)

loss_function = DiceCELoss(to_onehot_y=True,softmax = True, include_background=False)
early_stopper = EarlyStopper(patience=3, min_delta=0.3)
"""
########################################################


total_time = 0
val_interval = 1 #not a large dataset, so it is fine
print_interval = 1
epoch_loss_values = []
losses_validation = []


model.eval()
with torch.no_grad():
    loss_val = 0
    counter = 1
    for val_data in val_loader:
        val_inputs, val_labels = (
            val_data["image"].to(device),
            val_data["label"].to(device),
        )
        #get the loss to the validation set
        outputs = model(val_inputs)
        loss_val += loss_function(outputs, val_labels).item()
        print(f"val_loss: {loss_val/counter}")
        #print(find_largest_containing_circle(val_inputs, (1.5, 1.5, 2.0)))
        counter+=1

    loss_val_avg = loss_val / len(val_loader)
    losses_validation.append(loss_val_avg)
            
    print(f"validation average loss: {loss_val_avg:.4f}")
    f = open("valLossesMEGAZORD.txt", "a")
    f.write(f"validation average loss: {loss_val_avg:.4f}\n")
    f.close()

