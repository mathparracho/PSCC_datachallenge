from monai.utils import first, set_determinism 
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
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
    ToTensor,
    Resized,
    FillHoles,
    RemoveSmallObjects,
    KeepLargestConnectedComponent
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference, SimpleInferer
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
import nibabel as nib
import numpy as np
from monai.networks.nets import UNet, SwinUNETR
import torch.nn as nn
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

seed = 22
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

data_dir = "/tsi/data_education/data_challenge"
train_images = sorted(glob.glob(os.path.join(data_dir, "train/volume", "*.nii*")))
#train_images = sorted(glob.glob(os.path.join(data_dir, "train/lungs_seg", "*.nii*")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "train/seg", "*.nii*")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "train/lungs_seg", "*.nii*")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[0:3], data_dicts[90:110] #no paper, ele treinou pra 16 e testou em 4
print(train_labels)



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
        Resized(keys=["image", "label"], spatial_size=(192,192,192))
    ]
)



# 16 images -> transform
train_ds = Dataset(data=train_files, transform=train_transforms)

# 16 images -> transform + data augmentation
#augm_ds= Dataset(data=train_files, transform=[train_transforms, augm_transforms])

# Not declararing the batch size -> whole dataset -> 16(data) + 16(data augmentation) = 32 images per epoch
# this means 1 batch = 32 images
#train_ds = ConcatDataset([train_ds, augm_ds])

train_loader = DataLoader(train_ds)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds)


print("\n\n\n\n\n\n")
print(train_loader)
print("\n\n\n\n\n\n")
##############################################


# rodar so se for importar o modelo ja treinado
#model = torch.load("./models/final/model24SWINFINAL.h5").to(device)
#loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#optimizer = torch.optim.Adam(model.parameters(), 1e-4)

"""
def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

modelVolSeg = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
optimizerVolSeg = torch.optim.AdamW(modelVolSeg.parameters(), 1e-4,amsgrad=True)

modelVolSeg, optimizerVolSeg, epochsVolSeg = load_ckp("./models/model50SwinT2Ckp.pt",modelVolSeg,optimizerVolSeg)

modelVolLungSeg = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
optimizerVolLungSeg = torch.optim.AdamW(modelVolLungSeg.parameters(), 1e-4,amsgrad=True)

modelVolLungSeg, optimizerVolLungSeg, epochsVolLungSeg = load_ckp("./models/modelSwinVOLUME-SEG.pt",modelVolLungSeg,optimizerVolLungSeg)

modelLungSegSeg = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
optimizerLungSegSeg = torch.optim.AdamW(modelLungSegSeg.parameters(), 1e-4,amsgrad=True)
modelLungSegSeg, optimizerLungSegSeg, epochsLungSegSeg = load_ckp("./models/modelSwinSEGLUNG.pt",modelLungSegSeg,optimizerLungSegSeg)


"""
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

model = torch.load("./models/final/model15UNETFINAL.h5").to(device)

###########################################
images = []
outputs = []
labels = []

oneHot = AsDiscrete(threshold=0.5)


for val_data in train_loader:
    val_inputs, val_labels = (
        val_data["image"].to(device),
        val_data["label"].to(device),
        #val_data["image"].cpu(),
        #val_data["label"].cpu(),
    )
    #print(f"valLabelsShape--- {val_labels.shape}")
    val_labels = oneHot(val_labels)
    #print(1 in val_labels[0,:,:,:,:])

    

    print("prevendo")
    images.append(val_inputs)
    output = model(val_inputs)
    

    ############################
    print(f"outputShape {output.shape}")
    print(f"torchargmax {torch.argmax(output.cpu(), dim=1).shape}")
    print(torch.argmax(output.cpu(), dim=1)[0,0,0,108])
    print(output[:,:,0,0,0])

    """
    postProcessing1 = Compose(
            [AsDiscrete(argmax=True),
            #FillHoles(),
            RemoveSmallObjects(min_size=3)
                ]
            )
    output = postProcessing1(output)
    """

    output = torch.argmax(output.cpu(), dim=1)
    postProcessing2 = Compose(
            [#AsDiscrete(threshold=0.5),
            FillHoles(),
            KeepLargestConnectedComponent()
            #RemoveSmallObjects(min_size=5)
            ]
            )
    output = postProcessing2(output)

    print("##############")
    print(f"postprocessingMOnai {output.shape}")
    #print(1 in output.cpu()[:,:,108])
    print(1 in output[0, 0, :, 108])
    #print(output[:,:,0,0,0])
    ###########################


    #print(1 in torch.argmax(output.cpu(), dim=1)[0, :, :, 108])

    outputs.append(output)
    labels.append(val_labels)
####################################################



import matplotlib.pyplot as plt
import torch

# Assuming outputs, labels, and images are your variables
output = outputs[0]
label = labels[0][0][0]
image = images[0][0][0]

print("-------",output.shape)

# Set the slice range from 45 to 50
slice_start = 100
slice_end = 110

# Calculate the number of slices
num_slices = slice_end - slice_start

# Create a subplot with 'num_slices' rows and 3 columns
fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5 * num_slices))

for i in range(num_slices):
    # Increment the slice index
    current_slice = slice_start + i
    
    # Plot the Model Output
    #axes[i, 0].imshow(torch.argmax(output.cpu(), dim=1)[0, :, :, current_slice])
    axes[i, 0].imshow(output[0, :, :, current_slice].cpu())
    axes[i, 0].set_title(f'Model Output - Slice {current_slice}')

    # Plot the Label
    axes[i, 1].imshow(label.cpu()[:, :, current_slice])
    axes[i, 1].set_title(f'Label - Slice {current_slice}')

    # Plot the Input Image
    axes[i, 2].imshow(image.cpu()[:, :, current_slice])
    axes[i, 2].set_title(f'Input Image - Slice {current_slice}')

    print(f"salvando imagem {i}")

# Adjust layout for better visualization
plt.tight_layout()

# Save the plots as PNG files
for i in range(num_slices):
    current_slice = slice_start + i
    fig.savefig(f"./images/slice_{current_slice}_plot15UNETFINAL.png")

# Close the figure
plt.close(fig)
