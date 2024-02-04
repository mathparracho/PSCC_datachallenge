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
    FillHolesd,
    RemoveSmallObjectsd
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
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

seed = 22
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


#CHECKKKKKKKKKKKKKKKKKKKKKKKK HERE
test_images_dir = "/tsi/data_education/data_challenge/test/volume"

test_images = sorted(glob.glob(os.path.join(test_images_dir,"*.nii*")))
test_data = [{"image": image} for image in test_images]

##TRANFSORMS##
#################################################

testD_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        DivisiblePadd(["image"], 16),
        Resized(keys=["image"], spatial_size=(192,192,192))
    ]
)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=testD_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
        FillHolesd(keys="pred"),
        #RemoveSmallObjectsd(keys="pred",min_size=300),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./data/test/predictionsSWIN12", output_postfix="", resample=False),
    ]
)

# EU TO MALUCOOOOOOOOOO

####################################################
test_org_ds = Dataset(data=test_data, transform=testD_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1)
###########################################################


#model = torch.load("./models/final/model24SWINFINAL.h5").to(device)

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
#model = SegEnsemble(modelA, modelB, modelC).to(device)

model = torch.load("./models/final/model12Swin.h5").to(device)

#########################################################

#loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
#dice_metric = DiceMetric(include_background=False, reduction="mean")
inferer = SimpleInferer()
model.eval()

results = []
counter = 0

with torch.no_grad():
    for test_data in test_org_loader:
        counter += 1
        test_inputs = test_data["image"].to(device)
        test_data["pred"] = inferer(test_inputs,  model)
        print(f"prevendo {counter}")

        test_data = [post_transforms(i) for i in decollate_batch(test_data)]
