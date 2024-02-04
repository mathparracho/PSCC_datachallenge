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
    ToTensor
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


device = "cuda" if torch.cuda.is_available() else "cpu"


test_images_dir = "/home/ids/ext-1437/project/data/train/volume"

test_images = sorted(glob.glob(os.path.join(test_images_dir,"*.nii")))
test_data = [{"image": image} for image in test_images]

##TRANFSORMS##
#################################################

testD_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        DivisiblePadd(["image"], 16)
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
        AsDiscreted(keys="pred", argmax=True,threshold = 5),
        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir="./data/test/predictions1B", resample=False),
    ]
)

"""
test_transforms = Compose(
    [ 
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        CropForeground(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        DivisiblePad(16)
    ]
)

simple_transform = Compose([ToTensor(), LoadImage(image_only=True), EnsureChannelFirst()])
"""

####################################################
test_org_ds = Dataset(data=test_data, transform=testD_transforms)
test_org_loader = DataLoader(test_org_ds, batch_size=1)
###########################################################


model = torch.load("model10secondTry.h5").to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")
inferer = SimpleInferer()
model.eval()

results = []
counter = 0

"""
def sigmoid(result):
    result = result.detach().cpu().numpy()
    for x in range(result.shape[1]):
        for y in range(result.shape[2]):
            for z in range(result.shape[3]):
                voxelLabels = result[:,x,y,z]
                voxelLabelsSigmoid = []
                # Aplicar a função sigmoid
                for i in voxelLabels:
                    probabilities = 1 / (1 + np.exp(-i))
                    voxelLabelsSigmoid.append(probabilities)
                if voxelLabelsSigmoid[0] > voxelLabelsSigmoid[1]:
                    voxelOutput = 0
                else:
                    voxelOutput = 1
                result[:,x,y,z] = voxelOutput
    return result
"""

with torch.no_grad():
    for test_data in test_org_loader:
        counter += 1
        test_inputs = test_data["image"].to(device)
        test_data["pred"] = inferer(test_inputs,  model)
        print(f"prevendo {counter}")

        test_data = [i for i in decollate_batch(test_data)]
        #test_data = [post_transforms(i) for i in decollate_batch(test_data)]
        
        test_output = from_engine(["pred"])(test_data)
        for i, result in enumerate(test_output):
            print(result[:,0,0,0])
            binaryResult = np.zeros((result.shape[1],result.shape[2],result.shape[3]),dtype=int)
            binaryResult[:,:,:] = np.argmax(result.detach().cpu().numpy(),axis=0)
            print(1 in binaryResult[:,:,:])


        """
        results.extend(from_engine(["pred"])(test_data))
        
        for i, result in enumerate(results):
            #print(sigmoid(result))
            result = result.detach().cpu().numpy()
            print(result.shape)
            binaryResult = np.zeros((result.shape[1],result.shape[2],result.shape[3]),dtype=int)
            print(binaryResult)
            binaryResult[:,:,:] = np.argmax(result,axis=0)
            print(1 in binaryResult)
        """


        """
        for i, result in enumerate(results):
            
            print("############################################")
            print(result.detach().cpu().numpy()[:,0,0,0])
            
            voxelLabels = list(result.detach().cpu().numpy()[:,0,0,0])
            voxelLabelsSigmoid = []
            # Aplicar a função sigmoid
            for i in voxelLabels:
                probabilities = 1 / (1 + np.exp(-i))
                voxelLabelsSigmoid.append(probabilities)
            if voxelLabelsSigmoid[0] > voxelLabelsSigmoid[1]:
                voxelOutput = 0
            else:
                voxelOutput = 1
            print(voxelLabelsSigmoid)
        """    

            
         
        """
        for i, result in enumerate(results):

            print("############################################")
            print(result.detach().cpu().numpy())
            # Aplicar softmax para obter as probabilidades
            probs = np.exp(result.detach().cpu().numpy())
            probs /= np.sum(probs, axis=-1, keepdims=True)

            # Definir um limiar (threshold)
            threshold = 0.5  # Você pode ajustar o limiar conforme necessário

            # Binarizar as saídas com base no limiar
            binary_output = (probs[..., 1] > threshold).astype(int)
            print(binary_output)
            print(1 in binary_output[0][0])
        """

for i, result in enumerate(results):
    output_filename = os.path.join("/home/ids/ext-1437/project/data/test/predictions1B" , f"result_{i}.nii.gz")
    
    
    #######################################
    #seja o que deus quiser
    # Aplicar a função sigmoid
    probabilities = 1 / (1 + np.exp(-result.detach().cpu().numpy()))

    # Aplicar o threshold de 0.5 para obter rótulos binários
    threshold = 0.5
    binary_labels = (probabilities > threshold).astype(int)
    ##########################################
    
    
    """
    #########################################
    #seja o que deus quiser 2
    # Aplicar softmax para obter as probabilidades
    probs = np.exp(result.detach().cpu().numpy())
    probs /= np.sum(probs, axis=-1, keepdims=True)

    # Definir um limiar (threshold)
    threshold = 0.5  # Você pode ajustar o limiar conforme necessário

    # Binarizar as saídas com base no limiar
    binary_output = (probs[..., 1] > threshold).astype(np.int)
    """
    #nib.save(nib.Nifti1Image(-result.detach().cpu().numpy(), affine=None), output_filename)
    nib.save(nib.Nifti1Image(binary_labels, affine=None), output_filename)

"""
for filename in os.listdir(test_images_dir):
    img_path = "/home/ids/ext-1437/project/data/test/volume/" + filename
    img = nib.load(img_path)
    data = simple_transform(img_path).to(device)
    with torch.no_grad():
        pred = inferer(inputs=data, network=model)
        print(f'prevendo img {filename}')
        prediction_image = nib.Nifti1Image(pred, affine=img.affine)
        nib.save(prediction_image, f'/home/ids/ext-1437/project/data/test/predictions/{filename}pred.nii')
"""

