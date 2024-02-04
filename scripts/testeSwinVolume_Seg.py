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
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

data_dir = "/tsi/data_education/data_challenge"
train_images = sorted(glob.glob(os.path.join(data_dir, "train/volume", "*.nii*")))
#train_labels = sorted(glob.glob(os.path.join(data_dir, "train/seg", "*.nii*")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "train/lungs_seg", "*.nii*")))
data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
train_files, val_files = data_dicts[:-50], data_dicts[-50:] #no paper, ele treinou pra 16 e testou em 4
print(train_labels)


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

###############################################################
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
    plt.imsave(f"./images/gt{i}.png",val_labels[0][0].cpu()[:, :, 90+i], cmap="gray")
"""


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

###################################################################
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
"""
model = SwinUNETR(img_size=(192,192,192), in_channels=1, out_channels=2, use_checkpoint=True,attn_drop_rate=0.2, dropout_path_rate=0.2,drop_rate=0.2).to(device)
#loss_function = DiceCELoss()
loss_function = DiceCELoss(to_onehot_y=True,softmax = True, include_background=False)
optimizer = torch.optim.AdamW(model.parameters(), 1e-4,amsgrad=True)
early_stopper = EarlyStopper(patience=3, min_delta=0.3)


total_time = 0
max_epochs = 20
val_interval = 3 #not a large dataset, so it is fine
print_interval = 1
epoch_loss_values = []
losses_validation = []

oneHot = AsDiscrete(threshold=0.5)

for epoch in range(max_epochs):
    start_time = datetime.now()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train() # tell Dropout and BatchNorm to work bcs it is training
    epoch_loss = 0
    step = 0

    # getting data for each batch
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        #print("labels shape antes",labels.size)
        #print("labels antes",labels[0,0,:,0,0])
        labels = oneHot(labels)
        #print("volume shapes",inputs.shape)
        #print("labels shape depois",labels.size)
        #print("labels depois",labels[0,0,:,0,0])

        #print("chegou")
        # normal pipeline
        optimizer.zero_grad()
        #print("aqui")
        outputs = model(inputs)
        #print("deppois do outpit")
        loss = loss_function(outputs, labels)
        #print("loss")
        loss.backward()
        optimizer.step()
       

        #loss.item -> each batch loss
        epoch_loss += loss.item()
        if (epoch) % print_interval == 0:
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")

    # measuring time
    actual_time = datetime.now() - start_time
    print(f"time to train this epoch: {actual_time}")
    total_time += actual_time.total_seconds()
    # saving the loss for the actual epoch
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        model.eval() # tell Dropout and BatchNorm to "turn off" bcs I am evaluating the model
        with torch.no_grad():
            loss_val = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_labels = oneHot(val_labels)
                #get the loss to the validation set
                outputs = model(val_inputs)
                loss_val = loss_function(outputs, val_labels).item()

            loss_val_avg = loss_val / val_loader.batch_size
            losses_validation.append(loss_val_avg)
            
            print(f"validation average loss: {loss_val_avg:.4f}")
            
            f = open("valLossesModelSwinVOLUME-SEG.txt", "a")
            f.write(f"validation average loss: {loss_val_avg:.4f}")
            f.close()
            
            if early_stopper.early_stop(loss_val_avg):
                print("early stopped!")
                break

    #salvar checkpoint
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    }
    save_ckp(checkpoint, True, "./models/modelSwinVOLUME-SEG.pt", "./models/modelSwinVOLUME-SEG.h5")
    
    #via das duvidas se bugar o checkpoint...
    model.eval()
    torch.save(model, "modelSwinVOLUME-SEG.h5")

    if epoch+1 == 1 or epoch+1 == 5 or epoch+1 == 10 or epoch+1 == 15 or epoch+1 == 20:
        model.eval()
        pathName = "./models/"+"model" + str(epoch+1) + "SwinVOLUME-SEG.h5"
        torch.save(model,pathName)
