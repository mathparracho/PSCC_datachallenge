import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = nib.load(r"/tsi/data_education/data_challenge/train/lungs_seg/LUNG1-002_lungseg.nii.gz")
img2 = nib.load(r"/tsi/data_education/data_challenge/train/volume/LUNG1-002_vol.nii.gz")
img3 = nib.load(r"/tsi/data_education/data_challenge/train/seg/LUNG1-002_seg.nii.gz")

limg = [img,img2,img3]

for k in range(3):

    im = limg[k].get_fdata()
    print(im.shape)

    max = np.max(im)
    print('max', max)
    min = np.min(im)
    print('min', min)
    im = im - np.min(im)
    im = im / np.max(im)
    for i in range(15):
        image = Image.fromarray((im[:,:,40 + i] * 255).astype(np.uint8))
        image.save(f'./lungseg/{k}lung{95+i}.png')

