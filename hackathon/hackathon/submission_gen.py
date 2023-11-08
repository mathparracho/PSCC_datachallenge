import os
import glob

import nibabel as nib
import pandas as pd
from post_proc import mask2rle, find_largest_containing_circle

def submission_gen(predpath: list, outputpath: str):
  '''
  Create a submission csv from a path of segmentation prediction.
  predpath: Path of your fodler containing the predictions
  /!\ The path should directly contain the .nii.gz files
  outputpath: Path of where the csv will be saved
  '''
  pred_files = glob.glob(f"{predpath}/*")
  rle_list = []
  recist_list = []
  volume_list = []
  patient_id_list = []
  shape_list = []
  for file in sorted(pred_files):
      img = nib.load(file)
      data = img.get_fdata()
      shape_list.append(data.shape)
      rle_list.append(mask2rle(data))
      recist, predicted_volume, largest_circle, largest_slice = find_largest_containing_circle(data, img.header['pixdim'][1:4])
      recist_list.append(recist)
      volume_list.append(predicted_volume)
      patient_id_list.append(file.split('/')[-1].split('_')[0])
  df = pd.DataFrame({'id':patient_id_list,
                  'rle': rle_list,
                  'recist': recist_list,
                  'volume': volume_list,
                  'data_shape': shape_list})
  df.to_csv(outputpath, index = False)
  return f"submission file saved at {outputpath}"
