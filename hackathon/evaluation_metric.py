import numpy as np
import pandas as pd


def rle2mask(mask_rle: str, shape, label=1):
    """
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    if mask_rle == 0:
        return np.zeros(shape)
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1] * shape[2], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = label
    return img.reshape(shape)  # Needed to align to RLE direction


def custom_dice(y_true, y_pred, label=1):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    label_positions = y_true_flat == label

    true_positives = np.sum(label_positions * y_pred_flat)
    false_positives = np.sum((1 - label_positions) * y_pred_flat)
    false_negatives = np.sum(label_positions * (1 - y_pred_flat))

    epsilon = 1e-7
    dice = (2.0 * true_positives + epsilon) / (
        2.0 * true_positives + false_positives + false_negatives + epsilon
    )

    return dice


def score(
    solution: pd.DataFrame, submission: pd.DataFrame
) -> float:
    """
    ======================================================================================================================#

    inputs:
    For seg -> compute DICE
    for recist -> compute MAE
    for volume -> compute MAE

    Return Mean of the three metrics

    Doctest:
    #>>> import numpy as np
    #>>> import pandas as pd
    #>>> rle_pred = ['1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1', '1 2 4 2 7 2 12 1 14 3 18 1']
    #>>> recist_pred = [20,10,30,40]
    #>>> vol_pred = [100,200,300,400]
    #>>> shapes = ['(3, 3, 2)','(3, 3, 2)','(3, 3, 2)','(3, 3, 2)']
    #>>> rle_true = ['4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2', '4 1 7 1 9 2 13 2 16 2']
    #>>> recist_true = [5,25,30,89]
    #>>> vol_true = [100,400,368, 472]
    #>>> sub = pd.DataFrame({'id':[1,2,3,4], 'rle': rle_pred, 'volume': vol_pred, 'recist':recist_pred, 'data_shape': shapes})
    #>>> sol = pd.DataFrame({'id':[1,2,3,4], 'rle': rle_true, 'volume': vol_true, 'recist':recist_true, 'data_shape': shapes})
    #>>> score(sub, sol, 'id')
    """

    # Initialize Dice computer
    #     dice = Dice(average='macro', num_classes = 2)
    # Iterate throught rows of dataframe
    for subtuple, soltupe in zip(submission.iterrows(), solution.iterrows()):
        sub, sol = subtuple[1], soltupe[1]
        # Convert rle to mask
        sub_array = rle2mask(
            sub["rle"], np.fromstring(sub["data_shape"][1:-1], sep=",", dtype="int")
        )
        sol_array = rle2mask(
            sol["rle"], np.fromstring(sol["data_shape"][1:-1], sep=",", dtype="int")
        )
        sol_array = np.expand_dims(sol_array, axis=(0, 1))
        sub_array = np.expand_dims(sub_array, axis=(0, 1))
        # Compute Dice, recist and volume and store them
        seg_error = np.append(
            seg_error, custom_dice(y_true=sol_array, y_pred=sub_array)
        )
        recist_error = np.append(
            recist_error, np.abs(sub["recist"] - sol["recist"]) / sol["recist"]
        )
        vol_error = np.append(
            vol_error, np.abs(sub["volume"] - sol["volume"]) / sol["volume"]
        )
    # Rescale vol_error and recist_error to be the same order of magnitude by tresholding
    recist_error = np.where(recist_error > 1, 1, recist_error)
    vol_error = np.where(vol_error > 1, 1, vol_error)
    # Make sure error is maxed if array contains a repetition of the same element.
    score = np.mean([1 - np.mean(seg_error), np.mean(recist_error), np.mean(vol_error)])

    return score
