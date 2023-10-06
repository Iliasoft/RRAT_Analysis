import os
import random
import sys
import numpy as np
import torch
import pandas as pd
import pickle
import random
import png
import scipy.ndimage
from tqdm import tqdm
import math

TRUE_LABEL_KEY = "true_label"
FILE_NAME_KEY = "file_name"
ERROR_CODE_KEY = "error"
IMG_NAMES_KEY = 'img_names'
LABELS_KEY = "labels"
IMAGE_NAMES_KEY = "fns"
LEGACY_PREDS_KEY = "legacy_prediction"
ID_KEY = "id"

MATRIX_LEN = 148
MATRIX_WIDTH = 32

EMBEDDINGS_FN = "embeddings.pkl"
AUGMENTED_EMBEDDINGS_FN = "embeddings_augmented.pkl"
AUGMENTED_EMBEDDINGS_MAP_FN = "embeddings_augmented_mapping.pkl"
DATA_EXTRACT_FN = "metadata.pkl"

def read_binary_file(file_name, augmented_points_num=0):
    N_EXPECTED_HEADERS = 24
    NL_DELIMITER_CODE = 10
    NUM_PAR_KEY = "numpar"
    NUM_POINTS_KEY = "npoints"
    NUM_RAYS_KEY = "rays"
    NUM_BANDS_KEY = "nbands"

    try:
        with open(file_name, mode='rb') as file:
            file_data = np.fromfile(file, np.dtype('B'))

    except FileNotFoundError:
        return {ERROR_CODE_KEY: 0}, None
    except IOError:
        return {ERROR_CODE_KEY: 1}, None

    nl_indexes = np.insert(np.nonzero(np.equal(file_data[:1000], NL_DELIMITER_CODE))[0], 0, [-1])
    if len(nl_indexes) < N_EXPECTED_HEADERS + 1:
        return {ERROR_CODE_KEY: 2}, None  # unknown PNT header format

    int_metadata = [NUM_PAR_KEY, "dispersion", "module", "modulus", NUM_BANDS_KEY, "point", "ray", NUM_RAYS_KEY, NUM_POINTS_KEY]
    float_metadata = ["fbands", "wbands", "snr", "tresolution", "fcentral", "wb_total"]
    date_str_metadata = ["date_begin", "time_begin"]
    metadata = {FILE_NAME_KEY: file_name}

    for header_seq in range(N_EXPECTED_HEADERS):
        h = file_data[nl_indexes[header_seq] + 1:nl_indexes[header_seq + 1]].tobytes().decode("utf-8").split()
        if len(h) < 2:
            return {ERROR_CODE_KEY: 3}  # "invalid header"
        elif len(h) == 2:
            try:
                metadata[h[0]] = int(h[1]) if h[0] in int_metadata else (float(h[1]) if h[0] in int_metadata else h[1])
            except ValueError:
                return {ERROR_CODE_KEY: 4}  # "invalid header"

        elif len(h) > 2 and h[0] in float_metadata:
            metadata[h[0]] = list(map(float, h[1:]))
        elif len(h) > 2 and h[0] in date_str_metadata:
            metadata[h[0]] = " ".join(h[1:])
        else:
            metadata[h[0]] = h[1:]

    if metadata[NUM_PAR_KEY] != 24 or metadata[NUM_RAYS_KEY] != 1 or metadata[NUM_BANDS_KEY] != 32:
        return {ERROR_CODE_KEY: 5}, None  # unsupported type if PNT file

    spectrum_data_start_position = nl_indexes[N_EXPECTED_HEADERS] + 1
    ###################
    npoints = metadata[NUM_POINTS_KEY]
    channels = metadata[NUM_BANDS_KEY] + 1
    data = np.reshape(
        file_data[spectrum_data_start_position:].view('float32'),
        (npoints, channels)  # (len(modulus), channels, rays, npoints)
    )
    # we don't need the Mean channel
    data = data[:, :MATRIX_WIDTH]

    if data.shape[0] > MATRIX_LEN:
        # calculate how much data will be cropped
        l_crop = int((data.shape[0] - MATRIX_LEN)/2)
        r_crop = data.shape[0] - MATRIX_LEN - l_crop
        data = data[l_crop:data.shape[0] - r_crop, :]

    # basic data cleanup
    Q1 = -0.62#np.percentile(data, 3, method='midpoint')
    Q3 = 0.62# np.percentile(data, 97, method='midpoint')

    indx = np.where(data > Q3)
    data[indx] = Q3

    indx = np.where(data < Q1)
    data[indx] = Q1

    scale_factor = MATRIX_LEN / data.shape[0]
    fixed_size_data = augmentation_scale(data, scale_factor).ravel()

    augmented_points = []

    for aug in range(augmented_points_num):

        augmented_points.append(
            torch.tensor(generate_augmented_point(data), dtype=torch.float32)
        )

    return metadata, torch.tensor(fixed_size_data, dtype=torch.float32), augmented_points, data


def create_picture(data, fn):

    mm, mx = np.min(data.ravel()), np.max(data.ravel())
    multiplicator = 255 / (mx - mm)
    channels, npoints = data.shape[1], data.shape[0]
    img = np.empty((channels, npoints), dtype=np.uint8)
    for y in range(channels):
        for x in range(npoints):
            img[y, x] = 255 - int(multiplicator * (data[x, y] - mm))
    png.from_array(img, 'L').save(fn)


def generate_augmented_point(data):
    augmented = data

    scale_factor = MATRIX_LEN / augmented.shape[0]
    augmented = augmentation_scale(augmented, scale_factor)

    return augmented.ravel()


def augmentation_scale(source_data, scaling_factor):

    return scipy.ndimage.zoom(source_data, (scaling_factor, 1), order=1)


def generate_shift(data, v=0.05):

    shift_factor_x = int(
        random.uniform(
            -math.ceil(v * data.shape[0]),
            math.ceil(v * data.shape[0])
        )
    )
    augmented = np.roll(data, shift_factor_x, axis=0)

    shift_factor_y = int(
        random.uniform(
            -math.ceil(0.5 * v * data.shape[1]),
            math.ceil(0.5 * v * data.shape[1])
        )
    )

    return np.roll(augmented, shift_factor_y, axis=1)


if __name__ == '__main__':

    augmentation_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    df = pd.read_csv(os.path.join(sys.argv[1], sys.argv[2]))
    df_labeled = df[df[TRUE_LABEL_KEY].notnull()]
    df_labeled_true = df_labeled.loc[df_labeled[TRUE_LABEL_KEY] == 1.0]
    df_labeled_false = df_labeled.loc[df_labeled[TRUE_LABEL_KEY] == 0.0]
    print(f"Dataset: {sys.argv[2]}, Total: {len(df)}, Labeled: {len(df_labeled)} 1: {len(df_labeled_true)} 0: {len(df_labeled_false)}")

    embeddings = torch.empty(len(df), MATRIX_LEN * MATRIX_WIDTH, dtype=torch.float32)
    embeddings_augmented = torch.empty((int(len(df_labeled_true) * augmentation_factor), MATRIX_LEN * MATRIX_WIDTH), dtype=torch.float32)

    true_labels = np.empty(len(df), dtype=np.uint8)
    unscaled_data = np.empty(0, dtype=np.float32)

    embedding_seq = 0
    aug_seq = 0
    file_names = []
    object_ids = []
    augmented_embeddings_mapping_to_original = []
    legacy_prediction = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        object_ids.append(row[0])

        file_name = sys.argv[1]
        if row[0][0] == 'H':
            file_name += "/transients2016-high/"
        else:
            file_name += "/transients2016-low/"

        true_labels[embedding_seq] = int(row[2]) if row[2] == row[2] else 3
        legacy_prediction.append(round(row[1], 2))

        file_name += row[0][2:].replace("-", "/") + ".pnt"

        _, img_data, augmented, _ = read_binary_file(file_name, augmentation_factor if true_labels[embedding_seq] else 0)
        embeddings[embedding_seq] = img_data
        
        for a in augmented:
            embeddings_augmented[aug_seq] = a
            aug_seq += 1
            augmented_embeddings_mapping_to_original.append(embedding_seq)

        file_names.append(row[0])
        embedding_seq += 1

    with open(os.path.join(sys.argv[1], EMBEDDINGS_FN), 'wb') as f:
        pickle.dump(embeddings, f)

    with open(os.path.join(sys.argv[1], DATA_EXTRACT_FN), 'wb') as f:
        pickle.dump({FILE_NAME_KEY: object_ids, LEGACY_PREDS_KEY: legacy_prediction, LABELS_KEY: true_labels}, f)

    with open(os.path.join(sys.argv[1], AUGMENTED_EMBEDDINGS_FN), 'wb') as f:
        pickle.dump(embeddings_augmented, f)

    with open(os.path.join(sys.argv[1], AUGMENTED_EMBEDDINGS_MAP_FN), 'wb') as f:
        pickle.dump(augmented_embeddings_mapping_to_original, f)
