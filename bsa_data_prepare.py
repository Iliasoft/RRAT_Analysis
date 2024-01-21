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
from bsa_point_file import read_point_file
from bsa_headers import *

TRUE_LABEL_KEY = "true_label"

IMG_NAMES_KEY = 'img_names'
LABELS_KEY = "labels"
IMAGE_NAMES_KEY = "fns"
LEGACY_PREDS_KEY = "legacy_prediction"
ID_KEY = "id"

WHITE_NOISE_AUGMENTATION_LEN = 0

EMBEDDINGS_FN = "embeddings.pkl"
AUGMENTED_EMBEDDINGS_FN = "embeddings_augmented.pkl"
AUGMENTED_EMBEDDINGS_MAP_FN = "embeddings_augmented_mapping.pkl"
DATA_EXTRACT_FN = "metadata.pkl"
WHITE_NOISE_FN = "white_noise.pkl"


stats = {}
stats["mean"] = []
stats["std"] = []
stats["min"] = []
stats["max"] = []
stats["channel_means"] = []
stats["len"] = []


def create_picture(data, fn):

    mm, mx = np.min(data.ravel()), np.max(data.ravel())
    multiplicator = 255 / (mx - mm)
    channels, npoints = data.shape[1], data.shape[0]
    img = np.empty((channels, npoints), dtype=np.uint8)
    for y in range(channels):
        for x in range(npoints):
            img[y, x] = 255 - int(multiplicator * (data[x, y] - mm))
    png.from_array(img, 'L').save(fn)

# если образец короче чем нужно мы его надстраиваем слева и справа
# если образец длинее чем нужно мы его обрезаем слева и справа

def augmentation_scale_no_result(data, scaling_factor):
    augmented = data
    if augmented.shape[0] > MATRIX_LEN:
        #fragment_start_position = int(augmented.shape[0] / 2) - int(MATRIX_LEN / 2)
        #augmented = augmented[fragment_start_position:fragment_start_position + MATRIX_LEN]
        augmented = scipy.ndimage.zoom(augmented, (MATRIX_LEN / data.shape[0], 1), order=2)

    elif augmented.shape[0] < MATRIX_LEN:
        left_positions_to_extend = int((MATRIX_LEN - augmented.shape[0])/ 2)
        right_positions_to_extend = MATRIX_LEN - left_positions_to_extend - augmented.shape[0]

        augmented_left_augmentation = np.empty((0, 32), dtype=np.float32)
        for i in range(left_positions_to_extend):
            x = int(random.uniform(0, augmented.shape[0]))
            augmented_left_augmentation = np.concatenate((augmented_left_augmentation, np.reshape(augmented[x], (1, 32))))

        augmented_right_augmentation = np.empty((0, 32), dtype=np.float32)
        for i in range(right_positions_to_extend):
            x = int(random.uniform(0, augmented.shape[0]))
            augmented_right_augmentation = np.concatenate((augmented_right_augmentation, np.reshape(augmented[x], (1, 32))))

        augmented = np.concatenate((augmented_left_augmentation, augmented, augmented_right_augmentation))
    else:
        pass

    #s = np.std(augmented)
    #m = np.mean(augmented)
    #augmented = (augmented - m) / s

    return augmented.ravel()


def generate_shift(data, v=0.2):
    shift_factor_x = int(
        random.uniform(
            -math.ceil(0.5 * v * data.shape[0]),
            math.ceil(0.5 * v * data.shape[0])
        )
    )

    augmented = np.roll(data, shift_factor_x, axis=0)
    shift_factor_y = int(
        random.uniform(
            -math.ceil(v * data.shape[1]),
            math.ceil(v * data.shape[1])
        )
    )
    
    return np.roll(augmented, shift_factor_y, axis=1)


if __name__ == '__main__':

    _, img_data, augmented, data = read_point_file("F:/BSA/N1-N2-Jan-2021, N0-Dec-2020/401/000002.pnt", 0)
    create_picture(data, "F:/BSA/dd.png")
    exit(0)
    #################
    augmentation_factor = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    dst_folder = sys.argv[4] if len(sys.argv) > 4 else sys.argv[1]

    df = pd.read_csv(os.path.join(sys.argv[1], sys.argv[2]))
    df_labeled = df[df[TRUE_LABEL_KEY].notnull()]
    df_labeled_true = df_labeled.loc[df_labeled[TRUE_LABEL_KEY] == 1]
    df_labeled_false = df_labeled.loc[df_labeled[TRUE_LABEL_KEY] == 0]
    print(f"Dataset: {sys.argv[2]}, Total: {len(df)}, Labeled: {len(df_labeled)} 1: {len(df_labeled_true)} 0: {len(df_labeled_false)}")
    if len(sys.argv) > 3:
        df = df_labeled
        print("Assuming training part only to be embedded due to non-default augmentation factor")

    embeddings = torch.empty(len(df) + WHITE_NOISE_AUGMENTATION_LEN, MATRIX_LEN * MATRIX_WIDTH, dtype=torch.float32)
    embeddings_augmented = torch.empty((int(len(df_labeled_true) * augmentation_factor), MATRIX_LEN * MATRIX_WIDTH), dtype=torch.float32)

    gt_labels = np.zeros(len(df) + WHITE_NOISE_AUGMENTATION_LEN, dtype=np.uint8)
    # unscaled_data = np.zeros(0, dtype=np.float32)

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

        gt_labels[embedding_seq] = int(row[2]) if row[2] == row[2] else 3
        legacy_prediction.append(round(row[1], 2))

        file_name += row[0][2:].replace("-", "/") + ".pnt"

        _, img_data, augmented, _ = read_point_file(file_name, augmentation_factor if gt_labels[embedding_seq] == 1 else 0)
        embeddings[embedding_seq] = img_data
        
        for a in augmented:
            embeddings_augmented[aug_seq] = a
            aug_seq += 1
            augmented_embeddings_mapping_to_original.append(embedding_seq)

        file_names.append(row[0])
        embedding_seq += 1

    with open(os.path.join(sys.argv[1], WHITE_NOISE_FN), 'rb') as f:
        white_noise_hour = pickle.load(f)

    for i in range(WHITE_NOISE_AUGMENTATION_LEN):

        gt_labels[embedding_seq] = 0
        object_ids.append(None)
        legacy_prediction.append(None)
        file_names.append(None)
        fragment_start_position = int(i * (len(white_noise_hour) / WHITE_NOISE_AUGMENTATION_LEN))

        data_scaled = white_noise_hour[fragment_start_position:fragment_start_position + MATRIX_LEN, :].ravel()
        s = np.std(data_scaled)
        m = np.mean(data_scaled)
        data_scaled_normalized = (data_scaled - m) / s
        embeddings[embedding_seq] = torch.tensor(data_scaled_normalized)
        embedding_seq += 1

    with open(os.path.join(dst_folder, EMBEDDINGS_FN), 'wb') as f:
        pickle.dump(embeddings, f)

    with open(os.path.join(dst_folder, DATA_EXTRACT_FN), 'wb') as f:
        pickle.dump({FILE_NAME_KEY: object_ids, LEGACY_PREDS_KEY: legacy_prediction, LABELS_KEY: gt_labels}, f)

    with open(os.path.join(dst_folder, AUGMENTED_EMBEDDINGS_FN), 'wb') as f:
        pickle.dump(embeddings_augmented, f)

    with open(os.path.join(dst_folder, AUGMENTED_EMBEDDINGS_MAP_FN), 'wb') as f:
        pickle.dump(augmented_embeddings_mapping_to_original, f)
