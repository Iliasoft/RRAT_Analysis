import glob
import os
import sys
import numpy as np
import png
from tqdm import tqdm
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from bsa_train import PDClassifier
from bsa_headers import MATRIX_LEN, MATRIX_WIDTH, LONG_FILE_EXT, LONG_FILE_NAME_ATTRIBUTES_SEPARATOR, POSITIVES_DIR
from bsa_point_file import generate_augmented_point
from bsa_long_file import read_long_file

TRUE_LABEL_KEY = "true_label"
FILE_NAME_KEY = "file_name"
ERROR_CODE_KEY = "error"
IMG_NAMES_KEY = 'img_names'
LABELS_KEY = "labels"
IMAGE_NAMES_KEY = "fns"
LEGACY_PREDS_KEY = "legacy_prediction"
ID_KEY = "id"

limit_by_moule_and_ray = False
limited_module = 4
limited_ray = 7

list_of_long_files = [
    #"010121_23_N1_00.pnthr"
    #"090121_01_N1_00.pnthr"
    #"130823_02_N1_00.pnthr",
    #"130823_10_N1_00.pnthr",
    #"130823_16_N1_00.pnthr",
]


class BSABinFile(Dataset):

    def __init__(self, ray_data, sequencing_window_len, step_size):

        self.sequencing_window_len = sequencing_window_len
        self.ray_data = ray_data
        self.step_size = step_size

    def __len__(self):
        return int((self.ray_data.shape[0] - self.sequencing_window_len) / self.step_size)

    def __getitem__(self, index):
        '''
        if index < 2 or index > self.__len__() - 3:
            return generate_augmented_point(self.ray_data[index * self.step_size:index * self.step_size + self.sequencing_window_len])
        else:
            # взать 2 рамки справа, запрашиваемую рамку и 2 рамки слева

            scale_factor = MATRIX_LEN / self.sequencing_window_len
            augmented = augmentation_scale(
                self.ray_data[(index - 2) * self.step_size:(index + 2) * self.step_size + self.sequencing_window_len],
                scale_factor
            )
            # basic data cleanup
            s = np.std(augmented)
            #print(f"{index * self.step_size} {s:.2f}")
            m = np.mean(augmented)
            augmented = (augmented - m) / s
            augmented = augmented[int(scale_factor * 2 * self.step_size):+ int(scale_factor * self.sequencing_window_len + scale_factor * 2 * self.step_size)]
            return augmented.ravel()
        '''

        return generate_augmented_point(self.ray_data[index * self.step_size:index * self.step_size + self.sequencing_window_len])


def generate_ray_pictures(data, dir):

    if limit_by_moule_and_ray:
        create_picture(
            np.reshape(data[:, limited_module:limited_module + 1, limited_ray:limited_ray + 1, :], (data.shape[0], data.shape[3])),
            os.path.join(dir, f"{limited_module}_{limited_ray}.png")
        )
        return

    for m in range(data.shape[1]):
        for r in range(data.shape[2]):

            create_picture(
                np.reshape(data[:, m:m + 1, r:r + 1, :], (data.shape[0], data.shape[3])),
                os.path.join(dir, f"{m}_{r}.png")
            )


def pre_process_data(data, mean_window_size):
    '''
    create_picture(
        np.reshape(data[:, limited_ray, limited_module], (data.shape[0], data.shape[3])),
        os.path.join("F:/BSA/Test/original_0_6.png")
    )
    '''
    normalize_ray_baseline(data[:, limited_ray, limited_module], mean_window_size)

    '''
    create_picture(
        np.reshape(data[:, limited_ray, limited_module], (data.shape[0], data.shape[3])),
        os.path.join("F:/BSA/Test/normalized_0_6.png")
    )
    '''


def normalize_ray_baseline(data, mean_window_size):

    for t in range(0, data.shape[0] - mean_window_size, mean_window_size):
        means_per_channel = np.mean(data[t:t + mean_window_size], axis=0)
        max_all_channels = np.max(means_per_channel)

        for c in range(data.shape[1]):
            #data[t:t + mean_window_size, c:c + 1] *= max_all_channels / means_per_channel[c]
            data[t:t + mean_window_size, c:c + 1] -= np.median(data[t:t + mean_window_size, c:c + 1])

    return data


def pre_process_data_bl(data, mean_window_size):

    for t in range(0, data.shape[0] - mean_window_size, mean_window_size):
        data[t:t + mean_window_size, :] -= np.mean(data[t:t + mean_window_size, :], axis=0)


def line_f(x, k, b):
    return x * k + b


def pre_process_data_nbl(data, mean_window_size):

    means_by_channel = np.empty((int(data.shape[0] / mean_window_size), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
    k_by_channel = np.empty((int(data.shape[0] / mean_window_size), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
    b_by_channel = np.empty((int(data.shape[0] / mean_window_size), data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)

    for t in range(int(data.shape[0] / mean_window_size)):
        means_by_channel[t] = np.mean(data[t * mean_window_size:(t + 1) * mean_window_size, :], axis=0)
        print(f"{np.max(means_by_channel[t]):.2f}, {t * mean_window_size}")

    for t in range(int(data.shape[0] / mean_window_size) - 1):
        k_by_channel[t] = (means_by_channel[t + 1] - means_by_channel[t]) / mean_window_size
        b_by_channel[t] = means_by_channel[t]

    x = np.arange(mean_window_size)
    for t in range(int(data.shape[0] / mean_window_size)):
        delta = np.array(list(map(line_f, x, np.reshape(k_by_channel[t], (-1, MATRIX_WIDTH)), np.reshape(b_by_channel[t], (-1, MATRIX_WIDTH)))))
        data[t * mean_window_size:(t + 1) * mean_window_size] -= np.reshape(delta, (k_by_channel[t].shape[0], k_by_channel[t].shape[1], -1))


def pre_process_data_adj(data, mean_window_size):

    means_leap_length = np.empty((0), dtype=np.int16)

    last_mean = np.max(np.mean(data[0:mean_window_size, :], axis=0))
    position = mean_window_size
    means_leap_length = np.append(means_leap_length, 0)
    step_size = int(mean_window_size / 4)
    assert not step_size % 2  # step_size must be even number

    while position < data.shape[0]:
        for current_pos in range(position, min([position + mean_window_size + 1, data.shape[0]]), step_size):
            means = np.mean(data[means_leap_length[-1]:current_pos, :], axis=0)

            if abs(np.max(means) - last_mean) / abs(np.max(means)) > 0.03:
                #print("<<>>>", int((current_pos % mean_window_size) / step_size), current_pos, abs(np.max(means) - last_mean) / abs(np.max(means)))
                pass
            elif current_pos + step_size >= min([position + mean_window_size, data.shape[0]]):
                position = current_pos + step_size
            else:
                continue

            last_mean = np.max(means)

            means_leap_length = np.append(means_leap_length, current_pos)
            position = current_pos + step_size
            break

    means_by_channel = np.empty((len(means_leap_length) - 1, data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
    #k_by_channel = np.empty((len(means_leap_length) - 2, data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)
    #b_by_channel = np.empty((len(means_leap_length) - 2, data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)

    for t in range(len(means_by_channel)):
        means_by_channel[t] = np.mean(data[means_leap_length[t]:means_leap_length[t + 1], :], axis=0)

    for t in range(len(means_by_channel) - 1):
        l = int(means_leap_length[t + 1] - (means_leap_length[t + 1] - means_leap_length[t]) / 2)
        r = int(means_leap_length[t + 2] - (means_leap_length[t + 2] - means_leap_length[t + 1]) / 2)
        window_size = r - l
        k_by_channel = (means_by_channel[t + 1] - means_by_channel[t]) / window_size
        b_by_channel = means_by_channel[t]

        x = np.arange(window_size)
        k_for_x = np.full((window_size, k_by_channel.shape[0], k_by_channel.shape[1], k_by_channel.shape[2]), k_by_channel)
        b_for_x = np.full((window_size, b_by_channel.shape[0], b_by_channel.shape[1], b_by_channel.shape[2]), b_by_channel)
        delta = np.array(list(map(line_f, x, k_for_x, b_for_x)))

        data[l:r] -= delta
    # special handling of initial window (since it has no left point
    l = 0
    r = int(means_leap_length[1] / 2)
    window_size = r - l
    k_by_channel = (means_by_channel[1] - means_by_channel[0]) / window_size
    b_by_channel = means_by_channel[0]

    x = np.arange(window_size)
    k_for_x = np.full((window_size, k_by_channel.shape[0], k_by_channel.shape[1], k_by_channel.shape[2]), k_by_channel)
    b_for_x = np.full((window_size, b_by_channel.shape[0], b_by_channel.shape[1], b_by_channel.shape[2]), b_by_channel)
    delta = np.array(list(map(line_f, x, k_for_x, b_for_x)))

    data[l:r] -= delta


def create_picture(data, fn):

    mm, mx = np.percentile(data.ravel(), 0.1), np.percentile(data.ravel(), 99.9)

    data[np.where(data > mx)] = mx
    data[np.where(data < mm)] = mm

    multiplicator = 255 / (mx - mm)
    channels, npoints = data.shape[1], data.shape[0]
    img = np.empty((channels, npoints), dtype=np.uint8)
    for y in range(channels):
        for x in range(npoints):
            img[y, x] = 255 - int(multiplicator * (data[x, y] - mm))
    png.from_array(img, 'L').save(fn)

    return img


def predict_datafile(dst_dir, data, model, min_confidence_for_positive, prediction_window_size, step_size, rightmost_points_to_ignore=-1):
    module_ray_img_dir = os.path.join(dst_dir, POSITIVES_DIR)
    Path(os.path.join(sys.argv[2], module_ray_img_dir)).mkdir(parents=True, exist_ok=True)

    data = data[:-rightmost_points_to_ignore]
    t = tqdm(total=data.shape[1] * data.shape[2])
    for m in range(data.shape[1]):
        for r in range(data.shape[2]):
            if limit_by_moule_and_ray and not (r == limited_ray and m == limited_module):
                t.update(1)
                continue

            predictions = predict_in_ray(
                np.reshape(data[:, m:m + 1, r:r + 1, :], (data.shape[0], data.shape[3])),
                model,
                min_confidence_for_positive,
                prediction_window_size,
                step_size
            )
            t.update(1)
            if len(predictions):

                for p in predictions:
                    create_picture(
                        p[1],
                        os.path.join(dst_dir, f"{m}{LONG_FILE_NAME_ATTRIBUTES_SEPARATOR}{r}{LONG_FILE_NAME_ATTRIBUTES_SEPARATOR}{p[0] * step_size}{LONG_FILE_NAME_ATTRIBUTES_SEPARATOR}{p[2]:.2f}.png")
                    )


def find_suboptimal_frames_in_window(next_frame, ids, frames, step_size, confidences, window_size):
    frames_for_passivation = []
    near = [next_frame]
    near_confidence = [confidences[next_frame]]
    for id in range(next_frame + 1, max(ids) + 1):
        if (frames[id] - frames[next_frame]) * step_size <= window_size:
            near.append(id)
            near_confidence.append(confidences[id])
        else:
            next_frame = id
            break

    #print(near)
    #print(near_confidence)
    if next_frame == ids[-1] or ids[-1] in near:
        next_frame = -1

    max_conf = np.argmax(near_confidence)
    for n in range(len(near)):
        if n != max_conf:
            frames_for_passivation.append(near[n])

    return frames_for_passivation, next_frame


def list_suboptimal_frames(frames, step_size, confidences, window_size):
    if not len(frames):
        return []
    ids = np.arange(len(frames))
    frame_ids_for_passivation = []
    next_frame = ids[0]
    while next_frame != -1:
        frames_for_passivation, next_frame = find_suboptimal_frames_in_window(next_frame, ids, frames, step_size, confidences, window_size)
        frame_ids_for_passivation.extend(frames_for_passivation)

    return frame_ids_for_passivation


def predict_in_ray(data, model, min_confidence_for_positive, prediction_window_size, step_size):
    ds = BSABinFile(data, prediction_window_size, step_size)

    dl = DataLoader(
        dataset=ds,
        batch_size=int(len(ds) / 2.9),
        num_workers=0,
        shuffle=False,
        collate_fn=None,
        #pin_memory=True,
        #pin_memory_device='cuda:0'
    )

    positive_frames = np.array([], dtype='uint16')
    positive_frames_confidences = np.array([], dtype='float32')

    results = []
    #f = 0
    processed_images = 0
    for inputs in dl:

        inputs = inputs.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            outputs = model(inputs).detach().cpu()
            preds = torch.argmax(outputs, 1).numpy()
            positive_indexes = np.nonzero(np.equal(preds, 1))[0]

            if len(positive_indexes):
                positive_frames = np.hstack((positive_frames, processed_images + positive_indexes))

                confidences = torch.max(outputs, 1)
                pis = [confidences[0][pi] for pi in positive_indexes]
                positive_frames_confidences = np.hstack((positive_frames_confidences, pis))
                # print([sds[pi] for pi in positive_indexes])
        #f += 1
        processed_images += len(inputs)

    ignore_frames = list_suboptimal_frames(positive_frames, step_size, positive_frames_confidences, prediction_window_size)
    #print(ignore_frames)
    for i, f in enumerate(positive_frames):
        if positive_frames_confidences[i] >= min_confidence_for_positive and i not in ignore_frames:
            results.append((f, np.reshape(ds[f], (-1, MATRIX_WIDTH)), positive_frames_confidences[i]))

    return results


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PDClassifier(MATRIX_LEN * MATRIX_WIDTH, 2).to(DEVICE)
    MODEL_FN = f"bsa_binary_classification_{str(DEVICE)}.pth"
    model.load_state_dict(torch.load(MODEL_FN))
    model.eval()
    print("AI model initialized, inferencing on", DEVICE)
    min_confidence_for_positive = float(sys.argv[2])
    for path_to_parse in sys.argv[3:]:
        for fn_for_processing in glob.glob(os.path.join(path_to_parse, "*." + LONG_FILE_EXT)):

            long_file_name = os.path.basename(fn_for_processing)
            if list_of_long_files and long_file_name not in list_of_long_files:
                continue

            dst_dir_name = os.path.basename(fn_for_processing)[:os.path.basename(fn_for_processing).index(".")]
            Path(os.path.join(sys.argv[1], dst_dir_name)).mkdir(parents=True, exist_ok=True)

            _, data, _ = read_long_file(os.path.join(path_to_parse, fn_for_processing))
            print("Datafile opened")
            #generate_ray_pictures(data, "f:/BSA/original_extraction")

            pre_process_data_adj(data, mean_window_size=104)
            print("Datafile normalized")
            #generate_ray_pictures(data, "f:/BSA/normalized_extraction")
            #exit(0)
            predict_datafile(
                os.path.join(sys.argv[1], dst_dir_name),
                data,
                model,
                min_confidence_for_positive,
                prediction_window_size=148*2,#74
                step_size=15,
                rightmost_points_to_ignore=90
            )
