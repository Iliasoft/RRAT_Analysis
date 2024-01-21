import numpy as np
import torch
import scipy.ndimage
import os
from datetime import datetime, timedelta
from bsa_headers import *


def read_point_file(file_name, augmented_points_num=0):
    try:
        with open(file_name, mode='rb') as file:
            file_data = np.fromfile(file, np.dtype('B'))

    except FileNotFoundError:
        return {ERROR_CODE_KEY: 0}
    except IOError:
        return {ERROR_CODE_KEY: 1}

    nl_indexes = np.insert(np.nonzero(np.equal(file_data[:HEADER_SECTION_EXPECTED_MAX_LENGTH], NL_DELIMITER_CODE))[0], 0, [-1])
    if len(nl_indexes) < N_POINT_FILE_EXPECTED_HEADERS + 1:
        return {ERROR_CODE_KEY: 2}  # unknown PNT header format

    metadata = {FILE_NAME_KEY: file_name}

    for header_seq in range(N_POINT_FILE_EXPECTED_HEADERS):
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

    if metadata[NUM_PAR_KEY] != N_POINT_FILE_EXPECTED_HEADERS or metadata[NUM_RAYS_KEY] != 1 or metadata[NUM_BANDS_KEY] != MATRIX_WIDTH:
        return {ERROR_CODE_KEY: 5}  # unsupported type if PNT file

    spectrum_data_start_position = nl_indexes[N_POINT_FILE_EXPECTED_HEADERS] + 1
    ###################
    npoints = metadata[NUM_POINTS_KEY]
    channels = metadata[NUM_BANDS_KEY] + 1
    data = np.reshape(
        file_data[spectrum_data_start_position:].view('float32'),
        (npoints, channels)
    )

    # we don't need the mean channel for prediction
    data = data[:, :MATRIX_WIDTH]

    if data.shape[0] > MATRIX_LEN:
        # calculate how much data will be cropped
        l_crop = int((data.shape[0] - MATRIX_LEN) / 2)
        r_crop = data.shape[0] - MATRIX_LEN - l_crop
        data = data[l_crop:data.shape[0] - r_crop, :]

    '''
    # basic data cleanup
    Q1 = -0.6#np.percentile(data, 3, method='midpoint')
    Q3 = 0.6# np.percentile(data, 97, method='midpoint')
    indx = np.where(data > Q3)
    data[indx] = Q3
    indx = np.where(data < Q1)
    data[indx] = Q1
    '''

    scale_factor = MATRIX_LEN / data.shape[0]
    data_scaled = augmentation_scale(data, scale_factor).ravel()
    '''
    if augmented_points_num:
        scale_factor = MATRIX_LEN / data.shape[0]
        data_scaled = augmentation_scale(data, scale_factor).ravel()

    else:
        data_scaled = augmentation_scale(data, 0).ravel()
    '''
    s = np.std(data_scaled)
    m = np.mean(data_scaled)
    data_scaled_normalized = (data_scaled - m) / s

    augmented_points = []

    for aug in range(augmented_points_num):
        augmented_points.append(
            torch.tensor(generate_augmented_point(data), dtype=torch.float32)
        )

    return metadata, torch.tensor(data_scaled_normalized, dtype=torch.float32), augmented_points, data


def augmentation_scale(source_data, scaling_factor):
    return scipy.ndimage.zoom(source_data, (scaling_factor, 1), order=2)


def generate_augmented_point(data):
    augmented = data
    scale_factor = MATRIX_LEN / augmented.shape[0]
    augmented = augmentation_scale(augmented, scale_factor)
    s = np.std(augmented)
    m = np.mean(augmented)
    augmented = (augmented - m) / s

    return augmented.ravel()


def build_param_str(param, value):

    byted_param = [ord(s) for s in str(param)]

    if param in point_file_metadata_separation_required:
        byted_param.append(NL_PARAM_VALUE_SEPARATOR_CODE)

    if param in point_file_metadata_padding:
        for i in range(point_file_metadata_padding[param] - len(param) - len(str(value))):
            byted_param.append(NL_PADDING_CODE)

    byted_param.extend([ord(s) for s in str(value)])
    byted_param.append(NL_DELIMITER_CODE)

    return byted_param


def create_point_file(file_name, metadata, data):

    metadata[FBANDS_KEY] = " ".join([str(s) for s in metadata[FBANDS_KEY]])
    metadata[WBANDS_KEY] = " ".join([str(s) for s in metadata[WBANDS_KEY]])
    data_to_be_stored = []

    for param in point_file_headers_sequence:
        data_to_be_stored.extend(build_param_str(param, metadata[param]))

    data = np.reshape(data.view('byte'), -1)
    try:
        with open(file_name, 'wb') as file:
            file.write(bytes(data_to_be_stored))
            file.write(data)
    except Exception as e:
        print(e)
        return False

    return True


# this method is called to create a point file based on detection module coordinates and long file data
def build_point_file(point_file_name, long_file_metadata, long_file_data, module, ray, start_point, n_points, st_hour, st_minute, st_second):

    point_data = np.reshape(long_file_data[start_point: start_point + n_points, module, ray], (n_points, long_file_data.shape[3]))
    point_file_metadata = {}

    point_file_metadata[NUM_PAR_KEY] = N_POINT_FILE_EXPECTED_HEADERS
    point_file_metadata[NUM_RAYS_KEY] = 1
    point_file_metadata[NUM_MODULUS_KEY] = 0
    point_file_metadata[NATIVE_DATETIME_KEY] = os.path.basename(long_file_metadata[FILE_NAME_KEY])
    point_file_metadata[T_RESOLUTION_KEY] = long_file_metadata[T_RESOLUTION_KEY]
    point_file_metadata[FBANDS_KEY] = long_file_metadata[FBANDS_KEY]
    point_file_metadata[WBANDS_KEY] = long_file_metadata[WBANDS_KEY]
    point_file_metadata[SOURCE_KEY] = long_file_metadata[SOURCE_KEY]
    point_file_metadata[ALPHA_KEY] = long_file_metadata[ALPHA_KEY]
    point_file_metadata[DELTA_KEY] = long_file_metadata[DELTA_KEY]
    point_file_metadata[F_CENTRAL_KEY] = long_file_metadata[F_CENTRAL_KEY]
    point_file_metadata[WB_TOTAL_KEY] = long_file_metadata[WB_TOTAL_KEY]
    point_file_metadata[NUM_BANDS_KEY] = long_file_metadata[NUM_BANDS_KEY]
    point_file_metadata[DATE_BEGIN] = long_file_metadata[DATE_BEGIN]
    point_file_metadata[DATE_END] = long_file_metadata[DATE_END]
    point_file_metadata[NUM_MODULE_KEY] = long_file_metadata[NUM_MODULUS_KEY][module]
    point_file_metadata[POINT_KEY] = start_point
    point_file_metadata[RAY_KEY] = str(ray + 1)
    point_file_metadata[NUM_POINTS_KEY] = n_points
    point_file_metadata[TIME_BEGIN] = long_file_metadata[TIME_BEGIN]
    point_file_metadata[TIME_END] = long_file_metadata[TIME_END]
    point_file_metadata[NUM_DISPERSION_KEY] = int(n_points * 0.7)  # not implemented
    point_file_metadata[SNR_KEY] = 0  # not implemented
    point_file_metadata[STAR_TIME_KEY] = TIME_COMPONENTS_SEPARATOR.join([str(st_hour), str(st_minute), str(st_second)])

    '''
    def pad_time_component(s):  
        return s if len(s) > 1 else "0" + s

    hour_start_date = long_file_metadata[DATE_BEGIN].split(TIME_UTC_SEPARATOR)
    assert len(hour_start_date) == 2
    hour_start_local_date = hour_start_date[0]
    hour_start_time = long_file_metadata[TIME_BEGIN].split(TIME_UTC_SEPARATOR)
    assert len(hour_start_time) == 2

    hour_start_local_time = hour_start_time[0]
    point_ut_correction = timedelta(hours=UT_CORRECTION, milliseconds=start_point * float(point_file_metadata[T_RESOLUTION_KEY]))
    point_ut_time = datetime.strptime(hour_start_local_date + " " + hour_start_local_time, '%d.%m.%Y %H:%M:%S') + point_ut_correction
    solar_timer.date = ephem.Date(point_ut_time)
    solar_time = str(solar_timer.sidereal_time()).split(TIME_COMPONENTS_SEPARATOR)

    assert len(solar_time) == 3
    solar_time[0] = pad_time_component(solar_time[0])
    solar_time[1] = pad_time_component(solar_time[1])
    solar_time[2] = pad_time_component(str(int(float(solar_time[2]))))
    point_file_metadata[STAR_TIME_KEY] = TIME_COMPONENTS_SEPARATOR.join(solar_time)
    print(point_file_metadata[NATIVE_DATETIME_KEY], "=>", point_file_metadata[STAR_TIME_KEY])
    '''

    create_point_file(point_file_name, point_file_metadata, point_data)


point_file_metadata_sample = {
    'file_name': 'f:/BSA/transients2016-high/030/000000.pnt',
    'numpar': 24,
    'dispersion': 12,
    'fbands': [109.039, 109.117, 109.195, 109.273, 109.352, 109.43, 109.508, 109.586, 109.664, 109.742, 109.82, 109.898,
    109.977, 110.055, 110.133, 110.211, 110.289, 110.367, 110.445, 110.523, 110.602, 110.68, 110.758, 110.836, 110.914,
    110.992, 111.07, 111.148, 111.227, 111.305, 111.383, 111.461],
    'module': 3,
    'modulus': 0,
    'native_datetime': '060815_07_N1_00.pnthr',
    'nbands': 32,
    'point': 49105,
    'ray': 6,
    'rays': 1,
    'snr': '6.92057',
    'star_time': '01:39:05',
    'tresolution': '12.4928',
    'source': 'source',
    'alpha': 'alpha',
    'delta': 'delta',
    'fcentral': '110.25',
    'wb_total': '2.5',
    'date_begin': '23.10.2015 UTC 23.10.2015',
    'time_begin': '23:20:47 UTC 19:20:47',
    'date_end': '23.10.2015',
    'time_end': '21:20:47',
    'wbands': [0.4150390625, 0.4150390625, 0.4150390625, 0.4150390625, 0.4150390625, 0.4248046875],
    'npoints': 36
}
'''
Point file header structure
== hardcode ==
'numpar' int: 24 (must always be 24)
'rays': int: 1 
'modulus' int 0: ???

== calculated by detection module
'dispersion' int : 13 (has to be calculated by detection module)
'snr': float: '6.92057' (has to be calculated by detection module)

== position of the block generated by detection module
'point' int: 49105 (starting point) 
'npoints' str: 36 
'module' int: 3 
'ray': int: 6 
'star_time': '01:39:05',
'time_begin': '23:20:47 UTC 19:20:47'
'time_end': str: '21:20:47'

== 1:1 mapping from source pnthr fil
'native_datetime' str: '060815_07_N1_00.pnthr' (name of a file with hourly recordings, which the point being created is originated from)
'tresolution' float: '12.4928'
'fbands' array of floats, [109.039, 109.117, 109.195, 109.273, 109.352, 109.43, 109.508, 109.586, 109.664, 109.742, 109.82, 109.898, 109.977, 110.055, 110.133, 110.211, 110.289, 110.367, 110.445, 110.523, 110.602, 110.68, 110.758, 110.836, 110.914, 110.992, 111.07, 111.148, 111.227, 111.305, 111.383, 111.461],
'source' str: 'source' 
'alpha' str: 'alpha'
'delta' str: 'delta'
'fcentral' float: '110.25'
'wb_total': float: '2.5'
'wbands' [0.4150390625, 0.4150390625, 0.4150390625, 0.4150390625, 0.4150390625, 0.4248046875]
'nbands' int: 32
'date_begin': '23.10.2015 UTC 23.10.2015'
'date_end': str: '23.10.2015'
'''
