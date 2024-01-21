import numpy as np
from bsa_headers import *


def read_long_file(file_name):

    try:
        with open(file_name, mode='rb') as file:

            file_data = np.fromfile(file, np.dtype('B'))
            # print(len(file_data))
    except FileNotFoundError:
        return {ERROR_CODE_KEY: 0}
    except IOError:
        return {ERROR_CODE_KEY: 1}

    nl_indexes = np.insert(np.nonzero(np.equal(file_data[:HEADER_SECTION_EXPECTED_MAX_LENGTH], NL_DELIMITER_CODE))[0], 0, [-1])
    if len(nl_indexes) < N_LONG_FILE_EXPECTED_HEADERS + 1:
        return {ERROR_CODE_KEY: 2}, None  # unknown PNT header format

    metadata = {FILE_NAME_KEY: file_name}

    for header_seq in range(N_LONG_FILE_EXPECTED_HEADERS):
        h = file_data[nl_indexes[header_seq] + 1:nl_indexes[header_seq + 1]].tobytes().decode("utf-8").split()

        if h[0] not in int_metadata and h[0] not in float_metadata and h[0] not in date_str_metadata:
            continue

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

    # print(metadata)
    if metadata[NUM_PAR_KEY] != N_LONG_FILE_EXPECTED_HEADERS or metadata[NUM_BANDS_KEY] != MATRIX_WIDTH:
        return {ERROR_CODE_KEY: 5}  # unsupported type if PNT file

    spectrum_data_start_position = nl_indexes[N_LONG_FILE_EXPECTED_HEADERS] + 1
    ###################
    # n_modules x 8 x n_channels x points x 4
    npoints = metadata[NUM_POINTS_KEY]
    bands = metadata[NUM_BANDS_KEY]
    modules = len(metadata[NUM_MODULUS_KEY])

    data = np.reshape(
        file_data[spectrum_data_start_position:].view('float32'),
        (npoints, modules, RAYS_PER_MODULE, bands + 1)
    )

    return metadata, data[:, :, :, :MATRIX_WIDTH], data
