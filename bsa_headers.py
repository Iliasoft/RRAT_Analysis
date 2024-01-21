import ephem

NUM_PAR_KEY = "numpar"
NUM_POINTS_KEY = "npoints"
NUM_RAYS_KEY = "rays"
NUM_BANDS_KEY = "nbands"
NUM_MODULUS_KEY = "modulus"
NUM_MODULE_KEY = "module"
T_RESOLUTION_KEY = "tresolution"

NUM_DISPERSION_KEY = "dispersion"
DATE_BEGIN = "date_begin"
DATE_END = "date_end"
TIME_BEGIN = "time_begin"
TIME_END = "time_end"
SOURCE_KEY = "source"
FBANDS_KEY = "fbands"
WBANDS_KEY = "wbands"
SNR_KEY = "snr"
FCENTRAL_KEY = "fcentral"
WB_TOTAL_KEY = "wb_total"
ALPHA_KEY = "alpha"
DELTA_KEY = "delta"
F_CENTRAL_KEY = "fcentral"
STAR_TIME_KEY = "star_time"
POINT_KEY = "point"
RAY_KEY = "ray"
NATIVE_DATETIME_KEY = "native_datetime"
NL_PARAM_VALUE_SEPARATOR_CODE = 0x9
NL_DELIMITER_CODE = 10
NL_PADDING_CODE = 0x20

ERROR_CODE_KEY = "error"
FILE_NAME_KEY = "file_name"
MATRIX_LEN = 148
MATRIX_WIDTH = 32
RAYS_PER_MODULE = 8

N_POINT_FILE_EXPECTED_HEADERS = 24
N_LONG_FILE_EXPECTED_HEADERS = 16
int_metadata = [
    NUM_PAR_KEY, NUM_DISPERSION_KEY, NUM_MODULE_KEY, NUM_MODULUS_KEY, NUM_BANDS_KEY, NUM_POINTS_KEY,
    POINT_KEY, NUM_POINTS_KEY, RAY_KEY, NUM_RAYS_KEY
]
float_metadata = [T_RESOLUTION_KEY, FBANDS_KEY, WBANDS_KEY, SNR_KEY, FCENTRAL_KEY, WB_TOTAL_KEY]
date_str_metadata = [
    DATE_BEGIN, TIME_BEGIN, DATE_END, TIME_END, NATIVE_DATETIME_KEY, STAR_TIME_KEY, SOURCE_KEY, ALPHA_KEY, DELTA_KEY
]

point_file_metadata_padding = {
    NUM_PAR_KEY: 14,
    NUM_MODULUS_KEY: 9,
    SOURCE_KEY: 18,
    ALPHA_KEY: 17,
    DELTA_KEY: 17,
    F_CENTRAL_KEY: 18,
    WB_TOTAL_KEY: 15,
    DATE_BEGIN: 37,
    TIME_BEGIN: 33,
    DATE_END: 22,
    TIME_END: 20,
    WBANDS_KEY: 89,
    NUM_POINTS_KEY: 14,
}

point_file_metadata_separation_required = (
    NUM_DISPERSION_KEY,
    FBANDS_KEY,
    NUM_MODULE_KEY,
    NUM_MODULUS_KEY,
    NATIVE_DATETIME_KEY,
    NUM_BANDS_KEY,
    POINT_KEY,
    RAY_KEY,
    NUM_RAYS_KEY,
    SNR_KEY,
    STAR_TIME_KEY,
    T_RESOLUTION_KEY
)

TIME_UTC_SEPARATOR = " UTC "
TIME_COMPONENTS_SEPARATOR = ":"
HEADER_SECTION_EXPECTED_MAX_LENGTH = 1700
LONG_FILE_NAME_ATTRIBUTES_SEPARATOR = '_'

point_file_headers_sequence = [
    NUM_PAR_KEY, NUM_DISPERSION_KEY, FBANDS_KEY, NUM_MODULE_KEY, NUM_MODULUS_KEY, NATIVE_DATETIME_KEY, NUM_BANDS_KEY,
    POINT_KEY, RAY_KEY, NUM_RAYS_KEY, SNR_KEY, STAR_TIME_KEY, T_RESOLUTION_KEY, SOURCE_KEY, ALPHA_KEY, DELTA_KEY,
    F_CENTRAL_KEY, WB_TOTAL_KEY, DATE_BEGIN, TIME_BEGIN, DATE_END, TIME_END, WBANDS_KEY, NUM_POINTS_KEY
]

solar_timer = ephem.Observer()
solar_timer.lon, solar_timer.lat = '37.631358', '54.820795'
solar_timer.elevation = 204

UT_CORRECTION = -4
FILE_NAME_START_TIME_CORRECTION = -1
LONG_FILE_EXT = "pnthr"
POINT_FILE_EXT = "pnt"
ALL_PNT_FILES = "*." + POINT_FILE_EXT

DEFAULT_T_RESOLUTION = float(12.4928)
YEAR_PREFIX = "20"
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600
DEFAULT_OBSERVATION_DURATION = int(3.5 * SECONDS_IN_MINUTE)
N0_SIGNATURE = "_N0_"
DEFAULT_MODULE_NAMES = ["1", "2", "3", "4", "5", "6"]
N0_MODULE_NAMES = ["4", "5", "6"]
L_PAD_NUMBER_SYMBOL = "0"

SECONDS_IN_DAY_BREAKDOWN_SECTOR = 204
LAST_DAY_SECTOR_NAME = 423
LAST_DAY_SECTOR_CUTOFF = (LAST_DAY_SECTOR_NAME - 1) * SECONDS_IN_DAY_BREAKDOWN_SECTOR

POSITIVES_DIR = "positives"
NEGATIVES_DIR = "negatives"
PNG_FILE_EXT = "png"
ALL_PNG_FILES = "*." + PNG_FILE_EXT
