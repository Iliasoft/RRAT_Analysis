import os
import math
from datetime import datetime, timedelta
from bsa_headers import *


def sidereal_time_to_seconds(h, m, s):
    return h * SECONDS_IN_HOUR + m * SECONDS_IN_MINUTE + s


def seconds_to_sidereal_time(t):
    h = t // SECONDS_IN_HOUR
    m = (t - h * SECONDS_IN_HOUR) // SECONDS_IN_MINUTE
    s = t - h * SECONDS_IN_HOUR - m * SECONDS_IN_MINUTE

    return h, m, s


def sidereal_time_to_dir_name(h, m, s):
    '''
    0...203 - 1
    204...407 - 2
    408...611 - 3
    85884...86087 - 422
    86088...  - 423

    :param h:
    :param m:
    :param s:
    :return:
    '''
    t = sidereal_time_to_seconds(h, m, s) + 1
    if t > LAST_DAY_SECTOR_CUTOFF:
        return str(LAST_DAY_SECTOR_NAME)
    else:
        t_s = str(math.ceil(t / SECONDS_IN_DAY_BREAKDOWN_SECTOR))
        return (L_PAD_NUMBER_SYMBOL * 2 + t_s)[-3:]


def fn_to_sidereal_time(png_file_name, long_file_name):

    long_file_components = long_file_name.split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)
    assert len(long_file_components) == 4

    year = int(YEAR_PREFIX + long_file_components[0][-2:])
    day = int(long_file_components[0][-6:-4])
    month = int(long_file_components[0][-4:-2])
    hour = int(long_file_components[1])

    png_fn_component = os.path.basename(png_file_name).split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)
    assert len(png_fn_component) == 4
    start_point = int(png_fn_component[2])
    ##################
    time_delta = timedelta(
        hours=FILE_NAME_START_TIME_CORRECTION + UT_CORRECTION,
        milliseconds=start_point * DEFAULT_T_RESOLUTION
    )
    start_datetime = datetime(year, month, day, hour) + time_delta

    solar_timer.date = ephem.Date(start_datetime)
    solar_time = str(solar_timer.sidereal_time()).split(TIME_COMPONENTS_SEPARATOR)
    assert len(solar_time) == 3
    solar_hour = int(solar_time[0])
    solar_minute = int(solar_time[1])
    solar_second = int(float(solar_time[2]))  # ok to round to avoid 60th second issue
    return solar_hour, solar_minute, solar_second
