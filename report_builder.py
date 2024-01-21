import sys
import glob
import png
import numpy as np
import os
import pandas as pd
from datetime import timedelta
from pathlib import Path
from tqdm import tqdm
from os import listdir
from os.path import isdir, join
from parse_long_file import pre_process_data_adj
from bsa_long_file import read_long_file
from bsa_point_file import build_point_file
from bsa_headers import *
from sidereal_time import *
'''
the task of this module is to crawl directory structure of pngs reviewed by operator
and generate a list with a record for each png file found
'''


def process_observations(pngs, directory_to_store_pnt_files, long_file_metadata, long_file_data):

    processed_dirs = set([])
    for png_file_name in pngs:

        png_fn_components = os.path.basename(png_file_name).split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)
        module = int(png_fn_components[0])
        ray = int(png_fn_components[1])
        start_point = int(png_fn_components[2])
        assert len(png_fn_components) == 4
        reader = png.Reader(filename=png_file_name)
        n_points, _, _, _ = reader.read_flat()

        st_hour, st_minute, st_second = fn_to_sidereal_time(png_file_name, long_file_name)

        point_file_dir_name = join(directory_to_store_pnt_files, sidereal_time_to_dir_name(st_hour, st_minute, st_second))
        Path(point_file_dir_name).mkdir(parents=True, exist_ok=True)

        existing_pnts = glob.glob(join(point_file_dir_name, ALL_PNT_FILES), recursive=False)
        max_existing_id = 0
        for pnt_fn in existing_pnts:
            try:
                if int(Path(pnt_fn).stem) > max_existing_id:
                    max_existing_id = int(Path(pnt_fn).stem)
            except ValueError:
                pass
        point_file_name = (L_PAD_NUMBER_SYMBOL * 5 + str(max_existing_id + 1))[-6:]
        point_file_name = join(point_file_dir_name, point_file_name + "." + POINT_FILE_EXT)  # "f:/BSA/0001.pnt"

        build_point_file(point_file_name, long_file_metadata, long_file_data, module, ray, start_point, int(n_points / 2), st_hour, st_minute, st_second)
        processed_dirs.add(png_file_name[:38])
    # print(processed_dirs)


def calculate_observation_file_name(observation):
    fn = str(observation[9]) \
        + LONG_FILE_NAME_ATTRIBUTES_SEPARATOR \
        + str(observation[6] - 1) \
        + LONG_FILE_NAME_ATTRIBUTES_SEPARATOR \
        + str(observation[7]) \
        + LONG_FILE_NAME_ATTRIBUTES_SEPARATOR \
        + observation[10]

    return os.path.join(observation[8], os.path.join(POSITIVES_DIR, fn))


def list_observations(pngs, long_file_name, module_names):

    long_file_attributes = long_file_name.split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)
    assert len(long_file_attributes) == 4
    ploshadka = long_file_attributes[2]

    positive_records = []
    for png_file_name in pngs:
        solar_hour, solar_minute, solar_second = fn_to_sidereal_time(png_file_name, long_file_name)
        # 0_4_277920_0.91 x 148 pix
        png_file_name = os.path.basename(png_file_name).split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)

        assert len(png_file_name) == 4
        module_name = module_names[int(png_file_name[0])]
        module = int(png_file_name[0])
        ray = int(png_file_name[1]) + 1
        start_point = int(png_file_name[2])
        suffix = png_file_name[3]

        positive_records.append(
            (
                sidereal_time_to_seconds(solar_hour, solar_minute, solar_second),
                solar_hour,
                solar_minute,
                solar_second,
                ploshadka,
                module_name,
                ray,
                start_point,
                long_file_name,
                module,
                suffix
            )
        )

    return positive_records


def grouper(iterable):
    prev = None
    group = []
    for item in iterable:
        if prev is None or item - prev <= DEFAULT_OBSERVATION_DURATION:
            group.append(item)
        else:
            yield group
            group = [item]
        prev = item
    if group:
        yield group


def grouper_2(numbers):
    groups = []
    for number in numbers:
        found_group = False
        for group in groups:
            for member in group:
                if abs(member - number) <= DEFAULT_OBSERVATION_DURATION:
                    group.append(number)
                    found_group = True
                    break

                # remove this if-block if a number should be added to multiple groups
                if found_group:
                    break
        if not found_group:
            groups.append([number])
    return groups


def group_observations_by_time(all_observations, max_pulses_in_valid_group):

    all_observations = sorted(all_observations, key=lambda observation: observation[0])  # sort by total siderial seconds
    seconds_of_all_observations = [s[0] for s in all_observations]
    time_to_idx = {all_observations[s][0]: s for s in range(len(all_observations))}
    time_clusters = dict(enumerate(grouper(seconds_of_all_observations), 1))
    # time_clusters = grouper_2(total_seconds)
    # time_clusters = {i:time_clusters[i] for i in range(len(time_clusters))}

    observations_for_pnts_build = []
    grouped_observations = []

    for cluster in time_clusters.keys():
        observation_times = time_clusters[cluster]
        if len(observation_times) > max_pulses_in_valid_group:
            continue

        medium_time = str(timedelta(seconds=int(np.mean(observation_times))))
        span_time = ":".join(str(timedelta(seconds=int(np.max(observation_times) - np.min(observation_times)))).split(":")[1:])

        observation_descriptions = set()
        observation_dirs = set()
        for t in observation_times:
            #position = 0
            #idx = seconds_of_all_observations[position:].index(t)

            obs = all_observations[time_to_idx[t]]
            observation_descriptions.add(
                obs[4] + ", Module=" + str(obs[5]) + ", Ray=" + str(obs[6]) + ", " + obs[8] + " \n\r"
            )
            observation_dirs.add(
                sidereal_time_to_dir_name(*seconds_to_sidereal_time(t))
            )

            observations_for_pnts_build.append(obs)

        grouped_observations.append(
            (
                medium_time,
                span_time,
                len(observation_times),
                " ".join(observation_dirs),
                " ".join(observation_descriptions)
             )
        )

    return grouped_observations, observations_for_pnts_build


if __name__ == '__main__':
    directory_to_parse = sys.argv[1]
    max_pulses_in_valid_group = int(sys.argv[2])
    build_pnt_files_requested = len(sys.argv) > 2
    if build_pnt_files_requested:
        directory_to_store_pnt_files = sys.argv[3]
        directory_with_long_files = sys.argv[4]

    dirs_with_jpgs = [f for f in listdir(directory_to_parse) if isdir(join(directory_to_parse, f))]
    all_observations = []

    for d in tqdm(dirs_with_jpgs):

        dir_name = join(directory_to_parse, d)
        positive_pngs = glob.glob(join(join(dir_name, POSITIVES_DIR), ALL_PNG_FILES), recursive=False)

        all_observations.extend(
            list_observations(positive_pngs, d, DEFAULT_MODULE_NAMES if d.find(N0_SIGNATURE) == -1 else N0_MODULE_NAMES)
        )

    grouped_observations, observations_for_pnts_build = group_observations_by_time(
        all_observations,
        max_pulses_in_valid_group
    )

    df = pd.DataFrame(
        grouped_observations,
        columns=[
            "Pulses Group Mean Time",
            "Group Duration",
            "Pulses Count",
            "Point Files Directories",
            "Sources"
        ]
    )
    with pd.ExcelWriter("RRAT_search_report.xlsx") as writer:
        df.to_excel(writer, index=False)

    if build_pnt_files_requested:
        long_files = set([])
        for pf in observations_for_pnts_build:
            long_files.add(pf[8])

        long_files = sorted(long_files)

        for long_file in tqdm(long_files):

            long_file_attributes = long_file.split(LONG_FILE_NAME_ATTRIBUTES_SEPARATOR)
            assert len(long_file_attributes) == 4 and len(long_file_attributes[0]) == 6
            long_file_name = join(directory_with_long_files, join(long_file_attributes[2],
                                                                  join(YEAR_PREFIX + long_file_attributes[0][4:],
                                                                       join(long_file_attributes[0][2:4],
                                                                            join(long_file_attributes[0][:2],
                                                                                 long_file + "." + LONG_FILE_EXT)))))

            try:
                long_file_metadata, _, long_file_data = read_long_file(long_file_name)
            except ValueError as e:
                print(long_file_name, e)
                continue

            pre_process_data_adj(long_file_data, mean_window_size=104)

            observations_of_long_file = []
            for observation in observations_for_pnts_build:
                if observation[8] == long_file:
                    observations_of_long_file.append(
                        os.path.join(
                            directory_to_parse,
                            calculate_observation_file_name(observation)
                        )
                    )

            process_observations(
                observations_of_long_file,
                directory_to_store_pnt_files,
                long_file_metadata,
                long_file_data
            )
