import os
import argparse
import pandas as pd
import extract_word_phone_timing as et

"""
Script for extracting features from word-phone aligned timing files.
"""


def get_segments_from_file(timing_file):
    """
    :param timing_file: opened file containing list of segments ID with corresponding word-phone timing information
    :return: segment_dicts: list of segments, where each segment is a dict with entries "segment_start", "segment_end",
    and "timing_info" (a list of strings that contain timing info for each phone in segment)
    """
    file_contents = timing_file.read()
    seg_strings = file_contents.split('\n"')
    segment_dicts = []
    for seg in seg_strings:
        seg = seg.strip()
        seg_id = seg.split("\n")[0]
        lines = seg.split("\n")[1:]
        segment_start, segment_end = seg_id.split("_")[2:]
        segment_dicts.append({"segment_start": int(segment_start), "segment_end": int(segment_end),
                              "segment_id": seg_id, "timing_info": lines})
    return segment_dicts


def get_subject_data(input_dir, sub_meta_df):
    """
    Collect data for subject <sub_id>.
    :param input_dir: Directory containing call subdirectories that have word-phone timing files
    :param sub_meta_df: dataframe containing metadata information for subject with sub_id
    :return: sub_data_df: DataFrame with columns: "week_id", "call_id", "segment_start", "segment_end", and
    "timing_info" (stores list of phone timing information for each segment)
    """
    sub_data = []
    call_ids = sub_meta_df["call_id"].values
    for call_id in call_ids:
        call_dir = os.path.join(input_dir, call_id)
        if os.path.exists(call_dir):
            timing_file = open(os.path.join(call_dir, "word_phone_ali_timing.txt"), 'r')
            segment_dicts = get_segments_from_file(timing_file)
            timing_file.close()
            # add entries to sub_data for each segment
            week = sub_meta_df[sub_meta_df["call_id"] == call_id].week
            date = sub_meta_df[sub_meta_df["call_id"] == call_id].date
            time = sub_meta_df[sub_meta_df["call_id"] == call_id].time
            for seg_dict in segment_dicts:
                # add relevant metadata information
                seg_dict["week"] = week
                seg_dict["date"] = date
                seg_dict["time"] = time
                seg_dict["call_id"] = call_id
                sub_data.append(seg_dict)
    # create dataframe from list of dicts
    sub_data_df = pd.DataFrame(sub_data)
    return sub_data_df


def collect_timing_info_by_level(sub_id, sub_data_df, level):
    """
    :param sub_id: id of subject associated with sub_data_df
    :param sub_data_df: DataFrame containing segment timing info data from a single subject.
    :param level: Level to group/aggregate data by.
    :return: sub_timing_list: list of timing information entries for subject, where timing info entry is a tuple of the
    form (group id items (e.g. "callid" or ["subject", "week"], "group_id", list of phone-timing info for each segment
    in group), and transcripts are collected at the <level> level
    """
    # sort segments by date, time, segment start time so that timing lists are in order
    sub_data_df.sort(['date', 'time', 'segment_start'])
    sub_timing_list = []
    if level == "subject":
        # group all data together
        timing_info = [row["timing_info"] for index, row in sub_data_df.iterrows()]
        sub_timing_list.append((sub_id, timing_info))
    else:
        # group timing info based on data level
        if level == "week":
            group_by_list = ["subject", "week"]
        elif level == "day":
            group_by_list = ["subject", "week", "day"]
        elif level == "call":
            group_by_list = ["call_id"]
        else: # level is segment
            group_by_list = ["segment_id"]
        by_group = sub_data_df.groupby(group_by_list)
        for group_id, df in by_group:
            timing_info = [row["timing_info"] for index, row in df.iterrows()]
            sub_timing_list.append((group_by_list, group_id, timing_info))
    return sub_timing_list


def get_total_duration(sub_data_df, id_elms, group_id, duration_df):
    """
    :param sub_data_df: DataFrame containing segment timing info data from a single subject.
    :param id_elms: List of id types used in uniquely identifying data group.
    :param group_id: ID used to uniquely identify data group (e.g. callid, or [subject_id, week])
    :param duration_df: DataFrame mapping call_ids to call durations.
    :return: total_duration: If level > segment level, total duration of all calls corresponding to the data group
    (this is not just when participant is speaking). If at the segment level, just return total segment time.
    """
    if id_elms[0] == "segment_id":
        segment_row = sub_data_df[sub_data_df["segment_id"] == group_id]
        # multiply by *.001 to convert ms to seconds
        segment_duration = (float(segment_row["segment_start"]) - float(segment_row["segment_end"])) * .001
        return segment_duration
    else:
        # collect all calls present within data group
        group_df = sub_data_df[(sub_data_df[id_elms] == group_id).all(1)]
        call_ids = group_df["call_id"].values
        durations = duration_df[duration_df["call_id"].isin(call_ids)]["duration"].values
        return sum(durations)


def extract_features(input_dir, level, meta_df, duration_df):
    """
    Extract features for each group of timing data aggregated at the <level> level and return as Dataframe
    with group id as index and features as columns.
    :param input_dir: Directory containing subdirectories for each call, where each call directory contains word-phone
    timing files.
    :param level: Level (segment, call, day, week, subject) to group data by before extracting features.
    :param meta_df: Dataframe containing metadata information.
    :return: Dataframe containing feature values for each group of data
    """
    # feature_list is is list that contains a feat dict for group of data
    # each feat dict contains feature entries as well as information needed to uniquely identify data group
    feature_list = []
    sub_ids = meta_df["subject_id"].index.values
    for sub_id in sub_ids:
        sub_meta_df = meta_df[meta_df["subject_id"] == sub_id]
        sub_data_df = get_subject_data(input_dir, sub_meta_df)
        # group segments based on specified data level
        sub_timing_list = collect_timing_info_by_level(sub_id, sub_data_df, level)
        for id_elms, group_id, timing_info in sub_timing_list:
            # get total duration of data group (i.e. sum of times of all calls included)
            total_duration = get_total_duration(sub_data_df, id_elms, group_id, duration_df)
            group_feature_dict = et.get_feats(timing_info, total_duration)
            for idx, id_elm in enumerate(id_elms):
                group_feature_dict[id_elm] = group_id[idx]
            feature_list.append(group_feature_dict)
    feature_df = pd.DataFrame(feature_list)
    return feature_df


def main():
    # Read in and process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='Directory containing call subdirectories that contain'
                                                            'word-phone timing files')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to output feature csv file')
    parser.add_argument('-l', '--level', type=str, help='Level of data to extract features for. Options are: segment,'
                                                        'call, day, week, subject')
    parser.add_argument('-m', '--metadata_file_path', type=str, help='Path to pickled DataFrame containing metadata '
                                                                     'information, including mapping between '
                                                                     'subject_ids, call_ids, assess weeks, dates,'
                                                                     'times, and segment_ids. Rows in DataFrame'
                                                                     'should correspond to segments in the dataset')
    parser.add_argument('-d', '--duration_file_path', type=str, help='Path to pickled DataFrame that maps callids'
                                                                     'to total duration of call')
    args = parser.parse_args()

    # Load metadata (used for grouping data when extracting features at different levels)
    meta_df = pd.read_pickle(args.metadata_file_path)

    # Load call duration df
    duration_df = pd.read_pickle(args.duration_file_path)

    # Extract features for all transcripts
    feature_df = extract_features(args.input_dir, args.level, meta_df, duration_df)
    feature_df.to_pickle("word_phone_timing.pkl")


if __name__ == '__main__':
    main()
