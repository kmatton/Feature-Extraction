import os
import argparse
import pandas as pd
import numpy as np

"""
Module for extracting speech intelligibility features (i.e. ASR confidence) from Kaldi confidence files.
"""


def get_feats(conf_scores):
    """
    :param conf_scores: List of confidence scores from all words in a group of data.
    :return: feat_dict: Dictionary mapping features to values
    """
    feat_dict = {}
    feat_dict["conf_max"] = max(conf_scores)
    feat_dict["conf_mean"] = np.mean(conf_scores)
    feat_dict["conf_std"] = np.std(conf_scores)
    feat_dict["conf_min"] = min(conf_scores)
    feat_dict["conf_med"] = np.median(conf_scores)
    return feat_dict

def get_segment_scores_from_file(conf_file):
    """
    :param conf_file: opened file containing list of segments ID with corresponding ASR confidence
    :return: segment_dicts: list of segments, where each segment is a dict with entries "segment_start", "segment_end",
    and "conf_scores" (list of confidence scores for each word in segment)
    """
    # map segment ID to info about that segment
    segment_dict = {}
    for line in conf_file:
        seg_id = line.split(" ")[0]
        conf_score = float(line.split(" ")[5])
        if seg_id not in segment_dict:
            segment_dict[seg_id] = {}
            segment_dict[seg_id]["segment_start"] = int(seg_id.split("_")[2])
            segment_dict[seg_id]["segment_end"] = int(seg_id.split("_")[3])
            segment_dict[seg_id]["conf_scores"] = []
        segment_dict[seg_id]["conf_scores"].append(conf_score)

    # transform segment dict into list of dicts where, segment id is key of dict
    segment_dicts = []
    for seg_id, d in segment_dict.items():
        d["segment_id"] = seg_id
        segment_dicts.append(d)
    return segment_dicts


def get_subject_data(input_dir, sub_meta_df):
    """
    Collect data for subject <sub_id>.
    :param input_dir: Directory containing call subdirectories that have ASR confidence files
    :param sub_meta_df: dataframe containing metadata information for subject with sub_id
    :return: sub_data_df: DataFrame with columns: "week_id", "call_id", "segment_start", "segment_end", and
    "confidence_scores" (stores list of confidence scores for each segment)
    """
    sub_data = []
    call_ids = sub_meta_df["call_id"].values
    for call_id in call_ids:
        call_dir = os.path.join(input_dir, call_id)
        if os.path.exists(call_dir):
            conf_file = open(os.path.join(call_dir, "conf_sym.txt"), 'r')
            segment_dicts = get_segment_scores_from_file(conf_file)
            conf_file.close()
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


def collect_conf_scores_by_level(sub_id, sub_data_df, level):
    """
    :param sub_id: id of subject associated with sub_data_df
    :param sub_data_df: DataFrame containing segment confidence scores from a single subject.
    :param level: Level to group/aggregate data by.
    :return: sub_conf_list: list of confidence score info for subject, where segment confidence score info
    entry is of the form (group id items (e.g. "callid" or ["subject", "week"], "group_id", list of confidence scores
    for all words in data group)
    """
    # sort segments by date, time, segment start time so that timing lists are in order
    sub_data_df.sort(['date', 'time', 'segment_start'])
    sub_conf_list = []
    if level == "subject":
        # group all data together + make flattened list
        conf_scores = [score for index, row in sub_data_df.iterrows() for score in row["conf_scores"]]
        sub_conf_list.append((sub_id, conf_scores))
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
            conf_scores = [score for index, row in sub_data_df.iterrows() for score in row["conf_scores"]]
            sub_conf_list.append((group_by_list, group_id, conf_scores))
    return sub_conf_list


def extract_features(input_dir, level, meta_df):
    """
    Extract features for each group of ASR confidence data aggregated at the <level> level and return as DataFrame
    with group id as index and features as columns.
    :param input_dir: Directory containing subdirectories for each call, where each call directory contains ASR
    confidence files.
    :param level: Level (segment, call, day, week, subject) to group data by before extracting features.
    :param meta_df: DataFrame containing metadata information.
    :return: feature_df: DataFrame containing feature values for each group of data
    """
    # feature_list is is list that contains a feat dict for group of data
    # each feat dict contains feature entries as well as information needed to uniquely identify data group
    feature_list = []
    sub_ids = meta_df["subject_id"].index.values
    for sub_id in sub_ids:
        sub_meta_df = meta_df[meta_df["subject_id"] == sub_id]
        sub_data_df = get_subject_data(input_dir, sub_meta_df)
        # group confidence scores based on specified data level
        sub_conf_list = collect_conf_scores_by_level(sub_id, sub_data_df, level)
        for id_elms, group_id, conf_scores in sub_conf_list:
            group_feature_dict = get_feats(conf_scores)
            for idx, id_elm in enumerate(id_elms):
                group_feature_dict[id_elm] = group_id[idx]
            feature_list.append(group_feature_dict)
    feature_df = pd.DataFrame(feature_list)
    return feature_df


def main():
    # Read in and process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', type=str, help='Directory containing call subdirectories that contain'
                                                            'ASR confidence files')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to output feature csv file')
    parser.add_argument('-l', '--level', type=str, help='Level of data to extract features for. Options are: segment,'
                                                        'call, day, week, subject')
    parser.add_argument('-m', '--metadata_file_path', type=str, help='Path to pickled DataFrame containing metadata '
                                                                     'information, including mapping between '
                                                                     'subject_ids, call_ids, assess weeks, dates,'
                                                                     'times, and segment_ids. Rows in DataFrame'
                                                                     'should correspond to segments in the dataset')
    args = parser.parse_args()

    # Load metadata (used for grouping data when extracting features at different levels)
    meta_df = pd.read_pickle(args.metadata_file_path)

    # Extract features for all transcripts
    feature_df = extract_features(args.input_dir, args.level, meta_df)
    feature_df.to_pickle("asr_confidence.pkl")


if __name__ == '__main__':
    main()
