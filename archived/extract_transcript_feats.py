import os
import argparse
import pandas as pd
import numpy as np
import extract_graph
import extract_liwc_2007
import extract_non_verbal
import extract_lexical_diversity
import extract_pos

"""
Module for extracting features from call transcript files.
"""


FEATURE_FUNCTION_DICT = {"LIWC": extract_liwc_2007.extract_LIWC_feats,
                         "graph": extract_graph.extract_graph_feats,
                         "lexical_diversity": extract_lexical_diversity.extract_lexical_diversity_feats,
                         "POS": extract_pos.extract_pos_features,
                         "non_verbal": extract_non_verbal.extract_non_verbal_feats}


def get_subject_data(transcript_dir, sub_meta_df):
    """
    Collect data for subject <sub_id>.
    :param transcript_dir: Directory containing transcripts.
    :param sub_meta_df: dataframe containing metadata information for subject with sub_id
    :return: sub_data_df: DataFrame with columns: "week_id", "call_id", "segment_start", "segment_end", and
    "segment_hypotheses" (stores list of text hypotheses for each segment)
    """
    sub_data = []
    call_ids = sub_meta_df["call_id"].values
    for call_id in call_ids:
        if os.path.exists(os.path.join(transcript_dir, call_id)):
            seg_hyp_dict = {}
            for transcript_fp in os.listdir(os.path.join(transcript_dir, call_id)):
                transcript_file = open(transcript_fp)
                segments = transcript_file.readlines()
                transcript_file.close()
                for segment in segments:
                    seg_id = segment.strip().split(" ")[0]
                    seg_text = segment.strip().split(" ")[1:]
                    if seg_id not in seg_hyp_dict:
                        # store list of segment hypotheses
                        seg_hyp_dict[seg_id] = []
                    seg_hyp_dict[seg_id].append(seg_text)
            # add entry to sub_data with segment hypotheses
            week = sub_meta_df[sub_meta_df["call_id"] == call_id].week
            date = sub_meta_df[sub_meta_df["call_id"] == call_id].date
            time = sub_meta_df[sub_meta_df["call_id"] == call_id].time
            for seg_id, seg_hyps in seg_hyp_dict.items():
                segment_start, segment_end = seg_id.split("_")[2:]
                sub_data.append({"week": week, "call_id": call_id, "date": date, "time": time, "segment_id": seg_id,
                                 "segment_start": int(segment_start), "segment_end": int(segment_end),
                                 "segment_hypotheses": seg_hyps})
    # create dataframe from list of dicts
    sub_data_df = pd.DataFrame(sub_data)
    return sub_data_df


def merge_segments_by_hyp(data_df):
    """
    :param data_df: Dataframe containing all segment entries (rows) to be merged. For each row, the Dataframe should
    have the following columns: "segment_start", "segment_end", and "segment_hypotheses" (list of text hypotheses for
    the given segment). In order to preserve order of segments, they should be sorted within data_df.
    :return: transcript_hyps: List of merged transcript hypotheses. Length is equal to the number of different
    ASR hypotheses produced for th given transcript. Each transcript hypothesis in the list is a list containing
    tuples of the form (segment start, segment end, segment text).
    """
    num_hyps = len(data_df.idx[0]["segment_hypotheses"])
    transcript_hyps = []
    for hyp_num in range(num_hyps):
        # collect all segments from a single ASR hypothesis (i.e. parameter setting)
        transcript_hyps.append([(row["segment_start"], row["segment_end"], row["segment_hypotheses"][hyp_num])
                                for index, row in data_df.iterrows()])
    return transcript_hyps


def collect_transcript_by_level(sub_id, sub_data_df, level):
    """
    :param sub_id: id of subject associated with sub_data_df
    :param sub_data_df: Dataframe containing segment text hypotheses for all call data from a single subject.
    :param level: Level to group/aggregate data by.
    :return: sub_transcript_list: list of transcripts for subject, where transcript is a tuple of the
    form (transcript id items (e.g. "callid" or ["subject", "week"], "transcript_id", list of transcript hypotheses),
    and transcripts are collected at the <level> level
    """
    # sort segments by date, time, segment start time so that transcriptions are in order
    sub_data_df.sort(['date', 'time', 'segment_start'])
    sub_transcript_list = []
    if level == "subject":
        # group all data into one transcript
        transcript_hyps = merge_segments_by_hyp(sub_data_df)
        sub_transcript_list.append((sub_id, transcript_hyps))
    else:
        # group transcript based on data level
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
            transcript_hyps = merge_segments_by_hyp(df)
            sub_transcript_list.append((group_by_list, group_id, transcript_hyps))
    return sub_transcript_list


def extract_features(transcript_dir, level, feat_extract_fn, meta_df):
    """
    Extract features for groups of transcripts aggregated at <level> level
    and return as Dataframe with transcript id (determined based on data level) as index and features as columns.
    :param transcript_dir: Directory containing subdirectories for each call, where each call directory contains ASR
     hypothesis word sequences (transcripts).
    :param level: Level (segment, call, day, week, subject) to group data by before extracting features.
    :param feat_extract_fn: function that takes a single transcript as input and outputs dictionary mapping feature to
    value
    :param meta_df: Dataframe containing metadata information.
    :return: Dataframe containing average features for each transcript across all hypotheses
    """
    # feature_list is is list that contains a feat dict for each transcript
    # each feat dict contains feature entries as well as information needed to uniquely identify transcript
    feature_list = []
    sub_ids = meta_df["subject_id"].index.values
    for sub_id in sub_ids:
        sub_meta_df = meta_df[meta_df["subject_id"] == sub_id]
        sub_data_df = get_subject_data(transcript_dir, sub_meta_df)
        # group segments based on specified data level
        sub_transcript_list = collect_transcript_by_level(sub_id, sub_data_df, level)
        for id_elms, transcript_id, transcript_hyps in sub_transcript_list:
            transcript_feature_dicts = []
            for hyp in transcript_hyps:
                hyp_feature_dict = feat_extract_fn(hyp)
                transcript_feature_dicts.append(hyp_feature_dict)
            feats = transcript_feature_dicts[0].keys()
            # aggregate features across hypotheses (take mean) to get single set of features for the given transcript
            transcript_feature_dict = {}
            for feat in feats:
                transcript_feature_dict[feat] = np.mean([d[feat] for d in transcript_feature_dicts])
            for idx, id_elm in enumerate(id_elms):
                transcript_feature_dict[id_elm] = transcript_id[idx]
            feature_list.append(transcript_feature_dict)
    feature_df = pd.DataFrame(feature_list)
    return feature_df


def main():
    # Read in and process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transcript_dir', type=str, help='Directory containing call transcript subdirectories')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to output feature csv file')
    parser.add_argument('-l', '--level', type=str, help='Level of data to extract features for. Options are: segment,'
                                                        'call, day, week, subject')
    parser.add_argument('-f', '--feature_type', type=str, help='Feature type to extract. Options are: LIWC, graph, '
                                                               'lexical_diversity, POS, non_verbal')
    parser.add_argument('-m', '--metadata_file_path', type=str, help='Path to pickled DataFrame containing metadata '
                                                                     'information, including mapping between '
                                                                     'subject_ids, call_ids, assess weeks, dates,'
                                                                     'times, and segment_ids')
    args = parser.parse_args()

    # Determine feature extraction function based on feature_type argument
    feat_extract_fn = None
    if args.feature_type in FEATURE_FUNCTION_DICT:
        feat_extract_fn = FEATURE_FUNCTION_DICT[args.feature_type]
    else:
        print("ERROR Invalid feature type: {}".format(args.feature_type))
        exit()

    # Load metadata (used for grouping data when extracting features at different levels)
    meta_df = pd.read_pickle(args.metadata_file_path)

    # Extract features for all transcripts
    feature_df = extract_features(args.transcript_dir, args.level, feat_extract_fn, meta_df)
    feature_df.to_pickle("{}.pkl".format(args.feature_type))


if __name__ == '__main__':
    main()
