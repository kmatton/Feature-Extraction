import extract_liwc
import numpy as np


"""
Contains functions to compute rumination features.
"""


STAT_FN_DICT = {'mean': np.mean, 'max': max, 'min': min, 'median': np.median, 'std': np.std}


"""
NOTES: other feature options
Correlation between LIWC categories and emotion values
MI between LIWC emotion categories and other categories i.e. salient categories w.r.t. each emotion category
Adding how presence of laughter and also speech intelligibility change in segments with self-references
Some kind of text similarity score across instances of "sad" speech segments i.e. are you repeatedly sad about the
same thing?
"""


def get_self_perception_measures(segments_liwc, segments_emotion, feature_dict, num_hyps=1):
    """
    Computes statistics from emotion values of segments that contain personal pronouns.
    :param segments_liwc: List of dictionaries containing LIWC features for each segment.
    :param segments_emotion: List of dictionaries containing emotion features for each speech segment.
    :param feature_dict: Dictionary mapping feature name to value for transcript.
    :param num_hyps: Number of hypotheses per segment
    """
    # record emotion values that occur for segments that contain self-references
    liwc_emotions = ['affect', 'posemo', 'negemo', 'anx', 'anger', 'sad']
    dim_emotions = ['act_low', 'act_mid', 'act_high', 'val_low', 'val_mid', 'val_high']
    liwc_emo_dict = {emo: [] for emo in liwc_emotions}
    dim_emo_dict = {emo: [] for emo in dim_emotions}
    self_references = 0
    for liwc_dict, seg_emotion in zip(segments_liwc, segments_emotion):
        # get personal pronoun usage
        # for now not also considering if there is a reference to another person in the sentence
        # this is a very naive way of doing this so may want to adapt in the future
        if liwc_dict['ppron_liwc'] < 1 / float(num_hyps):
            # if there is on average less than one personal pronoun per hypothesis don't count as self-reference
            continue
        self_references += 1
        for emo in liwc_emotions:
            liwc_emo_dict[emo].append(liwc_dict['{}_liwc'.format(emo)])
        for emo in dim_emotions:
            dim_emo_dict[emo].append(seg_emotion[emo])
    for stat_name, stat_fn in STAT_FN_DICT.items():
        for emo in liwc_emotions:
            feature_dict['i_{}_liwc_{}'.format(emo, stat_name)] = stat_fn(liwc_emo_dict[emo])
        for emo in dim_emotions:
            feature_dict['i_{}_{}'.format(emo, stat_name)] = stat_fn(dim_emo_dict[emo])


def extract_rumination_feats(segments_text, segments_emotion):
    """
    Computes rumination features for list of transcript segments (and associated emotion predictions)
    and stores in dictionary.
    :param segments_text: List of transcript hypotheses for each speech segment.
           Each segment hypothesis is represented as a list of words.
    :param segments_emotion: List of emotion features for each speech segment.
    :return: feature_dict: Dictionary mapping feature name to value for transcript.
    """
    feature_dict = {}
    # get liwc categories for each segment
    # initialize empty dict to store liwc features for each segment
    segments_liwc = [{}] * len(segments_text)
    for segment_hyps, liwc_dict in zip(segments_text, segments_liwc):
        # note: extracting features in this way gives equal weight to each word rather than
        # each hypothesis as I did before for call-level LIWC features
        # However lengths of each hyp should be similar, so shouldn't make a big difference
        extract_liwc.get_LIWC_cat_vals(segment_hyps, liwc_dict)
    num_hyps = len(segments_text[0])
    get_self_perception_measures(segments_liwc, segments_emotion, feature_dict, num_hyps)
    return feature_dict
