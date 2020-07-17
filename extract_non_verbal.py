import numpy as np


def extract_non_verbal_feats(transcript):
    """
    Computes ratio of non-verbal expressions and unknown words to total items in transcript.
    :param transcript: List of transcript segments. Each segment is represented as a tuple of the form (start time,
    stop time, text).
    :return: feature_dict: Dictionary mapping feature name to value for transcript
    """
    # first pre-process text
    # remove can ignore timing info because not relevant to computing LIWC
    transcript = [seg[2] for seg in transcript]
    # split into words
    segments = [seg.strip().split(" ") for seg in transcript]

    feature_dict = {}
    tokens = [token for segment in segments for token in segments]
    total_count = len(tokens)
    feature_dict['laughter'] = tokens.count('[laughter]') / total_count if total_count else float('nan')
    feature_dict['noise'] = tokens.count('[noise]') / total_count if total_count else float('nan')
    feature_dict['unk'] = tokens.count('<unk>') / total_count if total_count else float('nan')
    return feature_dict
