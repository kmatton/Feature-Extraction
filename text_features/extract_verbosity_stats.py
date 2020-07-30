import numpy as np
from textstat.textstat import textstat


"""
Extract statistics related to speech verbosity and complexity.
"""


def get_word_stats(segments, feats_dict):
    """
    Compute statistics related to number of words used and length of words used.
    :param segments: list of segments, where each segment is a list of words
    :param feats_dict: dictionary to store computed feature values
    """
    word_count_list = []
    word_lengths = []
    long_count = 0
    for segment in segments:
        word_count_list.append(len(segment))
        for word in segment:
            word_lengths.append(len(word))
            if len(word) > 6:
                long_count += 1
    # Compute segment level statistics
    feats_dict['wc_mean'] = np.mean(word_count_list) if word_count_list else float('nan')
    feats_dict['wc_median'] = np.median(word_count_list) if word_count_list else float('nan')
    feats_dict['wc_stdev'] = np.std(word_count_list) if word_count_list else float('nan')
    feats_dict['wc_min'] = min(word_count_list) if word_count_list else float('nan')
    feats_dict['wc_max'] = max(word_count_list) if word_count_list else float('nan')
    feats_dict['total_count'] = sum(word_count_list) if word_count_list else float('nan')

    # Compute fraction of words across whole call that are long (i.e. 6+ words)
    feats_dict['lw_count'] = (long_count / feats_dict['total_count']) if feats_dict['total_count'] else float('nan')
    # Compute mean length of any word used
    feats_dict['word_len'] = np.mean(word_lengths) if word_lengths else float('nan')


def get_syll_stats(segments, feats_dict):
    """
    Computes statistics related to number of syllables present in each word in transcript.
    :param segments: list of segments, where each segment is a list of words
    :param feats_dict: dictionary to store computed feature values
    """
    syll_count_list = []
    for segment in segments:
        for word in segment:
            syll_count_list.append(textstat.syllable_count(word))
    feats_dict['syll_mean'] = np.mean(syll_count_list) if syll_count_list else float('nan')
    feats_dict['syll_median'] = np.median(syll_count_list) if syll_count_list else float('nan')
    feats_dict['syll_stdev'] = np.std(syll_count_list) if syll_count_list else float('nan')
    feats_dict['syll_min'] = min(syll_count_list) if syll_count_list else float('nan')
    feats_dict['syll_max'] = max(syll_count_list) if syll_count_list else float('nan')


def get_verbosity_stats(segments):
    """
    Computes verbosity + complexity measures and stores in dict.
    :param segments: list of text segments, where each segment is a string
    :return: feats_dict: dict storing computed feature values
    """
    # break up segments into words
    segments = [s.split(" ") for s in segments]
    feats_dict = {}
    get_word_stats(segments, feats_dict)
    get_syll_stats(segments, feats_dict)
    return feats_dict
