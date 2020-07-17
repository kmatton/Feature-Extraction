from collections import Counter
import liwc
from textstat.textstat import textstat
import numpy as np
import itertools
import preprocess_text as pt


"""
Contains functions to compute LIWC measures. This includes proportions of words in each category in the LIWC 2007
dictionary as well as linguistic process measures computed as part of the LIWC tool (i.e. word count, etc.).
"""


PARSE, CAT_NAMES = liwc.load_token_parser('LIWC2007_English_adapted.dic')
# took out kind <of> and like from dictionary due to inconsistent formatting with rest of dictionary


def get_words_stats(segments, feats_dict):
    """
    Compute statistics related to number of words used and length of words used.
    :param segments: list of segments, where each segment is a list of words
    :param feats_dict: dictionary to store feature values for the transcript
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
    :param feats_dict: dictionary to store feature values for the transcript
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


def get_LIWC_cat_vals(segments, feats_dict):
    """
    Computes proportions of words in transcript in each LIWC dictionary category
    :param segments: list of segments, where each segment is a list of words
    :param feats_dict: dictionary to store feature values for the transcript
    """
    words = list(itertools.chain.from_iterable(segments))
    # Generate lists of all bigrams and trigrams because some are in LIWC vocabulary (.e.g "is don't know", "you know")
    bigrams = []
    trigrams = []
    for segment in segments:
        for i in range(len(segment) - 1):
            bigrams.append(segment[i]+" "+segment[i+1])
            if i < len(segment) - 2:
                trigrams.append(segment[i]+" "+segment[i+1]+" "+segment[i+2])
    num_words = float(len(words))
    cat_counts = Counter(category for word in words for category in PARSE(word))
    b_cat_counts = Counter(category for bigram in bigrams for category in PARSE(bigram))
    t_cat_counts = Counter(category for trigram in trigrams for category in PARSE(trigram))
    for cat in CAT_NAMES:
        if not num_words:
            feats_dict[cat+'_liwc'] = np.NaN
        else:
            feats_dict[cat+'_liwc'] = 0
            if cat in cat_counts:
                feats_dict[cat+'_liwc'] += float(cat_counts[cat])
            if cat in b_cat_counts:
                feats_dict[cat+'_liwc'] += float(b_cat_counts[cat])
            if cat in t_cat_counts:
                feats_dict[cat+'_liwc'] += float(t_cat_counts[cat])
            # normalize by number of words in transcript
            # want bigram/trigram keywords to have same weight as single words
            feats_dict[cat+'_liwc'] /= num_words


def extract_LIWC_feats(transcript):
    """
    Computes LIWC features for list of transcript segments and stores in dictionary.
    :param transcript: List of transcript segments. Each segment is represented as a tuple of the form (start time,
    stop time, text).
    :return: feature_dict: Dictionary mapping feature name to value for transcript
    """
    # first pre-process text
    # remove can ignore timing info because not relevant to computing LIWC
    transcript = [seg[2] for seg in transcript]
    # split into words
    segments = [seg.strip().split(" ") for seg in transcript]
    # remove non-verbal expressions
    segments = pt.remove_non_verbal_exp(segments)

    # compute feature values
    feature_dict = {}
    get_LIWC_cat_vals(segments, feature_dict)
    get_syll_stats(segments, feature_dict)
    get_words_stats(segments, feature_dict)
    return feature_dict
