import itertools
from collections import Counter

import liwc
import numpy as np

from text_features.config import Config


"""
Contains functions to compute LIWC measures. This includes proportions of words in each category in the LIWC 2007
dictionary as well as linguistic process measures computed as part of the LIWC tool (i.e. word count, etc.).
"""

config = Config()
PARSE, CAT_NAMES = liwc.load_token_parser(config.LIWC_2007_PATH)


def extract_liwc_feats(segments):
    """
    Computes LIWC features for list text segments and stores in dictionary.
    :param segments: List of text segments, where each segment is a string. Segments are used to determine what
                     words are consecutive in order to identify bigrams + trigrams.
    :return: feats_dict: Dictionary mapping feature name to value for transcript
    """
    # compute feature values
    feats_dict = {}
    segments = [s.split(" ") for s in segments]
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
    return feats_dict
