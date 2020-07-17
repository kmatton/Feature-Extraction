import numpy as np
import preprocess_text as pt


"""
Contains functions to compute measures of lexical diversity. This includes MATTR (moving average type-token ratio) 
and Honore's Statistic.
"""


def compute_MATTR(words, feats_dict, window):
    """
    Computes MATTR for given window size.
    :param words: list of all words in transcript
    :param feats_dict: dictionary to store feature values for the transcript
    :param window: size of window to compute each TTR over.
    If window size is larger than number of words across all segments, uses total number of words as window size
    and prints warning message.
    """
    original_window = window
    if len(words) == 0:
        # if transcript is empty, MATTR is not defined
        feats_dict["MATTR_{}".format(original_window)] = float('nan')
        return
    if len(words) < window:
        print("WARNING: window size {} greater than number of words in transcript {}. Using window size {}.".format(
            window, len(words), len(words)))
        window = len(words)
    # store counts of words in current window
    vocab_dict = {}
    # add words from first window
    for i in range(window):
        word = words[i]
        if word not in vocab_dict:
            vocab_dict[word] = 0
        vocab_dict[word] += 1
    # store list of TTR for each window
    ttrs = [len(vocab_dict.keys())/float(window)]
    # keep track of first word in window (will be the one word not also in next window)
    first_word = words[0]
    for i in range(1, len(words) - window + 1):
        # remove instance of first word in previous window
        vocab_dict[first_word] -= 1
        if vocab_dict[first_word] == 0:
            del vocab_dict[first_word]
        first_word = words[i]
        # add word that wasn't in previous window (last word in this window)
        last_word = words[i + window - 1]
        if last_word not in vocab_dict:
            vocab_dict[last_word] = 0
        vocab_dict[last_word] += 1
        ttrs.append(len(vocab_dict.keys())/float(window))
    feats_dict["MATTR_{}".format(original_window)] = np.mean(ttrs)


def get_honores_statistic(words, feats_dict):
    """
    Computes Honore's Statistic, a measure which emphasizes the use of words that are only used once.
    :param words: list of all words in transcript
    :param feats_dict: dictionary to store feature values for the transcript
    """
    total_words = len(words)
    unique_words = len(set(words))
    single_time_words = len([word for word in words if words.count(word) == 1])
    if total_words == 0:
        feats_dict["HS"] = float('nan')
        return
    # smooth statistic so not undefined when # unique words = # single time words
    epsilon = 1e-5
    feats_dict["HS"] = 100 * np.log(total_words / float(1 - single_time_words / float(unique_words + epsilon)))


def extract_lexical_diversity_feats(transcript):
    """
    Computes lexical diversity features for list of transcript segments and stores in dictionary.
    :param transcript: List of transcript segments. Each segment is represented as a tuple of the form (start time,
    stop time, text).
    :return: Dictionary mapping feature name to value for transcript
    """
    # first pre-process text
    # remove can ignore timing info because not relevant to computing lexical diversity measures
    transcript = [seg[2] for seg in transcript]
    # split into words
    segments = [seg.strip().split(" ") for seg in transcript]
    # remove non-verbal expressions
    segments = pt.remove_non_verbal_exp(segments)

    words = [word for segment in segments for word in segment]
    feats_dict = {}
    for window in [10, 25, 50]:
        compute_MATTR(words, feats_dict, window)
    get_honores_statistic(words, feats_dict)
    return feats_dict
