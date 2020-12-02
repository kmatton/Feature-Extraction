import numpy as np

from IPython import embed 

"""
Contains functions to parse word-phone aligned timing file information (produced by Kaldi) and to
compute timing related features.
"""

def get_feats_from_times(times_dict):
    """
    Extract timing features from timing information collected in times_dict.
    :param times_dict: dictionary that stores durations of segments, words, phones, and silences, as well as
    words per second and phone per second rates for each segment.
    :return:
    """
    #Check if times_dict contains "phones" 
    sound_types =  ["segments", "silences", "words", "wps"]
    if 'phones' in set(times_dict.keys()):
        sound_types =  ["segments", "silences", "words", "phones", "wps", "pps"]        

    feat_dict = {}
    # segment lengths are in seconds, but silences, words, and phones are in ms
    for sound_type in sound_types:
        times = times_dict[sound_type]
        feat_dict["{}_max".format(sound_type)] = max(times) if times else float('nan')
        feat_dict["{}_min".format(sound_type)] = min(times) if times else float('nan')
        feat_dict["{}_mean".format(sound_type)] = np.mean(times) if times else float('nan')
        feat_dict["{}_med".format(sound_type)] = np.median(times) if times else float('nan')
        feat_dict["{}_std".format(sound_type)] = np.std(times) if times else float('nan')
    
    sil_duration = np.sum(times_dict["silences"]) * 0.001 # convert to seconds
    feat_dict["sil_duration"] = sil_duration
    spk_duration = np.sum(times_dict["segments"])
    feat_dict["spk_duration"] = spk_duration
    feat_dict["spk_sil_ratio"] = spk_duration / sil_duration if sil_duration else float('nan')
    feat_dict["sps"] = len(times_dict["silences"]) / spk_duration if spk_duration else float('nan')
    feat_dict["wps"] = len(times_dict["words"]) / spk_duration if spk_duration else float('nan')
    feat_dict["sil_count"] = len(times_dict["silences"])
    feat_dict["short_utt_count"] = len([x for x in times_dict["segments"] if x <= 1])
    feat_dict["segment_count"] = len(times_dict["segments"])
    feat_dict["word_count"] = len(times_dict["words"])
    if 'phones' in set(times_dict.keys()):
        feat_dict["pps"] = len(times_dict["phones"]) / spk_duration if spk_duration else float('nan')
        feat_dict["phone_count"] = len(times_dict["phones"])
    return feat_dict


def get_feats(times_dict, total_duration):
    """
    :param times_dict: dictionary of features computed from segment, word, and phone timing
    :param total_duration: total duration (SECONDS) of "call" (defined based on data grouping level), including time when
    participant isn't speaking 
    :return: feat_dict: dictionary mapping timing features to values
    """
    feat_dict = get_feats_from_times(times_dict)

    # add in features related to total call timing
    feat_dict["total_duration"] = total_duration
    #feat_dict['total_dur_sec'] = total_duration 
    feat_dict["spk_ratio"] = feat_dict["spk_duration"] / total_duration if total_duration else float('nan')
    feat_dict["sil_ratio"] = feat_dict["sil_duration"] / total_duration if total_duration else float('nan')
    if total_duration:
        feat_dict["segs_per_min"] = len(times_dict["segments"]) / float(total_duration / 60.0)
        # NOTE: could change this feature to compute number of utterances with respect to only time identified as
        # speech segments from SAD / only during sections of the conversation where participant is speaking
        feat_dict["short_utts_per_min"] = feat_dict["short_utt_count"] / float(total_duration / 60.0)
    else:
        feat_dict["segs_per_min"] = float('nan')
        feat_dict["short_utts_per_min"] = float('nan')
    return feat_dict
