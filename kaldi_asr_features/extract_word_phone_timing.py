import numpy as np

"""
Contains functions to parse word-phone aligned timing file information (produced by Kaldi) and to
compute timing related features.
"""


def get_times(timing_info):
    """
    Collect overall timing information of segments, silences, words, and phones. Also collect words per second and
    phones per second rates for each segment.
    :param timing_info: List of phone-level timing information. Each element in list gives
    information related to the timing of a single phone.
    :return: times_dict: dictionary that stores timing information extracted from timing_info list. Entries
    are lists of the durations of "segments", "silences", "words", and "phones", as well as lists of
    "wps" (words per second) and "pps" (phones per second) for each segment.
    """
    times_dict = {"segments": [], "silences": [], "words": [], "phones": [], "wps": [], "pps": []}
    for segment_info in timing_info:
        word_count = 0
        phone_count = 0
        word_start_time = -1
        for phone_info in segment_info:
            phone_info = phone_info.strip()
            items = phone_info.split(" ")
            if len(items) == 5:
                # new word or silence
                if word_start_time != -1:
                    # we have a previous word
                    word_end_time = int(items[0])
                    word_len = (word_end_time - word_start_time) * 25 # 25 ms per frame
                    times_dict["words"].append(word_len)
                word = items[4]
                if word == "[noise]" or word == "[laughter]":
                    word_start_time = -1
                    continue
                if word == "sil":
                    sil_len = (int(items[1]) - int(items[0])) * 25
                    times_dict["silences"].append(sil_len)
                    word_start_time = -1
                    continue
                word_start_time = int(items[0])
                word_count += 1
            phone_len = (int(items[1]) - int(items[0])) * 25
            phone_count += 1
            times_dict["phones"].append(phone_len)
        # add in info for last word if there is one
        # (can't just use presence of new word to indicate that this word ended)
        phone_info = segment_info[-1]
        phone_info = phone_info.strip()
        items = phone_info.split(" ")
        if word_start_time != -1:
            word_end_time = int(items[1])  # the last word that has been seen ends here
            word_len = (word_end_time - word_start_time) * 25
            times_dict["words"].append(word_len)
        if word_count == 0:
            continue  # empty segment
        seg_duration = float(int(items[1])) * 25 * .001  # convert ms to seconds
        times_dict["segments"].append(seg_duration)
        times_dict["wps"].append(word_count / seg_duration)
        times_dict["pps"].append(phone_count / seg_duration)
    return times_dict


def get_feats_from_times(times_dict):
    """
    Extract timing features from timing information collected in times_dict.
    :param times_dict: dictionary that stores durations of segments, words, phones, and silences, as well as
    words per second and phone per second rates for each segment.
    :return:
    """
    feat_dict = {}
    # segment lengths are in seconds, but silences, words, and phones are in ms
    for sound_type in ["segments", "silences", "words", "phones", "wps", "pps"]:
        times = times_dict[sound_type]
        feat_dict["{}_max".format(sound_type)] = max(times) if times else float('nan')
        feat_dict["{}_min".format(sound_type)] = min(times) if times else float('nan')
        feat_dict["{}_mean".format(sound_type)] = np.mean(times) if times else float('nan')
        feat_dict["{}_med".format(sound_type)] = np.median(times) if times else float('nan')
        feat_dict["{}_std".format(sound_type)] = np.std(times) if times else float('nan')
    sil_duration = np.sum(times_dict["silences"]) * .001 # convert to seconds
    feat_dict["sil_duration"] = sil_duration
    spk_duration = np.sum(times_dict["segments"])
    feat_dict["spk_duration"] = spk_duration
    feat_dict["spk_sil_ratio"] = spk_duration / sil_duration if sil_duration else float('nan')
    feat_dict["sps"] = len(times_dict["silences"]) / spk_duration if spk_duration else float('nan')
    feat_dict["wps"] = len(times_dict["words"]) / spk_duration if spk_duration else float('nan')
    feat_dict["pps"] = len(times_dict["phones"]) / spk_duration if spk_duration else float('nan')
    feat_dict["phone_count"] = len(times_dict["phones"])
    feat_dict["sil_count"] = len(times_dict["silences"])
    feat_dict["short_utt_count"] = len([x for x in times_dict["segments"] if x <= 1])
    feat_dict["segment_count"] = len(times_dict["segments"])
    feat_dict["word_count"] = len(times_dict["words"])
    return feat_dict


def get_feats(timing_info, total_duration):
    """
    :param timing_info: List of phone-level timing information. Each element in list gives
    information related to the timing of a single phone.
    :param total_duration: total duration of "call" (defined based on data grouping level), including time when
    participant isn't speaking - units should be seconds
    :return: feat_dict: dictionary mapping timing features to values
    """
    times_dict = get_times(timing_info)
    feat_dict = get_feats_from_times(times_dict)
    # add in features related to total call timing
    feat_dict["total_duration"] = total_duration
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
