import os
import argparse
import pandas as pd
import numpy as np

from IPython import embed 

"""
Script for extracting timing features from Microsoft recognition_results.csv 
"""


class MicrosoftTimingFeatureExtractor:
    def __init__(self, recognition_result_files):
        """
        :param recognition_result_files: list of paths to files containing recognition results from Microsoft
                                         speech-to-text model
        """
        self.recognition_result_files = recognition_result_files
        self.conf_dict = self._collect_timing()

    def _collect_timing(self):
        """
        Collect all timing information for recognizer entries that have the same 'feature_id'
        (or 'audio_file_id' if 'feature_id' is not present)
        :return: text_dict: dict mapping ids ('feature_id' or 'audio_file_id') to dict storing the text and basic text
                            (punctuation + capitalization removed) outputs from each recognizer entry with that id.
                            Both the 'text' and 'text_basic' entries are lists of text strings for each segment
                            identified by the ASR model.
        """
        time_dict = dict()
        df_list = []
        # collect results from all files
        for file_path in self.recognition_result_files:
            df = pd.read_csv(file_path)
            df_list.append(df)
        combined_df = pd.concat(df_list)
        if 'feature_id' in combined_df.columns:
            combined_df.set_index('feature_id', inplace=True)
        else:
            combined_df.set_index('audio_file_id', inplace=True)
        sort_column = "order" if "order" in combined_df.columns else "segment_number"
        for idx in set(combined_df.index.values):
            #print(idx)
            time_dict[idx] = self.get_times(combined_df)
            

            #columns
            #************
            #duration - of segment, what is the unit?
            #offset - of segment, what is the unit?
            #word_timing - string "[{'Duration':#, 'Offset':#, 'Word':'spoken_word'}, ...]"

            
            #if len(combined_df.loc[idx].shape) == 1:
            #    sorted_entries = combined_df.loc[idx, :]
            #    time_dict[idx]['confidence'] = [sorted_entries['confidence']]
            #else:
            #    sorted_entries = combined_df.loc[idx].sort_values(sort_column)
            #    time_dict[idx]['confidence'] = sorted_entries['confidence'].values
        return time_dict


    def get_times(self, combined_df):
        """
        Collect overall timing information of segments, silences, words, and phones. Also collect words per second and
        phones per second rates for each segment.
        :param combed_df: dataFrame of word-level timing information. Each row includes duration, offset information 
        for each segment and word within each segment. 
        :return: times_dict: dictionary that stores timing information extracted from combed_df. Entries
        are lists of the durations of "segments", "silences", and "words", as well as lists of
        "wps" (words per second) for each segment.
        """
        times_dict = {"segments": [], "silences": [], "words": [], "wps": []}
        for s in range(0, combined_df.shape[0]):
            seg = combined_df.iloc[s]
            #word_count = 0
            
            times_dict['segments'] = seg['duration'] #TODO convert to seconds from ???

            #TODO MAKE THIS A FUNCTION **** 
            word_timing_list = []
            word_timing_str = seg['word_timing'].strip('][').split("},")
            for ws in word_timing_str:
                word_dict = {}
                ws = ws.strip(" ")
                dl = ws.strip("}{").split(",")
                for d in dl:
                    d2 = d.split(":")
                    k = d2[0].strip(" ").strip("''")
                    #k = d2[0].strip("''")
                    v = d2[1].strip(" ")
                    if v.isnumeric(): #Duration, Offset are integers
                        v = int(v) 
                    else: #Word is a string
                        v = v.strip("''")
                    word_dict[k] = v
                word_timing_list.append(word_dict)  
            #***************************************************          
        
            #word_start_time = -1
            #for phone_info in segment_info:
            #    phone_info = phone_info.strip()
            #    items = phone_info.split(" ")
            #    if len(items) == 5:
            #        # new word or silence
            #        if word_start_time != -1:
            #            # we have a previous word
            #            word_end_time = int(items[0])
            #            word_len = (word_end_time - word_start_time) * 25 # 25 ms per frame
            #            times_dict["words"].append(word_len)
            #        word = items[4]
            #        if word == "[noise]" or word == "[laughter]":
            #            word_start_time = -1
            #            continue
            #        if word == "sil":
            #            sil_len = (int(items[1]) - int(items[0])) * 25
            #            times_dict["silences"].append(sil_len)
            #            word_start_time = -1
            #            continue
            #        word_start_time = int(items[0])
            #        word_count += 1
            #    phone_len = (int(items[1]) - int(items[0])) * 25
            #    phone_count += 1
            #    times_dict["phones"].append(phone_len)
            ## add in info for last word if there is one
            ## (can't just use presence of new word to indicate that this word ended)
            #phone_info = segment_info[-1]
            #phone_info = phone_info.strip()
            #items = phone_info.split(" ")
            #if word_start_time != -1:
            #    word_end_time = int(items[1])  # the last word that has been seen ends here
            #    word_len = (word_end_time - word_start_time) * 25
            #    times_dict["words"].append(word_len)
            #if word_count == 0:
            #    continue  # empty segment
            #seg_duration = float(int(items[1])) * 25 * .001  # convert ms to seconds
            #times_dict["segments"].append(seg_duration)
            #times_dict["wps"].append(word_count / seg_duration)
            #times_dict["pps"].append(phone_count / seg_duration)
        
        return times_dict




    def extract_timing_feats(self, output_dir):
        """
        Extract ASR confidence features for each entry in self.conf_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        conf_feats = []
        for key, val in self.conf_dict.items():
            conf = val['confidence']
            feats_dict = self.get_conf_feats(conf)
            conf_feats.append(feats_dict)
        feats_df = pd.DataFrame(conf_feats)
        feats_df.to_csv(os.path.join(output_dir, "asr_conf_features.csv"))
    
    def get_conf_feats(self, conf_scores):
        feat_dict = {}
        feat_dict["conf_max"] = max(conf_scores)
        feat_dict["conf_mean"] = np.mean(conf_scores)
        feat_dict["conf_std"] = np.std(conf_scores)
        feat_dict["conf_min"] = min(conf_scores)
        feat_dict["conf_med"] = np.median(conf_scores)
        return feat_dict


def _read_file_by_lines(filename):
    """
    Read a file into a list of lines
    """
    with open(filename, "r") as f:
        return f.read().splitlines()

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_asr_output_files', type=str,
                        help="Path to text file containing paths to the Microsoft speech-to-text recognition results "
                             "that you want to extract features from. Files should be CSV files as produced by the "
                             "asr-models-support/Microsoft/speech_to_text.py script in "
                             "https://github.com/kmatton/ASR-Helper.")
    parser.add_argument('--output_dir', type=str, help="Path to directory to output feature files to.")
    return parser.parse_args()

def main():
    args = _parse_args()
    recognition_result_files = _read_file_by_lines(args.ms_asr_output_files)
    timing_feat_extractor = MicrosoftTimingFeatureExtractor(recognition_result_files)
    timing_feat_extractor.extract_timing_feats(args.output_dir)


if __name__ == '__main__':
    main()
