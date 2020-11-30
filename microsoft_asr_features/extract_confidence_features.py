import os
import argparse
import pandas as pd
import numpy as np

from IPython import embed 

"""
Script for extracting speech intelligibility features (i.e. ASR confidence) from Microsoft recognition_results.csv 
"""


class MicrosoftASRConfidenceFeatureExtractor:
    def __init__(self, recognition_result_files):
        """
        :param recognition_result_files: list of paths to files containing recognition results from Microsoft
                                         speech-to-text model
        """
        self.recognition_result_files = recognition_result_files
        self.conf_dict = self._collect_conf()

    def _collect_conf(self):
        """
        Collect all confidence scores for recognizer entries that have the same 'feature_id'
        (or 'audio_file_id' if 'feature_id' is not present)
        :return: text_dict: dict mapping ids ('feature_id' or 'audio_file_id') to dict storing the text and basic text
                            (punctuation + capitalization removed) outputs from each recognizer entry with that id.
                            Both the 'text' and 'text_basic' entries are lists of text strings for each segment
                            identified by the ASR model.
        """
        conf_dict = dict()
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
            conf_dict[idx] = {}
            if len(combined_df.loc[idx].shape) == 1:
                sorted_entries = combined_df.loc[idx, :]
                conf_dict[idx]['confidence'] = [sorted_entries['confidence']]
            else:
                sorted_entries = combined_df.loc[idx].sort_values(sort_column)
                conf_dict[idx]['confidence'] = sorted_entries['confidence'].values
        return conf_dict

    def extract_conf_feats(self, output_dir):
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
    conf_feat_extractor = MicrosoftASRConfidenceFeatureExtractor(recognition_result_files)
    conf_feat_extractor.extract_conf_feats(args.output_dir)


if __name__ == '__main__':
    main()
