import argparse
import os
import sys 
import re
import pandas as pd
from group_audio_files import add_feature_id 

from IPython import embed

sys.path.append(os.getcwd()) 
sys.path.append('../text_features') #for slurm jobs  
from text_features.extract_graph import extract_graph_feats
from text_features.extract_lexical_diversity import extract_lexical_diversity_feats
from text_features.extract_liwc_2007 import extract_liwc_feats
from text_features.extract_pos import extract_pos_features
from text_features.extract_verbosity_stats import get_verbosity_stats
from text_features.text_util import remove_punctuation_except_apostrophes, split_into_sentences


"""
Script for extracting text features (i.e. those in text_features/) from the outputs of a Microsoft speech-to-text
model (https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/index-speech-to-text) as produced
by the asr-models-support/Microsoft/speech_to_text.py script in https://github.com/kmatton/ASR-Helper. 
More specifically, this script is designed to extract features from the recognizer.csv files produced by the
speech_to_text.py script.

Note: this script combines the text of entries with the same 'audio_file_id' within a recognizer.csv file, so that
features are computed for each audio_file (rather than segments or sentences within audio files). If you instead
want to extract features at a different level of the data (e.g. segment, day, or subject level features), add a 
'feature_id' column to your recognizer.csv files such that entries you want to be grouped before feature extraction
have the same 'feature_id'. For example, if you want to extract subject level features, provide the same 'feature_id'
to all audio file entries from the same subject. This script will look for a 'feature_id', but if not found will use 
the 'audio_file_id' instead. If you want the text results for entries with the same 'feature_id' to be combined in 
a particular order, also add an 'order' column to the recognition.csv files - items with the same 'feature_id' will
be sorted in ascending order based on the values in this column. If you do not provide an 'order' column, they
will be sorted based on the 'segment_number' column.
"""


class MicrosoftTextFeatureExtractor:
    def __init__(self, recognition_result_files, args):
        """
        :param recognition_result_files: list of paths to files containing recognition results from Microsoft
                                         speech-to-text model
        """
        self.recognition_result_files = recognition_result_files
        self.level = args.level 
        self.call_type = args.call_type
        self.metadata_path = args.metadata_path
        self.text_dict = self._collect_text()

    def _collect_text(self):
        """
        Collect all text for recognizer entries that have the same 'feature_id'
        (or 'audio_file_id' if 'feature_id' is not present)
        :return: text_dict: dict mapping ids ('feature_id' or 'audio_file_id') to dict storing the text and basic text
                            (punctuation + capitalization removed) outputs from each recognizer entry with that id.
                            Both the 'text' and 'text_basic' entries are lists of text strings for each segment
                            identified by the ASR model.
        """
        text_dict = dict()
        df_list = []
        # collect results from all files
        for file_path in self.recognition_result_files:
            df = pd.read_csv(file_path)
            df_list.append(df)
        combined_df = pd.concat(df_list)
        if self.level != None: 
            #add 'feature_id' column based on settings  
            combined_df = add_feature_id(combined_df, self.level, self.metadata_path, self.call_type)  
        if 'feature_id' in combined_df.columns:
            combined_df.set_index('feature_id', inplace=True)
        else:
            combined_df.set_index('audio_file_id', inplace=True)
        sort_column = "order" if "order" in combined_df.columns else "segment_number"
        for idx in set(combined_df.index.values):
            #print(idx)
            text_dict[idx] = {}
            if len(combined_df.loc[idx].shape) == 1:
                sorted_entries = combined_df.loc[idx, :]
                text_dict[idx]['text'] = [sorted_entries['text']]
                text_dict[idx]['text_basic'] = [sorted_entries['text_basic']]
            else:
                sorted_entries = combined_df.loc[idx].sort_values(sort_column)
                text_dict[idx]['text'] = sorted_entries['text'].values
                text_dict[idx]['text_basic'] = sorted_entries['text_basic'].values
        return text_dict

    def extract_features(self, feature_set_names, output_dir):
        """
        Extract features for each entry in self.text_dict.
        :param feature_set_names: names of feature sets to extract
        :param output_dir: path to directory to output feature files to
        """
        if "graph" in feature_set_names:
            print('extracting graph features')
            self.extract_graph_features(output_dir)
        if "lexical_diversity" in feature_set_names:
            print('extracting lexical diversity features')
            self.extract_lexical_diversity_features(output_dir)
        if "liwc" in feature_set_names:
            print('extracting LIWC features')
            self.extract_liwc_features(output_dir)
        if "pos" in feature_set_names:
            print('extracting POS features')
            self.extract_parts_of_speech_features(output_dir) 
        if "verbosity" in feature_set_names:
            print('extracting verbosity features')
            self.extract_verbosity_features(output_dir) 

    def extract_graph_features(self, output_dir):
        """
        Extract graph features for each entry in self.text_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        # keep segments as identified by speech-to-text model (i.e. don't break up sentences within utterances)
        # keep capitalization & keep apostrophes, but remove all other punctuation
        graph_feats = []
        for key, val in self.text_dict.items():
            text = val['text']
            text = [remove_punctuation_except_apostrophes(segment) for segment in text]
            feats_dict = extract_graph_feats(text)
            feats_dict['id'] = key
            graph_feats.append(feats_dict)
        feats_df = pd.DataFrame(graph_feats)
        feats_df = feats_df.set_index('id')
        feats_df = feats_df.reset_index()
        feats_df.to_csv(os.path.join(output_dir, "graph_features.csv"), index=False)

    def extract_lexical_diversity_features(self, output_dir):
        """
        Extract lexical diversity features for each entry in self.text_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        # use basic text version of ASR output (don't need capitalization or punctuation)
        # combine all segments into single document
        ld_feats = []
        for key, val in self.text_dict.items():
            text = val['text_basic'] 
            #text = " ".join(text)            
            text = " ".join(str(t) for t in text)
            feats_dict = extract_lexical_diversity_feats(text)
            feats_dict['id'] = key
            ld_feats.append(feats_dict)
        feats_df = pd.DataFrame(ld_feats)
        feats_df = feats_df.set_index('id')
        feats_df = feats_df.reset_index()
        feats_df.to_csv(os.path.join(output_dir, "lexical_diversity_features.csv"), index=False)

    def extract_liwc_features(self, output_dir):
        """
        Extract LIWC features for each entry in self.text_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        # split into sentences based on punctuation
        # then remove all punctuation except apostrophes & make lower case
        liwc_feats = []
        for key, val in self.text_dict.items():
            #text = " ".join(val['text'])
            text = " ".join(str(t) for t in val['text'])
            # split into sentences
            sentences = split_into_sentences(text)
            # remove punctuation & make lower case
            sentences = [remove_punctuation_except_apostrophes(sent).lower() for sent in sentences]
            feats_dict = extract_liwc_feats(sentences)
            feats_dict['id'] = key
            liwc_feats.append(feats_dict)
        feats_df = pd.DataFrame(liwc_feats)
        feats_df = feats_df.set_index('id')
        feats_df = feats_df.reset_index()
        feats_df.to_csv(os.path.join(output_dir, "liwc_features.csv"), index=False)

    def extract_parts_of_speech_features(self, output_dir):
        """
        Extract parts of speech (POS) features for each entry in self.text_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        # split into sentences based on punctuation
        # then remove all punctuation except apostrophes & make lower case
        pos_feats = []
        for key, val in self.text_dict.items():
            #text = " ".join(val['text'])
            text = " ".join(str(t) for t in val['text'])
            # split into sentences
            sentences = split_into_sentences(text)
            # remove punctuation & make lower case
            sentences = [remove_punctuation_except_apostrophes(sent).lower() for sent in sentences]
            #print(sentences) 
            feats_dict = extract_pos_features(sentences)
            feats_dict['id'] = key
            pos_feats.append(feats_dict)
        feats_df = pd.DataFrame(pos_feats)
        feats_df = feats_df.set_index('id')
        feats_df = feats_df.reset_index()
        feats_df.to_csv(os.path.join(output_dir, "pos_features.csv"), index=False)

    def extract_verbosity_features(self, output_dir):
        """
        Extract verbosity features for each entry in self.text_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        # split into sentences based on punctuation
        # then remove all punctuation except apostrophes & make lower case
        pos_feats = []
        for key, val in self.text_dict.items():
            #text = " ".join(val['text'])
            text = " ".join(str(t) for t in val['text'])
            # split into sentences
            sentences = split_into_sentences(text)
            # remove punctuation & make lower case
            sentences = [remove_punctuation_except_apostrophes(sent).lower() for sent in sentences]
            feats_dict = get_verbosity_stats(sentences)
            feats_dict['id'] = key
            pos_feats.append(feats_dict)
        feats_df = pd.DataFrame(pos_feats)
        feats_df = feats_df.set_index('id')
        feats_df = feats_df.reset_index()
        feats_df.to_csv(os.path.join(output_dir, "verbosity_features.csv"), index=False)


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
    parser.add_argument('--feature_list', type=str, nargs='+',
                        default=['graph', 'lexical_diversity', 'liwc', 'pos', 'verbosity'],
                        help="(Optional) Names of text features sets to extract. "
                             "Defaults to extracting all feature sets.")
    parser.add_argument('--output_dir', type=str, help="Path to directory to output feature files to.")
    parser.add_argument('--level', type=str, default=None, help='Data level (i.e. call, day)')
    parser.add_argument('--metadata_path', type=str, default=None, help="Path to segment metadata file (subject, call, etc.).") 
    parser.add_argument('--call_type', type=str, default='all', help='specifies type of call (assessment, personal, or all)')
    return parser.parse_args()


def main():
    args = _parse_args()
    recognition_result_files = _read_file_by_lines(args.ms_asr_output_files)
    text_feat_extractor = MicrosoftTextFeatureExtractor(recognition_result_files, args)
    text_feat_extractor.extract_features(set(args.feature_list), args.output_dir)


if __name__ == '__main__':
    main()