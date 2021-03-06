import os
import sys 
import argparse
import pandas as pd
import numpy as np
import math
from IPython import embed 
from group_audio_files import add_feature_id 

sys.path.append(os.getcwd()) #for slurm jobs  
sys.path.append('../timing_features') #for slurm jobs  
from timing_features.extract_word_phone_timing import get_feats

"""
Script for extracting timing features from Microsoft recognition_results.csv 
"""

class MicrosoftTimingFeatureExtractor:
    def __init__(self, recognition_result_files, args):
        """
        :param recognition_result_files: list of paths to files containing recognition results from Microsoft
                                         speech-to-text model
        """
        self.recognition_result_files = recognition_result_files
        self.level = args.level 
        self.call_type = args.call_type
        self.metadata_path = args.metadata_path
        self.time_dict = self._collect_timing()
        #set up duration_df 
        self.duration_df = pd.read_csv(args.duration_file_path)
        if 'duration_ms' not in set(self.duration_df.columns):
            self.duration_df['duration_ms'] = self.duration_df['duration']
        self.duration_df['duration_sec'] = self.duration_df['duration_ms'] * 10**-3
        if self.level == 'day':
            self.duration_df['id'] = self.duration_df['subject_id'].apply(str) + '_' + pd.to_datetime(self.duration_df['call_datetime']).dt.date.apply(str)
        else: # self.level == 'call':
            self.duration_df['id'] =  self.duration_df['call_id']
        

    def _collect_timing(self):
        """
        Collect all timing information for recognizer entries that have the same 'feature_id'
        (or 'audio_file_id' if 'feature_id' is not present)
        :return: time_dict: dict mapping ids ('feature_id' or 'audio_file_id') to word and segment timing info 
        """
        time_dict = dict()
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
            time_dict[idx] = {}
            #sort entries in dataframe            
            if len(combined_df.loc[idx].shape) == 1:
                sorted_entries = pd.DataFrame(columns=combined_df.columns)
                sorted_entries.loc[0] = combined_df.loc[idx, :]
            else:
                sorted_entries = combined_df.loc[idx].sort_values(sort_column)
            #compute initial timing dictionary of (seg durs, silence durs, word durs, wps)                  
            time_dict[idx]['timing'] = self.get_times(sorted_entries) 
        return time_dict

    def _parse_word_timing_str(self, seg_df):
        """
        Parse word_timing string(list(Dict)) and generate List(Dict) with a dictionary for each word (duration, offset, word)
        :param seg_df: single segment dataframe with word-level timing information. Each row (duration, offset, str(list(Dict)). 
        :return: seg_df: single segment dataframe with each row (duration, offset, list(Dict). 
        """
        word_timing_list = []
        #separate each string word dictionary 
        word_timing_str = seg_df['word_timing'].strip('][').split("},")
        
        #get information for each word dictionary 
        for ws in word_timing_str:
            word_dict = {}
            ws = ws.strip(" ")
            dl = ws.strip("}{").split(",")
            for d in dl:
                d2 = d.split(":")
                k = d2[0].strip(" ").strip("''")
                v = d2[1].strip(" ")
                if v.isnumeric(): #Duration, Offset are integers
                    v = int(v) 
                else: #Word is a string
                    v = v.strip("''")
                word_dict[k] = v
            word_timing_list.append(word_dict) 
        seg_df['word_timing'] = word_timing_list   
        return seg_df 

    def get_times(self, combined_df):
        """
        Collect overall timing information of segments, silences, words, and phones. Also collect words per second and
        phones per second rates for each segment.
        :param combed_df: dataFrame of word-level timing information. Each row includes duration, offset information 
        for each segment and word within each segment. 
        :return: seg_df: dictionary that stores timing information extracted from combed_df. Entries
        are lists of the durations of "segments", "silences", and "words", as well as lists of
        "wps" (words per second) for each segment.

        #Time Scales
        #***************** 
        #segment duration (seconds) 
        #word duration (ms)
        #silence duration (ms) 

        ## from microsoft speech_to_text.py 
            #--- Duration: The duration (in 100-nanosecond units) of the recognized speech in the audio stream.
            #-- Offset: The time (in 100-nanosecond units) at which the recognized speech begins in the audio stream.
        """
        times_dict = {"segments": [], "silences": [], "words": [], "wps": []}
        #get duration of segments (sec), silences (ms), and words (ms) and words-per-sec         
        for s in range(0, combined_df.shape[0]):
            seg = combined_df.iloc[s]
            seg = self._parse_word_timing_str(seg) 

            #Segment duration (sec) 
            seg_dur_sec = seg['duration'] *10**-7 #convert (100 ns units) to seconds 
            times_dict['segments'].append(seg_dur_sec)#seconds 

            #Word duration (ms) 
            for w in seg['word_timing']:
                #convert from (100 ns units) to ms 
                w_dur_ms = w['Duration']*10**-4
                times_dict["words"].append(w_dur_ms) 
             
            #Words per second (wps) 
            word_count = len(seg['word_timing'])
            times_dict["wps"].append(word_count/seg_dur_sec)

            # Duration of each silence (ms)         
            for i in range(0, len(seg['word_timing'])):
                if i == 0:# silence before first word 
                    sil_dur = seg['word_timing'][i]['Offset'] - seg['offset']
                    sil_dur_ms = sil_dur * 10**-4 #convert from (100 ns units) to ms 
                    times_dict["silences"].append(sil_dur_ms)

                else: #silences between words 
                    prev_word_end = seg['word_timing'][i-1]['Offset']+ seg['word_timing'][i-1]['Duration'] 
                    curr_word_start = seg['word_timing'][i]['Offset']
                    sil_dur = curr_word_start - prev_word_end 
                    sil_dur_ms = sil_dur * 10**-4 #convert from (100 ns units) to ms
                    times_dict["silences"].append(sil_dur_ms) 

                if i == len(seg['word_timing']) - 1: # silence after last word 
                    end_curr_word = seg['word_timing'][i]['Offset'] + seg['word_timing'][i]['Duration']
                    end_seg = seg['offset'] + seg['duration']
                    sil_dur = end_seg - end_curr_word 
                    sil_dur_ms = sil_dur * 10**-4 #convert from (100 ns units) to ms 
                    times_dict["silences"].append(sil_dur_ms) 
            #remove all zero length silences 
            times_dict['silences'] = [x for x in times_dict['silences'] if x != 0]
        return times_dict


    def extract_timing_feats(self, output_dir):
        """
        Extract timing features for each entry in self.times_dict and store in CSV file.
        :param output_dir: path to directory to output features to
        """
        #timing_feats = []
        for key, val in self.time_dict.items():
            times_dict = val['timing']
            total_dur_sec = self.duration_df.loc[self.duration_df['id'] == key, 'duration_sec'].sum()
            feats_dict = get_feats(times_dict, total_dur_sec)  
            feats_dict['id'] = key 
            #timing_feats.append(feats_dict)
            feats_df = pd.DataFrame(feats_dict, index=[0])
            feats_df = feats_df.set_index('id')
            feats_df = feats_df.reset_index()
            #save feature stats for index to file 
            output_path = os.path.join(output_dir, str(key) + '.csv')
            feats_df.to_csv(output_path, index=False) 


def chunks(l, n_chunks):
	n_in_chunk = math.ceil(float(len(l))/float(n_chunks))
	for i in range(0,len(l),n_in_chunk):
		yield l[i:i+n_in_chunk]

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
    parser.add_argument('-d', '--duration_file_path', type=str, help='Path to csv maps audio file to duration (ms)')
    parser.add_argument('--job_num', type=int, default=1) 
    parser.add_argument('--level', type=str, default=None, help='Data level (i.e. call, day)')
    parser.add_argument('--metadata_path', type=str, default=None, help="Path to segment metadata file (subject, call, etc.).") 
    parser.add_argument('--call_type', type=str, default='all', help='specifies type of call (assessment, personal, or all)')
    return parser.parse_args()

def main():
    args = _parse_args()
    recognition_result_files = _read_file_by_lines(args.ms_asr_output_files)
    #get subset 
    recognition_result_files = list(chunks(recognition_result_files,100))[int(args.job_num)-1]  
    print(len(recognition_result_files)) 
    print(recognition_result_files) 
    timing_feat_extractor = MicrosoftTimingFeatureExtractor(recognition_result_files, args)
    timing_feat_extractor.extract_timing_feats(args.output_dir)

if __name__ == '__main__':
    main()
