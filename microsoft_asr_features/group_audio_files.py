""" group_audio_files

Use metadata to identify which audio files to group together for feature extraction 

(i.e. group calls by day) 

"""

import os 
import sys
import argparse
import pandas as pd 
from IPython import embed


def add_feature_id(df, level, metadata_path, call_type):
    """
    Add feature_id column to dataframe for Microsoft Azure to group audio files by attribute  

    :param df: dataframe with speech to text results by audio file id 
    :return df: dataframe with 'feature_id' to group audio files for feature extraction
    """

    #get segment metadata 
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['call_datetime'] = pd.to_datetime(metadata_df['call_datetime'])
    metadata_df['call_date'] = metadata_df['call_datetime'].dt.date
    metadata_df['day_id'] = metadata_df['subject_id'].apply(str) + '_' + metadata_df['call_date'].apply(str) 

    #filter call type (personal, assessment, all) 
    if metadata_df['is_assessment'].dtypes == 'bool':
        metadata_df.loc[metadata_df['is_assessment'] == True, 'is_assessment'] = 't'
        metadata_df.loc[metadata_df['is_assessment'] == False, 'is_assessment'] = 'f'
    if call_type == 'personal':
        metadata_df = metadata_df.loc[metadata_df['is_assessment'] == 'f', :]
    elif call_type == 'assessment':
        metadata_df = metadata_df.loc[metadata_df['is_assessment'] == 't', :]
    elif call_type != 'all':
        print('Invalid call_type: ' + str(args.call_type))
        return 

    #get desired call_ids
    call_ids = sorted(metadata_df['call_id'].unique()) 
    df = df.loc[df['audio_file_id'].isin(set(call_ids)), :]
    
    #aggregation level 
    if level == 'day':  
        #map calls (audio_file_id) to day_ids 
        call_to_day_dict = dict(zip(metadata_df['call_id'].values, metadata_df['day_id'].values))
        df.loc[:,'feature_id'] = df['audio_file_id'].map(call_to_day_dict).values 
    return df 
