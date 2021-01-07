''' get_call_act_features 

Call Act features are a sub-set of timing features 
Get call act features from timing_features.csv file 

''' 

import os 
import pandas as pd 
from IPython import embed 

#SET UP 
base_feat_dir = '/nfs/turbo/chai-health/hnorthru/extracted_features/priori_v1'
timing_path = os.path.join(base_feat_dir, 'microsoft', 'day', 'personal', 'day_personal_timing.csv')
save_path = '.'
save_name = 'day_personal_call_act.csv' 

#metadata_cols = ['call_id']
metadata_cols = ['subject_id', 'date']

cols = [ 'segment_count', 'segments_max', 'segments_min', 'segments_mean', 'segments_med', 'segments_std', \
        'spk_duration', 'spk_ratio', 'total_duration', \
        'word_count', 'words_max', 'words_min', 'words_mean', 'words_med', 'words_std']

# Load Data 
timing_df = pd.read_csv(timing_path) 

#get desired subset of columns 
call_act_df = timing_df.loc[:, metadata_cols+cols]

# Save call_act features 
call_act_df.to_csv(os.path.join(save_path, save_name), index=False) 


'''
Call_act Features 
******************
Segment_count
Segments (max, mean, min, med, std)
Spk_duration
Spk_ratio
Total_duration
Word_count
Words (max, mean, med, min, std)


Timing Features
****************
Phone_count
Phones_max
Phones_med
Phones_min
Phones_std
Pps (count, mean, med, min, std, max)
Segment_count
Segments (max, mean, med, min, std)
Segs_per_min
Short_utt_count
Short_utts_per_min,
Sil_count
Sil_duration
Sil_ratio
silences (max, mean, med, min, std)
Spk_duration
Spk_ratio
Sps
Total_duration
Word_count
Words (max, mean, med, min, std)
Wps
Wps (max, mean, med, min, std)
'''















