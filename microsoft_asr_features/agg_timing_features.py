import os 
import pandas as pd 
import argparse

from IPython import embed 

#Parse Arguments 
#**********************
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, help="Path to directory with individual output feature files")
parser.add_argument('--output_dir', type=str, help="Path to directory to output combined feature files to.")
parser.add_argument('--level', type=str, default='call')
parser.add_argument('--call_type', type=str, default='all')
args = parser.parse_args()

#Aggregate individual files in input_dir 
#***********************************
files = os.listdir(args.input_dir) 
if len(files) > 0:
        df_temp = pd.read_csv(os.path.join(args.input_dir, files[0]))
        df_full = pd.DataFrame(columns=df_temp.columns) 
        for f in files: 
            df = pd.read_csv(os.path.join(args.input_dir, f))
            df_full = df_full.append(df, sort=True)                 
        df_full = df_full.set_index('id') 

        #Add metadata based on level 
        if args.level == 'day': 
                df_full['day_id'] = df_full.index 
                sub_ids = df_full['day_id'].apply(lambda x: x.split('_')[0]).values 
                dates = df_full['day_id'].apply(lambda x: x.split('_')[1]).values
                df_full.insert(loc=0, column='subject_id', value=sub_ids)
                df_full.insert(loc=1, column='date', value=dates) 
                df_full = df_full.drop(['day_id'], axis=1) 
        elif args.level == 'call':
                df_full.insert(loc=0, column='call_id', value=df_full.index.values) 

        #Save aggregate file 
        df_full.to_csv(os.path.join(args.output_dir, args.level + '_' + args.call_type + '_timing.csv'), index=False) 
