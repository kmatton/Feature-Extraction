import os 

"""
Base class for specifying general project configuration options.
"""

class Config:
    def __init__(self):
        self.project_root = '/nfs/turbo/McInnisLab/hnorthru/code/kmatton/Feature-Extraction'
        self.LIWC_2007_PATH = os.path.join(self.project_root, 'text_features/LIWC2007_English_adapted.dic')