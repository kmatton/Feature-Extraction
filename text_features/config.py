import configparser

"""
Read in arguments from text config file.
"""
cfg = configparser.ConfigParser()
cfg.read("config.txt")
LIWC_2007_PATH = cfg["DATA_LOCAtIONS"]["LIWC_2007_PATH"]
