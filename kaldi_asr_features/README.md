This directory contains scripts for extracting features that are derived from the outputs of a Kaldi ASR model.

The included scripts are as follows:
* `extract_asr_conf_feats.py` computes features from files that contain the confidence scores produced by a 
   Kaldi ASR model.
* `extract_timing_feats.py` computes features from files that contain audio + text aligned timing
   annotations, as produced by Kaldi.
* `extract_word_phone_timing.py` is a helper script for `extract_timing_feats.py`.
* `extract_non_verbal.py` computes counts of non-verbal expressions (normalized by total word count) present in 
   the transcripts produced by a Kaldi ASR model. This is specifically for models trained on the 
   [Fisher English corpus](https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2004-fisher-corpus.pdf).

To generate the input files required by the above feature extraction scripts see 
[this repository](https://github.com/kmatton/ASR-Helper). The relevant files are located in `asr-models-support/Kaldi`.

Note that some of these scripts are specifically designed to work with the PRIORI dataset, but they can easily be adapted to work with other datasets.
