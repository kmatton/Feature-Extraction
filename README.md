This repository contains scripts for extracting text and speech features. Features are primarily designed to be
extracted from the outputs of automatic speech recognition models (e.g. text, timing, confidence scores).

The subdirectories are as follows:
*  `text_features` contains scripts for extracting text features. They can be extracted from transcribed speech or written text.
* `microsoft_asr_features` contains scripts for extracting features specific to the output of [Microsoft's speech-to-text models](https://azure.microsoft.com/en-us/services/cognitive-services/speech-to-text/). This includes a script for extracting the `text-features` mentioned above form the output of Microsoft's speech-to-text model as well as scripts for extracting features related to word-level timing and model confidence scores. 
* `kaldi_asr_features` contains scripts for extracting feature specific to the output of Kaldi ASR models, including word and phone-level timing features, ASR confidence scores, and non-verbal expression counts.
* `archived` contains outdated/in-progress scripts that are not yet documented.

Note: the input files expected by the scripts within the `microsoft_asr_features` and `kaldi_asr_features` directories can be produced using the code in [this repository](https://github.com/kmatton/ASR-Helper) (see the `asr-models-support` directory).