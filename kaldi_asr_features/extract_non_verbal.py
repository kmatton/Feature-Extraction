"""
Extract counts of instances of non-verbal expressions within transcript, as output by Kaldi ASR model
trained on the Fisher English dataset
(https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/lrec2004-fisher-corpus.pdf)
"""


def extract_non_verbal_feats(segments):
    """
    Computes ratio of non-verbal expressions and unknown words to total items in transcript.
    :param segments: List of transcript segments. Each segment is represented as string.
    :return: feature_dict: Dictionary mapping feature name to value for transcript
    """
    # split into words
    segments = [seg.strip().split(" ") for seg in segments]

    feature_dict = {}
    tokens = [token for segment in segments for token in segment]
    total_count = len(tokens)
    feature_dict['laughter'] = tokens.count('[laughter]') / total_count if total_count else float('nan')
    feature_dict['noise'] = tokens.count('[noise]') / total_count if total_count else float('nan')
    feature_dict['unk'] = tokens.count('<unk>') / total_count if total_count else float('nan')
    return feature_dict
