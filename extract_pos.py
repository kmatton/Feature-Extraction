import nltk
import truecase
import preprocess_text as pt


"""
Contains functions to compute part of speech (POS) features. Uses PennTreebank POS types
are used ( https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).
"""


POS_KEY_LIST = ['ADJ', 'VERB', 'NOUN', 'ADV', 'DET', 'INT', 'PREP', 'CC', 'PNOUN', 'PSNOUN']


def update_feature_vals(tag, feats_dict):
    """
    Updates feature values (POS counts) based on input POS tag.
    :param tag: Penn TreeBank tag
    :param feats_dict: dictionary to store feature values for the transcript
    """
    if tag.startswith('J'):
        feats_dict['ADJ'] += 1
    elif tag.startswith('V'):
        feats_dict['VERB'] += 1
    elif tag.startswith('N'):
        feats_dict['NOUN'] += 1
    elif tag.startswith('R'):
        feats_dict['ADV'] += 1
    elif tag.startswith('D'):
        feats_dict['DET'] += 1
    elif tag.startswith('U'):
        feats_dict['INT'] += 1
    elif tag.startswith('I') or tag.startswith('T'):
        feats_dict['PREP'] += 1
    elif tag == 'CC':
        feats_dict['CC'] += 1
    elif tag == 'PRP':
        feats_dict['NOUN'] += 1
        feats_dict['PNOUN'] += 1
    elif tag == 'PRP$':
        feats_dict['PSNOUN'] += 1
        feats_dict['NOUN'] += 1
    elif tag.startswith('W'):
        if tag[1] == 'D':
            feats_dict['DET'] += 1
        elif tag[1] == 'R':
            feats_dict['ADV'] += 1
        elif tag.endswith('P'):
            feats_dict['PNOUN'] += 1
            feats_dict['NOUN'] += 1
        else:
            feats_dict['PSNOUN'] += 1


def get_pos_ratios(pos_dict):
    pos_dict['adj_ratio'] = float(pos_dict['ADJ']) / float(pos_dict['VERB']) if float(pos_dict['VERB']) else float('nan')
    pos_dict['v_ratio'] = float(pos_dict['NOUN']) / float(pos_dict['VERB']) if float(pos_dict['VERB']) else float('nan')
    if float(pos_dict['VERB']) + float(pos_dict['NOUN']):
        pos_dict['n_ratio'] = float(pos_dict['NOUN']) / (float(pos_dict['VERB']) + float(pos_dict['NOUN']))
    else:
        pos_dict['n_ratio'] = float('nan')
    pos_dict['pn_ratio'] = float(pos_dict['PNOUN']) / float(pos_dict['NOUN']) if float(pos_dict['NOUN']) else float('nan')
    pos_dict['sc_ratio'] = float(pos_dict['PREP']) / float(pos_dict['CC']) if float(pos_dict['CC']) else float('nan')


def extract_pos_features(transcript):
    """
    :param transcript: List of transcript segments. Each segment is represented as a tuple of the form (start time,
    stop time, text).
    :return: feats_dict: Dictionary mapping feature name to value for transcript
    """
    # first pre-process text
    # remove can ignore timing info because not relevant to computing POS measures
    transcript = [seg[2] for seg in transcript]
    # split into words
    segments = [seg.strip().split(" ") for seg in transcript]
    # remove non-verbal expressions
    segments = pt.remove_non_verbal_exp(segments)

    # initialize feature dictionary with POS types
    feats_dict = dict((key,0) for key in POS_KEY_LIST)
    num_words = 0
    # add POS count features
    for segment in segments:
        num_words += len(segment)
        # first transform to true case (i.e. capitalize if supposed to be), so that POS tagger works better
        segment_str = " ".join(segment)
        truecase_str = truecase.get_true_case(segment_str)
        pos_seg = nltk.pos_tag(truecase_str.split(" "))
        for word, tag in pos_seg:
            update_feature_vals(tag, feats_dict)
    get_pos_ratios(feats_dict)
    # convert counts to proportions
    for key in POS_KEY_LIST:
        count = float(feats_dict[key])
        feats_dict[key] = count / float(num_words)
    return feats_dict

