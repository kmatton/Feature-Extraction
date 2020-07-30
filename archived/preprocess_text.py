
"""
Contains functions to pre-process text before feature extraction, collecting vocab, etc.
"""


def get_stop_words():
    with open('stopwords.txt') as stopfile:
        return [line.strip() for line in stopfile]


STOPWORDS = get_stop_words()


def remove_stopwords(segments):
    segments = [[word for word in segment if word not in STOPWORDS] for segment in segments]
    return segments


def remove_non_verbal_exp(segments):
    """
    Remove instances of [noise], [laughter], and <unk>, which are not relevant for many of the language features.
    :param segments: List of transcript segments. Each segment is represented as a list of words.
    :return: segments: cleaned segments
    """
    remove_tokens = {'[noise]', '[laughter]', '<unk>'}
    cleaned_segments = []
    for segment in segments:
        segment = [word for word in segment if word not in remove_tokens]
        if segment:
            cleaned_segments.append(segment)
    return cleaned_segments
