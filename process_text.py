import gensim
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


"""
Contains functions for basic text processing operations (e.g. lemmatizing, extracting bigrams)
"""


LEMMATIZER = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    """
    :param treebank_tag: Penn Treebank POS tag
    :return: corresponding Word Net tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize(segments):
    """
    Converts speech segments from list of words to list of their associated lemmas.
    :param segments: a list of speech segments, where each segment is a list of the words that make up that segment
    :return: segments: speech segment list with words converted to their associated lemma
    """
    for segment in segments:
        pos_list = nltk.pos_tag(segment)
        for i in range(len(segment)):
            pos = get_wordnet_pos(pos_list[i][1])
            segment[i] = LEMMATIZER.lemmatize(segment[i], pos)
    return segments


def init_bigram_trigram_models(sentences, num_hypotheses=1):
    """
    Creates bigram and trigram models using the sentences provided. The Genism Phrases model builds bigram/trigram
    models based on words that frequently occur together in the provided data.
    See https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases.
    Parameters set based on: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/.
    Note default threshold is 10 and higher threshold means fewer phrases.
    :param sentences: list of all sentences (where each sentence each is a list of the words in a single speech
    segment) across all transcripts.
    :return: bigram_model: Gensim Phraser model for detecting bigrams in text.
    trigram_model: Gensim Phraser model for detecting trigrams in text.
    """
    # need to set threshold higher if multiple hypotheses per transcript because a given bigram/trigram because
    # num_hypotheses occurrences of a given bigram/trigram represents it occurring in one document with probability one
    bigram = gensim.models.Phrases(sentences, threshold=100*num_hypotheses)
    trigram = gensim.models.Phrases(bigram[sentences], threshold=100*num_hypotheses)
    # convert to faster implementation of Phrases (reduced functionality, but is all we need since we aren't going to
    # add more words to the models)
    bigram_model = gensim.models.phrases.Phraser(bigram)
    trigram_model = gensim.models.phrases.Phraser(trigram)
    return bigram_model, trigram_model


def extract_bigrams(bigram_model, texts):
    """
    Converts each item in texts from list of sentences made up on only unigrams, to list of sentences that also contains
    bigrams (e.g. "worda_wordb").
    :param bigram_model: trained Gensim Phraser model for detecting bigrams.
    :param texts: list of documents, where each document is list of sentences, and each sentence is list of unigrams.
    :return: object in same form as text that contains unigrams and bigrams.
    """
    return [bigram_model[doc] for doc in texts]


def extract_trigrams(bigram_model, trigram_model, texts):
    """
    Converts each item in texts from list of sentences made up on only unigrams, to list of sentences that also contains
    bigrams (e.g. "worda_wordb") and trigrams.
    :param bigram_model: trained Gensim Phraser model for detecting bigrams.
    :param trigram_model: trained Gensim Phraser model for detecting trigrams.
    :param texts: list of documents, where each document is list of sentences, and each sentence is list of unigrams.
    :return: object in same form as text that contains unigrams, bigrams, and trigrams.
    """
    return [trigram_model[bigram_model[doc]] for doc in texts]
