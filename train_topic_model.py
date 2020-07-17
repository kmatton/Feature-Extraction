import argparse, os
import itertools
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# spacy for lemmatization
# Plotting tools
import matplotlib.pyplot as plt

# import helper scripts
from preprocess_text import *
from process_text import *

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


"""
Program to train LDA topic model adapted from: https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
Adaptations to consider: subject specific topic models, also topic models trained specifically for leave one subject
out testing --> i.e. even though it's unsupervised, does it make sense to train topic model on all data?
"""


def collect_data(transcript_dir, min_word_count):
    """
    Reads all transcript files, cleans text, and stores files as a list of lists of words.
    :param transcript_dir: directory containing call subdirectories that hold transcripts.
    :param min_word_count: minimum transcript word count to be used in training model. Note that if at least one of a
    given call's hypothesis transcripts meets this word count criteria, all transcripts from the call are included.
    :return: sentences: list of all sentences (where each sentence each is a list of the words in a single speech
    segment) across all transcripts.
    call_sentence_counts: list of sentence count for each call_id, where sentence counts are in order that
    call_ids were processed (same order as call_ids).
    call_ids: list of all call_ids processed, in order that they were processed.
    """
    # collect all sentences across all transcripts
    sentences = []
    # store list of sentence counts for each call_id, so sentences can later be grouped by call_id
    call_sentence_counts = []
    call_ids = sorted([call_id for call_id in os.listdir(transcript_dir)])
    for call_id in call_ids:
        include = False
        call_sentences = []
        for transcript_file in os.listdir(os.path.join(transcript_dir, call_id)):
            transcript = open(transcript_file)
            # Expect first term in each line of file to be segment ID and don't include in list of words
            segments = [line.strip().split(" ")[1:] for line in transcript]
            segments = remove_non_verbal_exp(segments)
            # check transcript word count post cleaning
            if len(list(itertools.chain.from_iterable(segments))) >= min_word_count:
                include = True
            # clean text: remove stopwords and lemmatize
            segments = remove_stopwords(segments)
            segments = lemmatize(segments)
            call_sentences.extend(segments)
        if include:
            sentences.extend(call_sentences)
            call_sentence_counts.append(len(call_sentences))
    return sentences, call_sentence_counts, call_ids


def build_corpus(args):
    """
    Reads in call trasncripts, cleans/pre-processes text, and stores data in correct format for topic model training.
    :param args: argparse object that stores command line options as member variables
    :return: dct: Gensim copora dictionary object mapping word ids to word strings
    corpus: list of documents where each document is a list of the term frequencies of its words
    docs: list of documents where each document is list of its words
    """
    sentences, call_sentence_counts, call_ids = collect_data(args.transcript_dir, args.min_word_count)

    # determine number of hypotheses per call
    call_id = call_ids[0]
    transcripts = os.listdir(os.path.join(args.transcript_dir, call_id))
    num_hypotheses = len(transcripts)

    # create bigram and trigram models
    # note: executing this step on cleaned text (stopwords removed, lemmatized), but could do on original text instead
    bigram_model, trigram_model = init_bigram_trigram_models(sentences, num_hypotheses=num_hypotheses)

    # form bigrams
    sentences_w_bigrams = extract_bigrams(bigram_model, sentences)

    # group all transcripts for each call_id into a single document
    docs = []
    for idx in range(len(call_ids)):
        start_idx = sum(call_sentence_counts[:idx])
        end_idx = start_idx + call_sentence_counts[idx]
        doc = [word for sentence in sentences[start_idx:end_idx] for word in sentence]
        docs.append(doc)

    # create dictionary (word id -> word)
    dct = corpora.Dictionary(docs)

    # filter out super high and low frequency words
    dct.filter_extremes(no_below=5, no_above=.6)

    # represent each document as list of the term frequencies of its words
    corpus = [dct.doc2bow(doc) for doc in docs]

    return dct, corpus, docs


def train_topic_model(dct, corpus, docs, num_topics, verbose=True):
    """
    Trains <n_topics>-topic LDA Mallet topic model from input corpus.
    :param dct: Gensim copora dictionary object mapping word ids to word strings
    :param corpus: list of documents where each document is a list of the term frequencies of its words
    :param docs: list of documents where each document is list of its words
    :param num_topics: number topics to build model for
    :param verbose: if True, print topic + coherence information for topic model once trained
    :return: lda_mallet: trained LDA topic model (Mallet implementation: http://mallet.cs.umass.edu/index.php)
    coherence: model coherence score
    """
    mallet_path = 'C:/Users/katie/Documents/Research/feature_extraction/models/mallet-2.0.8/bin/mallet'
    lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dct)

    # compute model coherence score
    # compute coherence score
    coherence_model = CoherenceModel(model=lda_mallet, texts=docs, dictionary=dct, coherence='c_v')
    coherence = coherence_model.get_coherence()

    # print topic model information
    if verbose:
        pprint(lda_mallet.show_topics(formatted=False))
        print('\nCoherence Score: ', coherence)
    return lda_mallet, coherence


def sweep_num_topics(dct, corpus, docs, limit, start=2, step=5, display=True, verbose=True):
    """
    Compute coherence score for models trained with different numbers of topics.
    :param dct: Gensim copora dictionary object mapping word ids to word strings
    :param corpus: list of documents where each document is a list of the term frequencies of its words
    :param docs: list of documents where each document is list of its words
    :param limit: max number of topics
    :param start: min number of topics
    :param step: step size for num topic sweep
    :param display: if True, plot num_topics vs coherence
    :param verbose: if True, print coherence for each number of topics
    :return: model_list: list of trained LDA topic models
    coherence_values: coherence values corresponding to the LDA model with respective number of topics
    """
    model_list = []
    coherence_values = []
    for num_topics in range(start, limit, step):
        model, coherence = train_topic_model(dct, corpus, docs, num_topics, verbose=False)
        model_list.append(model)
        coherence_values.append(coherence)

    x = range(start, limit, step)

    if display:
        plt.plot(x, coherence_values)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend("coherence_values", loc="best")
        plt.show()
        plt.clf()

    if verbose:
        for nt, coherence in zip(x, coherence_values):
            print("Num Topics =", x, " has Coherence Value of", round(coherence, 4))
    return model_list, coherence_values


def main():
    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transcript_dir', type=str, help='Directory holding transcripts for training model.'
                                                                 'Needs to contain subdirectories for each call that '
                                                                 'contain the call transcripts (can be multiple '
                                                                 'hypotheses per call)')
    parser.add_argument('-w', '--min_word_count', type=int, default=50, help='Minimum call word count to be included in'
                                                                             'training of topic model')
    args = parser.parse_args()

    # build training corpus
    dct, corpus, docs = build_corpus(args)

    # train topic models with different numbers of topics and get coherence values for each
    model_list, coherence_vals = sweep_num_topics(dct, corpus, docs, 40)


if __name__ == '__main__':
    main()
