## Prerequisites
* [gensim](https://radimrehurek.com/gensim/)
* [NLTK](https://www.nltk.org/)
* [NetworkX](https://networkx.github.io/documentation/stable/index.html)
* [NumPy](https://numpy.org/)
* [Python LIWC library](https://pypi.org/project/liwc/) (specifically for LIWC 2007 category features)
* [LIWC 2007 dictionary .dic file](http://liwc.wpengine.com/) (specifically for LIWC 2007 category features)
* [TrueCase](https://pypi.org/project/truecase/)
* [textstat](https://pypi.org/project/textstat/)

## Feature extraction modules
`extract_graph.py` extracts speech graph features, which are designed to capture thought flow and disordered thinking
 patterns. The features are based on the papers [Speech graphs provide a quantitative measure of thought disorder in 
 psychosis](https://pubmed.ncbi.nlm.nih.gov/22506057/) by Mota et al. and [Automated Speech Analysis for Psychosis 
 Evaluation](https://link.springer.com/chapter/10.1007/978-3-319-45174-9_4) by Carrillo et al.
 
`extract_lexical_diversity.py` computes 
[MATTR (moving average type-token ratio)](https://www.tandfonline.com/doi/abs/10.1080/09296171003643098) 
and [Honore's Statistic](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5961820/).

`extract_verbosity_stats.py` computes features related to the verbosity of speech (e.g. word count) and the complexity 
 of speech (e.g. fraction of words with 6+ characters).
 
`extract_liwc_2007.py` computes the percentage of words that are in each category in the 
[LIWC 2007 dictionary](https://www.liwc.net/LIWC2007LanguageManual.pdf). The LIWC dictionary is proprietary, so it's not
included here, but you can obtain a copy [here](http://liwc.wpengine.com/). I found that in order to get the Python LIWC 
Library to work with the LIWC 2007 dictionary, I had to adapt the dictionary file slightly. Specifically, I removed the 
`kind <of>` entry from the dictionary. You may need to do that before using this module. Also, make sure to put the
path to your copy of the LIWC dictionary in the config.txt file.

`extract_pos.py` computes parts-of-speech (POS) features. These include the proportion of instances of POS, 
as well the ratio of occurrences of different POS.

## To-dos
* Figure out the best way to incorporate pre-processing into this feature extraction pipeline.