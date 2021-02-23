#!/usr/bin/env python
# coding: utf-8

""" Functions
- utils: data processing
- scraping : extracting data from websites
- text mining / NLP: text cleaning, tokenisation, lemmatisation, stopwords, LDA model, topic modelling

"""


import spacy
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import unidecode
from gensim.utils import simple_preprocess
import gensim
from sklearn import neighbors
import numpy as np
from gensim.models import CoherenceModel
# word embeddings
nlp = spacy.load('fr_core_news_md')  # python -m spacy download fr_core_news_md



# =============================================================================
#                                  divers
# =============================================================================


def flat_list(l: list) -> list:

    return [x for sub in l for x in sub]


# =============================================================================
#                                  scraping
# =============================================================================


def create_soup(url: str, enc: str):
    """ 
        In: url
        Out: source code  
    """

    req = requests.get(url)
    data = req.text
    soup = BeautifulSoup(data, from_encoding=enc)

    return soup


# =============================================================================
#                              text mining / NLP
# =============================================================================


def basic_cleaner(s: str, lower: bool, punct: bool) -> str:
    
    for el in ["n'", "l'", "qu'", "t'", "s'", "d'", "j'", "m'", "c'",
               "N'", "L'", "Qu'", "T'", "S'", "D'", "J'", "M'", "C'",
               '(...)', '[...]', '\x92', 'est-ce', 'Est-ce', 'comment', 'Comment']:
        s = s.replace(el, '')

    s = s.replace('\x9c', 'oe').replace(' - ', ' ').replace('\x92', ' ')
    s = unidecode.unidecode(s)
    s = re.sub("[^A-Za-z0-9.!?,;' ]+", '',
               s) if punct else re.sub("[^A-Za-z0-9 ]+", '', s)
    s = s.replace("'", ' ').replace('  ', ' ')
    s = ' '.join(s.split())
    s = s if not lower else s.lower()

    return s


def embed(tokens: list, nlp: spacy.lang, remove_stops: bool, lim_len: int):

    vectors = [x.vector for x in [nlp.vocab[token] for token in tokens] if len(x.text) > lim_len and x.has_vector] if not remove_stops else [
        x.vector for x in [nlp.vocab[token] for token in tokens] if len(x.text) > lim_len and x.has_vector and not x.is_stop]
    vectors = np.array(vectors)

    centroid = vectors.mean(axis=0) if len(
        vectors) > 0 else np.zeros(nlp.meta['vectors']['width'])

    return centroid


def predict_theme(tokens: list, theme_vectors: np.array, themes_list: list,
                  n_neighbors: int, nlp: spacy.lang,
                  remove_stops: bool, lim_len: int):

    centroid = embed(tokens, nlp, remove_stops, lim_len)

    neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(theme_vectors)
    closest_theme = neigh.kneighbors([centroid], return_distance=False)
    theme = themes_list[int(closest_theme)]

    return theme


def tokenize(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts: list, stop_words: list) -> list:
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def clean_lemmatize(data: list, bigram_mod, stop_words: list):
    data_tokens_nostops = remove_stopwords(data, stop_words)
    data_tokens_bigrams = make_bigrams(data_tokens_nostops, bigram_mod)
    data_lemmas_bigrams = lemmatization(data_tokens_bigrams, allowed_postags=[
                                        'NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_tokens_bigrams, data_lemmas_bigrams


def build_lda_model(nb: int, corpus: list, id2word: gensim.corpora.dictionary.Dictionary, data_lemma: list) -> (gensim.models.ldamodel.LdaModel, gensim.interfaces.TransformedCorpus, np.float64, np.float64, list):

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=nb,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    doc_lda = lda_model[corpus]
    perplexity_lda = lda_model.log_perplexity(corpus)
    coherence_model_lda = CoherenceModel(
        model=lda_model, texts=data_lemma, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    model_topics = lda_model.show_topics(formatted=False)

    return lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics


# conservation du meilleur modèle pour la suite
def lda_traitements(nb: int, corpus: list, id2word: gensim.corpora.dictionary.Dictionary, data_lemma: list) -> (gensim.models.ldamodel.LdaModel, gensim.interfaces.TransformedCorpus, np.float64, np.float64, list):

    lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        nb, corpus, id2word, data_lemma)

    return lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics


def format_topics_sentences(ldamodel: gensim.models.ldamodel.LdaModel, corpus: list, data) -> pd.core.frame.DataFrame:

    sent_topics_df = pd.DataFrame()

    # dans chaque document, trouver le thème dominant
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # trouver le thème dominant, la contribution du texte au modèle et les mots-clés pour chaque doc
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # si thème dominant
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series(
                    [int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic',
                              'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(data)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)

    return sent_topics_df


# mise en forme des résultats dans un df
def topics_df(lda_model: gensim.models.ldamodel.LdaModel, corpus: list, data: list) -> pd.core.frame.DataFrame:

    df_topic_sents_keywords = format_topics_sentences(
        lda_model, corpus, data)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        'Document_Index', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return df_dominant_topic


def most_common(c: collections.Counter, type_: str, nb: int) -> pd.core.frame.DataFrame:
    if type_ == 'words':
        df = pd.DataFrame(columns=['words', 'nb_occur'])

        for word, nb_occur in c.most_common(nb):
            L = df.shape[0]
            df.loc[L+1] = [word, nb_occur]

    elif type_ == 'bigrams':
        df = pd.DataFrame(columns=['bigrams', 'nb_occur'])

        for word, nb_occur in c.most_common(nb):
            if '_' in word:
                L = df.shape[0]
                df.loc[L+1] = [word, nb_occur]

    return df


def plot_most_common(df: pd.core.frame.DataFrame, type_: str):
    df.plot(x=type_, y='nb_occur', kind='bar', figsize=(14, 9))


def wordcloud(df: pd.core.frame.DataFrame, type_: str):
    dict_most_common = {}
    for k in range(1, len(df)):
        dict_most_common[df.loc[k][type_]] = int(df.loc[k]['nb_occur'])

    most_common_cloud = WordCloud(background_color='white',
                                  width=1600,
                                  height=900).generate_from_frequencies(dict_most_common)
    plt.figure(figsize=(14, 9), dpi=80)
    plt.imshow(most_common_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    
def sentiment(df: pd.core.frame.DataFrame, var_: str) -> pd.core.frame.DataFrame:

    list_content = df[var_].tolist()

    polarity = []
    subjectivity = []
    polarity_class = [0] * len(list_content)
    subjectivity_class = [0] * len(list_content)

    for i in range(len(list_content)):
        if list_content[i] == 'empty':
            polarity.append(0)
            subjectivity.append(0)

        else:
            content = list_content[i]
            blob = TextBlob(content, pos_tagger=PatternTagger(),
                            analyzer=PatternAnalyzer())
            polarity.append(blob.sentiment[0])
            subjectivity.append(blob.sentiment[1])

    df['polarity'] = polarity
    df['subjectivity'] = subjectivity

    for k in range(len(df)):
        p = df.loc[k]['polarity']

        if p < - 0.05:
            polarity_class[k] = 'negative'

        elif p > 0.125:
            polarity_class[k] = 'positive'

        else:
            polarity_class[k] = 'neutral'

        s = df.loc[k]['subjectivity']

        if s < 0.2:
            subjectivity_class[k] = 'objective'

        elif s > 0.5:
            subjectivity_class[k] = 'subjective'

        else:
            subjectivity_class[k] = 'neutral'

    df['polarity_class'] = polarity_class
    df['subjectivity_class'] = subjectivity_class

    df['polarity_subjectivity'] = [f'{x}_{y}' for x, y in zip(
        df.polarity_class, df.subjectivity_class)]

    return(df)
    
    