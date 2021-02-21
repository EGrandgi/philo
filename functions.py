#!/usr/bin/env python
# coding: utf-8

""" Fonctions
- divers : traitement de listes
- scraping : récupération du code source d'une page web
- traitement automatique du langage : nettoyage de texte, tokenisation, lemmatisation, stopwords, modèle LDA, topic modelling

"""


import spacy
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv
import re
import unidecode
from gensim.utils import simple_preprocess
import gensim
import nltk
from sklearn import neighbors
import numpy as np
nltk.download('stopwords')
stop_words = stopwords.words('french')
nlp = spacy.load('fr_core_news_md')
from gensim.models import CoherenceModel



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
#                      traitement automatique du langage
# =============================================================================


def basic_cleaner(s: str, lower: bool, punct: bool) -> str:
    
    for el in ["n'", "l'", "qu'", "t'", "s'", "d'", "j'", "m'", "c'",
               "N'", "L'", "Qu'", "T'", "S'", "D'", "J'", "M'", "C'",
               '(...)', '[...]', '\x92', 'est-ce', 'Est-ce']:
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


def build_lda_model(nb, corpus, id2word, data_lemma):

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
def lda_traitements(nb, corpus, id2word, data_lemma):
    m, d, p, c, t = lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        nb, corpus, id2word, data_lemma)
    return m, d, p, c, t


def format_topics_sentences(ldamodel, corpus, data):
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
    return(sent_topics_df)


def topics_df(lda_model, corpus, data):  # mise en forme des résultats dans un df
    df_topic_sents_keywords = format_topics_sentences(
        lda_model, corpus, data)
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        'Document_Index', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    return df_dominant_topic