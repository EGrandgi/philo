#!/usr/bin/env python
# coding: utf-8


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
nltk.download('stopwords')
stop_words = stopwords.words('french')
nlp = spacy.load('fr_core_news_md')
from gensim.models import CoherenceModel



# =============================================================================
#                                 Utils
# =============================================================================


def flat_list(l: list) -> list:

    return [x for sub in l for x in sub]


# =============================================================================
#                                 Scraping
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
#                                 NLP
# =============================================================================


def basic_cleaner(s: str, lower: bool) -> str:

    s = re.sub("[^A-Za-z.!?,;' ]+", '', unidecode.unidecode(s.replace('(...)',
                                                                      '').replace('[...]', '').replace('\x92', ' ').replace('\x9c', 'oe'))).replace("'", ' ').replace('  ', ' ')
    s = s if not lower else s.lower()

    return s


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
def lda_traitements(nb, corpus, id2word):
    m, d, p, c, t = lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        nb, corpus, id2word, data_lemma)
    return m, d, p, c, t


def format_topics_sentences(ldamodel, corpus, data):
    sent_topics_df = pd.DataFrame()

    # Dans chaque document, trouver le topic dominant
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Trouver le topic dominant, la contribution du doc au modèle et les keywords pour chaque doc
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # si topic dominant
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