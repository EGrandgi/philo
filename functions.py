#!/usr/bin/env python
# coding: utf-8


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
from nltk.corpus import stopwords
stop_words = stopwords.words('french')
import spacy
nlp = spacy.load('fr_core_news_md')


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


def remove_stopwords(texts: list) -> list:
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(
            [token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def clean_lemmatize(data: list):
    data_tokens_nostops = remove_stopwords(data)
    data_tokens_bigrams = make_bigrams(data_tokens_nostops)
    data_lemmas_bigrams = lemmatization(data_tokens_bigrams, allowed_postags=[
                                        'NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_tokens_bigrams, data_lemmas_bigrams