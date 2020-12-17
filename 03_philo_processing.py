#!/usr/bin/env python
# coding: utf-8


from gensim.models import CoherenceModel
import gensim.corpora as corpora
import spacy
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import os
import csv
from collections import Counter
from functions import *
import unidecode
import re
from gensim.utils import simple_preprocess
import gensim
import nltk
nltk.download('stopwords')
stop_words = stopwords.words('french')
stop_words.extend(['plus', 'meme', 'faire', 'tout',
                   'ainsi', 'aussi', 'donc', 'etc'])
nlp = spacy.load('fr_core_news_md')  # python -m spacy download fr_core_news_md

dir_tr = os.path.join(os.getcwd(), 'data', 'tr')
os.makedirs(dir_tr, exist_ok=True)
# dir_out = os.path.join(os.getcwd(), 'data', 'out')
# os.makedirs(dir_out, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Get clean data
df_all = pd.read_csv(os.path.join(dir_tr, 'clean_stats.csv'), sep='§',
                     encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


data = df_all.content_clower

# higher threshold fewer phrases.
bigram = gensim.models.Phrases(data, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

# ====== Exécution des fonctions
sentences = list(df_all.content_clower)
data = list(tokenize(sentences))
data_token, data_lemma = clean_lemmatize(data, bigram_mod, stop_words)
df_all['data_token'] = data_token
df_all['data_lemma'] = data_lemma

id2word = corpora.Dictionary(data_lemma)  # Dictionnaire
texts = data_lemma  # Create Corpus
corpus = [id2word.doc2bow(text) for text in texts]  # Term Document Frequency

df_lda_models_score = pd.DataFrame(
    columns=['num_topics', 'perplexity', 'coherence'])

for k in range(5, 40, 1):
    lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        k, corpus, id2word, data_lemma)
    L = df_lda_models_score.shape[0]
    df_lda_models_score.loc[L+1] = [k, perplexity_lda, coherence_lda]

nb = int(df_lda_models_score.loc[df_lda_models_score.idxmax(
    axis=0)['coherence'], 'num_topics'])

lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = lda_traitements(
    nb, corpus, id2word)

df_dominant_topic = topics_df(lda_model, corpus, data)
df_dominant_topic = df_dominant_topic[[
    'Document_Index', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]


# Save
df_dominant_topic.to_csv(os.path.join(dir_tr, f'dominant_topics.csv'), sep='§', index=False,
                         encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)