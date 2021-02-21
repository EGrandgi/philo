#!/usr/bin/env python
# coding: utf-8

""" Traitements sur les textes :
- topic modelling
- similarités entre mots
- A poursuivre...

"""


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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# =============================================================================
#                               word embeddings
# =============================================================================

themes_list = [theme.split('|') for theme in df_all.theme]
theme_vectors = np.array([embed(theme, nlp, True, 0) for theme in themes_list])
data = df_all.content_clower
predicted_themes = [predict_theme(
    doc.split(' '), theme_vectors, themes_list, 1, nlp, True, 0) for doc in data]
predicted_themes = ['|'.join(x) for x in predicted_themes]
df_all['predicted_theme'] = predicted_themes


# =============================================================================
#                               topic modelling
# =============================================================================


# résultat de 02_philo_clean_stats.py
df_all = pd.read_csv(os.path.join(dir_tr, 'clean_stats.csv'), sep='§',
                     encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


data = df_all.content_clower

# modèle par bigrams
# seuil + <=> nb de phrases -
bigram = gensim.models.Phrases(data, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

# tokenisation et lemmatisation
sentences = list(df_all.content_clower)
data = list(tokenize(sentences))
data_token, data_lemma = clean_lemmatize(data, bigram_mod, stop_words)
df_all['data_token'] = data_token
df_all['data_lemma'] = data_lemma

id2word = corpora.Dictionary(data_lemma)  # dictionnaire
texts = data_lemma
# création du corpus - Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

df_lda_models_score = pd.DataFrame(
    columns=['num_topics', 'perplexity', 'coherence'])

# recherche du modèle au meilleur score
for k in range(5, 40, 1):
    lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        k, corpus, id2word, data_lemma)
    L = df_lda_models_score.shape[0]
    df_lda_models_score.loc[L+1] = [k, perplexity_lda, coherence_lda]

nb = int(df_lda_models_score.loc[df_lda_models_score.idxmax(
    axis=0)['coherence'], 'num_topics'])

lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = lda_traitements(
    nb, corpus, id2word, data_lemma)

df_dominant_topic = topics_df(lda_model, corpus, data)
df_dominant_topic = df_dominant_topic[[
    'Document_Index', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]


# sauvegarde
df_dominant_topic.to_csv(os.path.join(dir_tr, 'dominant_topics.csv'), sep='§', index=False,
                         encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


# =============================================================================
#                              similarités entre mots
# =============================================================================


# liste vocabulaire
vocab = [df_all.data_token.loc[k] for k in range(len(df_all))]
vocab = list(set(flat_list(vocab)))

# construction de la matrice - fréquence des mots
mat = scipy.sparse.lil_matrix((len(vocab), len(df_all)))
for i in range(len(vocab)):
    word = vocab[i]
    for j in range(len(df_all)):
        doc = df_all.index[j]
        text = df_all.loc[j, 'data_token']
        if word in text:
            # matrice de fréquences des mots dans les documents
            mat[i, j] = text.count(word)

# quelques tests : similarités cosinus entre mots
# localisation des mots cibles dans le vocabulaire (numéro de ligne de la matrice)
ind_desir = vocab.index('desir')
ind_mal = vocab.index('mal')
ind_passion = vocab.index('passion')
ind_bien = vocab.index('bien')
ind_ame = vocab.index('ame')

# récupération des vecteurs des mots
vect_desir = mat[ind_desir].toarray()
vect_mal = mat[ind_mal].toarray()
vect_passion = mat[ind_passion].toarray()
vect_bien = mat[ind_bien].toarray()
vect_ame = mat[ind_ame].toarray()

vects = [vect_desir, vect_mal, vect_passion, vect_bien, vect_ame]
vects_names = ['desir', 'mal', 'passion', 'bien', 'ame']
df_sim = pd.DataFrame(columns=vects_names, index=vects_names)

# calcul des similarités cosinus
for i in range(len(vects)):
    for j in range(len(vects)):
        sim = 1 - spatial.distance.cosine(vects[i], vects[j])
        df_sim.loc[vects_names[i], vects_names[j]] = round(sim, 4)
df_sim


# à poursuivre...




