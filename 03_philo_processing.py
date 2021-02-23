#!/usr/bin/env python
# coding: utf-8

""" Traitements sur les textes :
- topic modelling
- similarités entre mots
- A poursuivre...

"""


# utils
from functions import *
import scipy
from scipy import spatial
import os
import csv

# plot
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# text mining / NLP
import gensim.corpora as corpora
from textblob import TextBlob
from textblob_fr import PatternTagger, PatternAnalyzer
import nltk
from nltk.corpus import stopwords

# stop words
nltk.download('stopwords')
stop_words = stopwords.words('french')
stop_words.extend(['plus', 'meme', 'faire', 'tout',
                   'ainsi', 'aussi', 'donc', 'etc',
                   'si', 'comme', 'quelque', 'dont',
                   'elles', 'elle', 'ils', 'il', 'si',
                   'tous', 'toutes', 'peut', 'cette',
                   'entre', 'sans', 'quand', 'toute',
                   'encore', 'celui', 'cela', 'point',
                   'celui', 'celle', 'ceux', 'celles',
                   'ni', 'car', 'toujours', 'jamais',
                   'souvent', 'parfois'])

# local
dir_tr = os.path.join(os.getcwd(), 'data', 'tr')
os.makedirs(dir_tr, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# =============================================================================
#                               data processing
# =============================================================================

print(f'get concatenated df (from 02_philo_clean_stats.py)')
df_all = pd.read_csv(os.path.join(dir_tr, 'clean_stats.csv'), sep='§',
                     encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE, engine='python')

print('tokenise and lemmatise')
sentences = list(df_all.content_clower)
data = list(tokenize(sentences))

bigram = gensim.models.Phrases(data, min_count=5, threshold=100)  # threshold + <=> nb of sentences
bigram_mod = gensim.models.phrases.Phraser(bigram)

data_token, data_lemma = clean_lemmatize(data, bigram_mod, stop_words)
df_all['data_token'] = data_token
df_all['data_lemma'] = data_lemma


# =============================================================================
#                                   analysis
# =============================================================================

# ========================================
#             word embeddings
# ========================================

themes_list = [theme.split('|') for theme in df_all.theme]
theme_vectors = np.array([embed(theme, nlp, True, 0) for theme in themes_list])
data = df_all.content_clower
predicted_themes = [predict_theme(
    doc.split(' '), theme_vectors, themes_list, 1, nlp, True, 0) for doc in data]
predicted_themes = ['|'.join(x) for x in predicted_themes]
df_all['predicted_theme'] = predicted_themes


# ========================================
#             topic modelling
# ========================================

print('get dictionary')
id2word = corpora.Dictionary(data_lemma)
texts = data_lemma

print('create corpus - Term Document Frequency')
corpus = [id2word.doc2bow(text) for text in texts]

print('search best score model')
df_lda_models_score = pd.DataFrame(
    columns=['num_topics', 'perplexity', 'coherence'])

for k in range(5, 40, 1):
    lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = build_lda_model(
        k, corpus, id2word, data_lemma)
    L = df_lda_models_score.shape[0]
    df_lda_models_score.loc[L+1] = [k, perplexity_lda, coherence_lda]

nb = int(df_lda_models_score.loc[df_lda_models_score.idxmax(
    axis=0)['coherence'], 'num_topics'])

print(f'compute model with {nb} topics')
lda_model, doc_lda, perplexity_lda, coherence_lda, model_topics = lda_traitements(
    nb, corpus, id2word, data_lemma)

print('compute dominant topics df')
df_dominant_topic = topics_df(lda_model, corpus, data)
df_dominant_topic = df_dominant_topic[[
    'Document_Index', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords']]

df_dominant_topic[df_dominant_topic.Topic_Perc_Contrib > 0.6]

print(f'save dominant topics df to {dir_tr} as dominant_topics.csv')
df_dominant_topic.to_csv(os.path.join(dir_tr, 'dominant_topics.csv'), sep='§', index=False,
                         encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


# ========================================
#       similarities between words
# ========================================


print('vocabulary')
vocab = [df_all.data_lemma.loc[k] for k in range(len(df_all))]
vocab = flat_list(vocab)
c = Counter(vocab)
vocab = list(set(vocab))

df_most_common_words = most_common(c, 'words', 40)
# plot_most_common(df_most_common_words, 'words')
wordcloud(df_most_common_words, 'words')

print('compute matrix - words frequencies in documents')
mat = scipy.sparse.lil_matrix((len(vocab), len(df_all)))
for i in range(len(vocab)):
    word = vocab[i]
    for j in range(len(df_all)):
        doc = df_all.index[j]
        text = df_all.loc[j, 'data_lemma']
        if word in text:
            mat[i, j] = text.count(word)
            
# some tests: cosine similarities between words
# words = ['nature', 'objet']
words = list(df_most_common_words.loc[:20].words)

print('locate target words in vocab and get words vectors')  # line number in the matrix
vects = []
for w in words:
    exec(f'ind_{w} = vocab.index("{w}")')
    exec(f'vect_{w} = mat[ind_{w}].toarray()')
    exec(f'vects.append(vect_{w})')
    
df_sim = pd.DataFrame(columns=words, index=words)
for i in range(len(vects)):
    for j in range(len(vects)):
        sim = 1 - spatial.distance.cosine(vects[i], vects[j])
        df_sim.loc[words[i], words[j]] = round(sim, 4)
        

# ========================================
#            sentiment analysis
# ========================================

df_all = sentiment(df_all, 'content_clower')
df_sentiment = df_all[['author', 'content_clower', 'polarity', 'subjectivity',
                       'polarity_class', 'subjectivity_class', 'polarity_subjectivity']]
df_sentiment.head()


