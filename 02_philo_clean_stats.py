#!/usr/bin/env python
# coding: utf-8

""" Uniformisation des 2 sources de données, statistiques
"""


import pandas as pd
import numpy as np
import os
import csv
from collections import Counter
from functions import basic_cleaner, flat_list
import unidecode
import re

dir_tr = os.path.join(os.getcwd(), 'data', 'tr')
os.makedirs(dir_tr, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# =============================================================================
#                      lecture des dataframes, concaténation
# =============================================================================


# maphilosophie - textes scrapés dans 01_philo_scraping.py
df = pd.read_csv(os.path.join(dir_tr, 'maphilosophie.csv'), sep='§',
                 encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

# acgrenoble - textes scrapés dans 01_philo_scraping.py
df2 = pd.read_csv(os.path.join(dir_tr, 'acgrenoble.csv'), sep='§',
                  encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

# concaténation des dataframes
df_all = pd.concat([df, df2], axis=0).reset_index(drop=True)

# corrections
df_all.year = [int(x) if str(x) != 'nan' else '' for x in df_all.year]
df_all.author = [x if str(x) != 'nan' else '' for x in df_all.author]

# ajout d'un id
df_all['id'] = range(df_all.shape[0])
df_all.id = [str(x).zfill(6) for x in df_all.id]

# nettoyage des textes : suppression des caractères spéciaux, des apostrophes, des (...) et [...]
# conservation de la casse
df_all['content_c'] = [basic_cleaner(s, False) for s in df_all.content]
df_all['content_clower'] = [basic_cleaner(
    s, True) for s in df_all.content]  # minuscules


# =============================================================================
#                         statistiques sur les textes
# =============================================================================


df_all['nchar'] = [len(s) for s in df_all.content_c]  # nombre de caractères
df_all['nwords'] = [s.count(' ') for s in df_all.content_c]  # nombre de mots
df_all['nsent'] = [s.count('.') + s.count('!') + s.count('?')
                   for s in df_all.content_c]  # nombre de phrases
df_all['nsent'] = [s if s > 0 else 1 for s in df_all.nsent]
df_all['nint'] = [s.count('?') for s in df_all.content_c]  # nombre de ?
df_all['nexcl'] = [s.count('!') for s in df_all.content_c]  # nombre de !
df_all['ncomm'] = [s.count(',') + s.count(';')
                   for s in df_all.content_c]  # nombre de , et ;
df_all['nupp'] = [len(re.findall('([A-Z])', s)) - (s.count('.') - s.count('...')
                                                   * 2 + s.count('!') + s.count('?')) for s in df_all.content_c]  # nombre de majuscules hors début de phrase
df_all['nupp'] = [x if x > 0 else 0 for x in df_all['nupp']]
df_all['nquot'] = [s.count('"') for s in df_all.content]  # nombre de "

# conversion en entiers
for var_ in ['nchar', 'nwords', 'nsent', 'nint', 'nexcl', 'ncomm']:
    df_all[var_] = [int(x) for x in df_all[var_]]

# ratios
df_all['rint'] = [np.round(x / y, 4)
                  for x, y in zip(df_all.nint, df_all.nchar)]  # nombre de ? sur nombre de caractères
df_all['rexcl'] = [np.round(x / y, 4)
                   for x, y in zip(df_all.nexcl, df_all.nchar)]  # nombre de ! sur nombre de caractères
df_all['rwords'] = [np.round(y / x, 2)
                    for x, y in zip(df_all.nsent, df_all.nwords)]  # nombre moyen de mots par phrase


# =============================================================================
#                        uniformisation des noms d'auteurs
# =============================================================================


unique_authors = np.unique(df_all.author)
unique_authors = [x for x in unique_authors if x != '']
df_corresp_auth = pd.DataFrame(columns=['author', 'author_full'])
df_corresp_auth.author = unique_authors
df_corresp_auth.author_full = [[x for x in unique_authors if a in x and '(' in x and '(' + a + ')' not in x][0] if len(
    [x for x in unique_authors if a in x and '(' in x and '(' + a + ')' not in x]) > 0 else a if '(' not in a and ')' not in a else a for a in unique_authors]
df_corresp_auth.author_full = [re.sub("[^A-Za-z.!?,; ]+", '', unidecode.unidecode(
    s.replace('\x92', ' '))).replace(' ', '_') for s in df_corresp_auth.author_full]
df_corresp_auth.author_full = ['Cournot_AntoineAugustin' if x ==
                               'Cournot_Antoine_Augustin' else x for x in df_corresp_auth.author_full]
df_corresp_auth.loc[df_corresp_auth.author ==
                    'Augustin', 'author_full'] = 'Augustin'
df_corresp_auth.loc[df_corresp_auth.author ==
                    'Hegel (Georg Wilhelm Friedrich)', 'author_full'] = 'Hegel_Georg_Wilhelm_F.'
df_all = pd.merge(df_all, df_corresp_auth, left_on='author',
                  right_on='author', how='outer')


# =============================================================================
#                                   sauvegarde
# =============================================================================


df_all.to_csv(os.path.join(dir_tr, 'clean_stats.csv'), sep='§', index=False,
              encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

