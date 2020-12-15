#!/usr/bin/env python
# coding: utf-8


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
# dir_out = os.path.join(os.getcwd(), 'data', 'out')
# os.makedirs(dir_out, exist_ok=True)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Get data from philo_scraping
df = pd.read_csv(os.path.join(dir_tr, 'maphilosophie.csv'), sep='§',
                 encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

# Get data from acgrenoble
df2 = pd.read_csv(os.path.join(dir_tr, 'acgrenoble.csv'), sep='§',
                  encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


# concatenate dataframes
df_all = pd.concat([df, df2], axis=0).reset_index(drop=True)

# correction
df_all.year = [int(x) if str(x) != 'nan' else '' for x in df_all.year]
df_all.author = [x if str(x) != 'nan' else '' for x in df_all.author]

# id
df_all['id'] = range(df_all.shape[0])
df_all.id = [str(x).zfill(6) for x in df_all.id]

# clean content
df_all['content_c'] = [basic_cleaner(s) for s in df_all.content]

# stats
df_all['nchar'] = [len(s) for s in df_all.content_c]
df_all['nwords'] = [s.count(' ') for s in df_all.content_c]
df_all['nsent'] = [s.count('.') + s.count('!') + s.count('?')
                   for s in df_all.content_c]
df_all['nsent'] = [s if s > 0 else 1 for s in df_all.nsent]
df_all['nint'] = [s.count('?') for s in df_all.content_c]
df_all['nexcl'] = [s.count('!') for s in df_all.content_c]
df_all['ncomm'] = [s.count(',') + s.count(';') for s in df_all.content_c]

# convert to int
for var_ in ['nchar', 'nwords', 'nsent', 'nint', 'nexcl', 'ncomm']:
    df_all[var_] = [int(x) for x in df_all[var_]]

# ratios
df_all['rint'] = [np.round(x / y, 4)
                  for x, y in zip(df_all.nint, df_all.nchar)]
df_all['rexcl'] = [np.round(x / y, 4)
                   for x, y in zip(df_all.nexcl, df_all.nchar)]
df_all['rwords'] = [np.round(y / x, 2)
                    for x, y in zip(df_all.nsent, df_all.nwords)]

# authors
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


# TO BE CONTINUED


# Save
df_all.to_csv(os.path.join(dir_tr, f'clean_stats.csv'), sep='§', index=False,
              encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

