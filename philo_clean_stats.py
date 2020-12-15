import pandas as pd
import numpy as np
import os
import csv
from collections import Counter
from functions import basic_cleaner

dir_tr = os.path.join(os.getcwd(), 'tr')
os.makedirs(dir_tr, exist_ok=True)
# dir_out = os.path.join(os.getcwd(), 'out')
# os.makedirs(dir_out, exist_ok=True)


# Get data from philo_scraping
df = pd.read_csv(os.path.join(dir_tr, 'maphilosophie.csv'), sep='§',
                 encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

# id
df['id'] = range(df.shape[0])
df.id = [str(x).zfill(6) for x in df.id]

# clean content
df['content_c'] = [basic_cleaner(s) for s in df.content]

# stats
df['nchar'] = [len(s) for s in df.content_c]
df['nwords'] = [s.count(' ') for s in df.content_c]
df['nsent'] = [s.count('.') + s.count('!') + s.count('?') for s in df.content_c]
df['nint'] = [s.count('?') for s in df.content_c]
df['nexcl'] = [s.count('!') for s in df.content_c]
df['ncomm'] = [s.count(',') + s.count(';') for s in df.content_c]

# convert to int
for var_ in ['nchar', 'nwords', 'nsent', 'nint', 'nexcl', 'ncomm']:
    df[var_] = [int(x) for x in df[var_]]
    
# ratios
df['rint'] = [np.round(x / y, 4) for x, y in zip(df.nint, df.nchar)]
df['rexcl'] = [np.round(x / y, 4) for x, y in zip(df.nexcl, df.nchar)]
df['wordssent'] = [np.round(y / x, 2) for x, y in zip(df.nsent, df.nwords)]


# TO BE CONTINUED



# Save
df.to_csv(os.path.join(dir_tr, f'clean_stats.csv'), sep='§', index=False,
          encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)

