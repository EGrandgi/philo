#!/usr/bin/env python
# coding: utf-8

""" Web scraping - Philosophical texts - in French
2 websites:
- http://www.maphilosophie.fr/textes.php
- http://www.ac-grenoble.fr/PhiloSophie/old2/bases/search.php (moteur de recherche par auteur)
Saved as csv: text, author, title, theme, year, link, source

"""


from functions import create_soup, flat_list, basic_cleaner
import pandas as pd
import numpy as np
import os
import csv
import unidecode
import re

dir_tr = os.path.join(os.getcwd(), 'data', 'tr')
os.makedirs(dir_tr, exist_ok=True)


# =============================================================================
#                              maphilosophie website
# =============================================================================


url = 'http://www.maphilosophie.fr/textes.php'
soup = create_soup(url, 'cp1252')

print(f'get text links from {url}')
links = []
for td in soup.find_all('td', class_='cellules'):
    if td.find('a') is not None:
        link = td.find('a')['href']
        if link.split('?$cle=')[1] != '':
            link = 'http://www.maphilosophie.fr/' + link.replace(' ', '%20').replace("'", "%27").replace('è', '%E8').replace('é', '%E9').replace('û', '%FB').replace('ê', '%EA').replace('î', '%EE').replace(
                'à', '%E0').replace('É', '%C9').replace('ô', '%F4').replace('ç', '%E7').replace('Ê', '%CA').replace('â', '%E2').replace('ï', '%EF').replace('\x92', '%92').replace('\x9c', '%9C')
            links.append(link)


print(f'get content from {len(links)} urls')
df = pd.DataFrame(columns=['source', 'link', 'author',
                           'title', 'theme', 'year', 'content'])
df.link = links
df.source = 'maphilosophie.fr'

for k in df.index:
    link = df.loc[k, 'link']
    soup = create_soup(link, 'cp1252')
    author = soup.find('div', class_='col-lg-12').find('h1').text
    author = author[:len(author) - 1] if author[len(author) -
                                                1:len(author)] == ' ' else author
    df.loc[k, 'author'] = author
    author_short = author.split('(')[0]
    author_short = author_short[:len(author_short) - 1] if author_short[len(
        author_short) - 1:len(author_short)] == ' ' else author_short
    df.loc[k, 'author_short'] = author_short
    df.loc[k, 'theme'] = soup.find(
        'div', class_='col-lg-12').find('h2').text
    info = soup.find('div', class_='ref').text.split(',')
    df.loc[k, 'title'] = info[1] if len(info) > 0 else ''

    year = info[len(info) - 1] if len(info) > 0 else ''
    year = re.findall('[1-3][0-9]{3}', year)
    year = ['0'] if year == [] else year
    year = min([int(x) for x in year])
    df.loc[k, 'year'] = year

    df.loc[k, 'content'] = soup.find('div', class_='corps').text.replace(
        '\x92', "'").replace('', '').replace('\n', ' ')

print('process year')
df.year = [x if x != 0 else '' for x in df.year]

print('process theme')
df.theme = [basic_cleaner(x, True, False) for x in df.theme]
df.theme = [x.replace(' - ', '|').replace(' ', '|').replace('/', '|').replace('&', '').replace('||', '|').replace("l'", '').replace('|de|', '|').replace(
    '|la|', '|').replace('|le|', '|').replace('|du|', '|').replace('\r', '').replace('\n', '').replace('(s', '').replace(')', '').replace("l'", '') for x in df.theme]
df.theme = [[z for z in x.split('|') if z not in ['la', 'le', 'un', 'une', 'les', 'des', 'du', 'de', 'et', 'en', 'quoi', 'au', 'pourquoi',
                                                  'sans', '?', '!', '', 'a', '(2)', 'est', 'que', 'ce', 'dans', 'nos', 'comme', 'il', 'ii', 'aux', 'par']] for x in df.theme]
df.theme = [[z.replace('"', '')
             for z in x if "'" not in z and '-' not in z] for x in df.theme]
df.theme = ['|'.join(x) for x in df.theme]

print('clean df')
inds = [i for i, x in zip(df.index, df.author_short) if 'monde' in x.lower()]
df = df.drop(inds, axis=0).reset_index(drop=True)

website = 'maphilosophie'
print(f'save df to {dir_tr} as {website}.csv')
df.to_csv(os.path.join(dir_tr, f'{website}.csv'), sep='§', index=False,
          encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)


# =============================================================================
#                       Académie Grenoble Philo website
# =============================================================================


# link: http://www.ac-grenoble.fr/PhiloSophie/old2/bases/search.php
# search engine by author name

print('get list of authors from maphilosophie, process list and create list of http://www.ac-grenoble.fr links')
authors_list = np.unique(df.author_short)
df = df.drop(['author_short'], axis=1)
authors_list = [unidecode.unidecode(x.lower()).replace(
    ' ', '+').replace("'", '%27').replace('-', '+') for x in authors_list if x != '']
authors_list.sort()
links = [
    f'http://www.ac-grenoble.fr/PhiloSophie/old2/bases/search.php?auteur={a}&texte=&reference=&theme=' for a in authors_list]

df2 = pd.DataFrame(columns=['source', 'link', 'author',
                            'title', 'theme', 'year', 'content'])
auth_vect = []
cont_vect = []
ref_vect = []
th_vect = []
year_vect = []
link_vect = []

print(f'get content from {len(links)} author links')
for link in links:

    soup = create_soup(link, 'cp1252')

    # get content
    author = [span.text for span in soup.find_all(
        'span', class_='sstitremarron')]
    content = [span.find('div', {'style': 'text-align: justify;'}).text.replace(
        '\x92', "'").replace('\r', '').replace('\n', '') for span in soup.find_all(
        'span', class_='texte') if span.find('div', {'style': 'text-align: justify;'}) is not None]
    reference = [span.find('div', {'style': 'text-align: right;'}).text.replace(
        '\x92', "'").replace('\r', '').replace('\n', '') for span in soup.find_all(
        'span', class_='texte') if span.find('div', {'style': 'text-align: right;'}) is not None]
    themes = [td.find('b').text for td in soup.find_all(
        'td', {'align': 'right'})]
    years = [re.findall('[1-3][0-9]{3}', s) for s in reference]

    # append to list
    auth_vect.append(author)
    cont_vect.append(content)
    ref_vect.append(reference)
    th_vect.append(themes)
    year_vect.append(years)
    link_vect.append([link] * len(themes))


# flatten
auth_vect = flat_list(auth_vect)
cont_vect = flat_list(cont_vect)
ref_vect = flat_list(ref_vect)
th_vect = flat_list(th_vect)
year_vect = flat_list(year_vect)
link_vect = flat_list(link_vect)

print('process year')
year_vect = [['0'] if x == [] else x for x in year_vect]
year_vect = [min([int(z) for z in x]) for x in year_vect]
year_vect = [x if x != 0 else '' for x in year_vect]

print('fill df')
df2['link'] = link_vect
df2['author'] = auth_vect
df2['title'] = ref_vect
df2['theme'] = th_vect
df2['content'] = cont_vect
df2['year'] = year_vect
df2['source'] = 'ac-grenoble.fr'

print('process theme')
df2.theme = [basic_cleaner(x, True, False) for x in df2.theme]
df2.theme = [x.replace(' - ', '|').replace(' ', '|').replace('/', '|').replace('&', '').replace('||', '|').replace("l'", '').replace('|de|', '|').replace(
    '|la|', '|').replace('|le|', '|').replace('|du|', '|').replace('\r', '').replace('\n', '').replace('(s', '').replace(')', '').replace("l'", '') for x in df2.theme]
df2.theme = [[z for z in x.split('|') if z not in ['la', 'le', 'un', 'une', 'les', 'des', 'du', 'de', 'et', 'en', 'quoi', 'au', 'pourquoi',
                                                   'sans', '?', '!', '', 'a', '(2)', 'est', 'que', 'ce', 'dans', 'nos', 'comme', 'il', 'ii', 'aux', 'par']] for x in df2.theme]
df2.theme = [[z.replace('"', '')
              for z in x if "'" not in z and '-' not in z] for x in df2.theme]
df2.theme = ['|'.join(x) for x in df2.theme]

website = 'acgrenoble'
print(f'save df to {dir_tr} as {website}.csv')
df2.to_csv(os.path.join(dir_tr, f'{website}.csv'), sep='§', index=False,
           encoding='utf-8-sig', escapechar='\\', quoting=csv.QUOTE_NONE)