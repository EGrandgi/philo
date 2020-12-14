#!/usr/bin/env python
# coding: utf-8


from bs4 import BeautifulSoup
import requests
import pandas as pd
import os
import csv
from functions import create_soup

dir_tr = os.path.join(os.getcwd(), 'tr')
os.makedirs(dir_tr, exist_ok=True)


# ## http://www.maphilosophie.fr/textes.php


url = 'http://www.maphilosophie.fr/textes.php'
soup = create_soup(url)

# Get texts links
links = []
for td in soup.find_all('td', class_='cellules'):
    if td.find('a') is not None:
        link = td.find('a')['href']
        if link.split('?$cle=')[1] != '':
            link = 'http://www.maphilosophie.fr/' + link.replace(' ', '%20').replace("'", "%27").replace(
                'è', '%E8').replace('é', '%E9')
            links.append(link)


# Store content in a dataframe
df = pd.DataFrame(columns=['link', 'author', 'title',
                           'subtitle', 'year', 'content'])
df.link = links

for k in df.index:
    print(k)
    link = df.loc[k, 'link']
    soup = create_soup(link)
    df.loc[k, 'author'] = soup.find('div', class_='col-lg-12').find('h1').text
    df.loc[k, 'subtitle'] = soup.find(
        'div', class_='col-lg-12').find('h2').text
    info = soup.find('div', class_='ref').text.split(',')
    df.loc[k, 'title'] = info[1]
    df.loc[k, 'year'] = info[len(info) - 1]
    df.loc[k, 'content'] = soup.find(
        'div', class_='corps').text.replace('\x92', "'")


# Save
website = 'maphilosophie'
df.to_csv(os.path.join(dir_tr, f'{website}.csv'),
          sep='§', index=False, encoding='utf-8-sig')

