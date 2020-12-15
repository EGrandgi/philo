#!/usr/bin/env python
# coding: utf-8


from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv



# =============================================================================
#                                 Functions
# =============================================================================


def create_soup(url, enc):
    """ 
        In: url
        Out: source code  
    """

    req = requests.get(url)
    data = req.text
    soup = BeautifulSoup(data, from_encoding=enc)

    return soup



def basic_cleaner(s):
    
    return re.sub("[^A-Za-z.!?,; ]+", '', unidecode.unidecode(s.replace('\x92', ' ').lower()))



def flat_list(l: list) -> list:
    
    return [x for sub in l for x in sub]
    
