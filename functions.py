# -*- coding: utf-8 -*-
"""

@author: EGrandgi

"""


from bs4 import BeautifulSoup
import requests
import pandas as pd
import csv



# =============================================================================
#                                 Functions
# =============================================================================


def create_soup(url):
    """ 
        In: url
        Out: source code  
    """

    req = requests.get(url)
    data = req.text
    soup = BeautifulSoup(data, 'lxml')

    return soup


