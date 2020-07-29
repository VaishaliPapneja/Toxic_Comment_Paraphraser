# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)

from nltk.stem.snowball import SnowballStemmer

import os 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from gensim.models import word2vec

# %%
from spacy.tokenizer import Tokenizer
from spacy.lang.en.examples import sentences
from nltk.corpus import wordnet
import spacy
import urllib
nlp = spacy.blank('en')
from datamuse import datamuse
api = datamuse.Datamuse()

# %%
class BestSyn:

    def get_datamuse_syn_list(self):
        json_data = api.words(ml=self.word, max=100)
        print('Json Data is:', json_data)
        word_list = []
        for x in json_data:
            word_list.append(x['word'])
        return word_list

    def __init__(self, word):
        self.word = word
        self.best_score = 0.4
        self.best_choice = ""


    def pull(self):
        words_list = self.get_datamuse_syn_list()
        nltk_score, score = 0, 0
        for syn_word in words_list:
            use_nltk = True
            nltk_raw_word = wordnet.synsets(self.word)[0]
            nltk_syn_word = wordnet.synsets(syn_word)[0]
            
            spacy_raw_word = nlp(self.word.lower())
            
            spacy_syn_word = nlp(syn_word.lower())

            
            spacy_score = spacy_raw_word.similarity(spacy_syn_word)
            
            if (use_nltk == True):
                nltk_score = nltk_syn_word.wup_similarity(nltk_raw_word)
                if (nltk_score == None):
                    nltk_score = 0
                score = (nltk_score+spacy_score)/2
            else:
                score = spacy_score

                
            if (((nltk_score>0.3) & (nltk_score<0.4)) | ((score>0.3) & (self.best_choice==""))):
                self.best_score = score
                self.best_choice = syn_word
        return [self.best_score, self.best_choice]

        

    
def __del__(self):
        self.word = False
        self.best_score = False
        self.best_choice = False
