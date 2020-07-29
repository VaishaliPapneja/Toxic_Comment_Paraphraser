# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)


import os 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from bestSyn import *
from nltk.stem.snowball import SnowballStemmer
import re
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

# %%
import nltk
nltk.download('stopwords')

# %%
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

# %%
@app.route('/predict',methods=['POST'])
def predict():
    
    stemmer = SnowballStemmer("english")
    def preprocessing(text):
        words_to_remove = ["ax","i","you","edu","s","t","m","subject","can","lines","re","what", "there","all","we",
                "one","the","a","an","of","or","in","for","by","on","but","is","in","a","not","with","as",
                "was","if","they","are","this","and","it","have","has","from","at","my","be","by","not","that","to",
                "from","com","org","like","likes","so","said","from","what","told","over","more","other",
                "have","last","with","this","that","such","when","been","says","will","also","where","why",
                "would","today", "in", "on", "you", "r", "d", "u", "hw","wat", "oly", "s", "b", "ht", 
                "rt", "p","the","th", "n", "was"]
        text = re.sub(r'\'|\"|\,|\.|\?|\+|\-|\/|\=|\(|\)|\n|"', '',text)
        text = re.sub("  ", " ", text)
        text = re.sub(r'\B<U+.*>|<U+.*>\B|<U+.*>','',text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9]', " ",text)
        text = re.sub(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"",text)
        text = re.sub(r'amp',"",text)
        text = re.sub(r'^\s+|\s+$'," ",text)
        #remove stopwords and words_to_remove
        stop_words = set(stopwords.words('english'))
        mystopwords = [stop_words, "via", words_to_remove]
        text = ' '.join([word for word in text.split() if word not in mystopwords])
        return text
    def processing(text):
        tokenized_text = word_tokenize(text)
        tokenized_text = [y for y in tokenized_text if not any(c.isdigit() for c in y)]
        stemmed_text = [stemmer.stem(word) for word in tokenized_text]
        return ' '.join(stemmed_text)

    if request.method == 'POST':
        CV_model = open('CV_model.pkl','rb')
        cv = joblib.load(CV_model)
        
        NB_toxic_classification_model = open('NB_toxic_classification_model.pkl','rb')
        clf = joblib.load(NB_toxic_classification_model)
        
        message = request.form['message']
        
        data = message
        toxic_list, sync_list = [], []
        flag=0
        for word in data.split():
            word_copy = word
            word = preprocessing(word)
            word = processing(word)
            vect = cv.transform([word]).toarray()
            my_prediction = clf.predict(vect)
            if(my_prediction=='ok'):
                pass
            else:
                toxic_list.append(word_copy)
                flag=1
                
    for word in toxic_list:
        word_syn = BestSyn(word).pull()[1]
        sync_list.append(word_syn)
    my_prediction = "Wait!! Your message appears to be toxic.\n"+"Would you like to change the following words: \n"+ (" ".join(toxic_list)) + " with " + (" ".join(sync_list)) if flag==1 else "Your message seems to be fine."
    return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)

# %%
