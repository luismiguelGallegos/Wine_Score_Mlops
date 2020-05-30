import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.pipeline import Pipeline

import re
import nltk

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.stem import *
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from sklearn.model_selection import train_test_split 

nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('winedata1.csv', sep=',', index_col=0)
data_t_t=data[['description','points']]
data_t_t=data_t_t.sample(frac=0.6,random_state=1).reset_index(drop=True)

#process wine description 
def process_text(raw_text):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))  
    not_stop_words = [w for w in words if not w in stops]
    
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`#"
    for p in punctuations:
        raw_text = raw_text.replace(p, f' {p} ')
    
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in not_stop_words]
    
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in lemmatized]
    
    return( " ".join( stemmed )) 

data_t_t['description'] = data_t_t['description'].apply(lambda x: process_text(x))


train, test = train_test_split(data_t_t, random_state=42, test_size=0.33, shuffle=True)
X_train = train.description
X_test = test.description

# Define a pipeline combining a text feature extractor with multi lable classifier
NB_pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words=stop_words)),
                ('clf', OneVsRestClassifier(MultinomialNB(
                    fit_prior=True, class_prior=None))),
            ])
 # train the model, there is more cleaning to do for a higher score
NB_pipeline.fit(X_train, train.points)
    # compute the testing accuracy
prediction = NB_pipeline.predict(X_test)
print(accuracy_score(test.points, prediction))


#user input create a dinamic variable program stops 
wine=input("Tell me your wine description : ")
wine2= [wine] 
print(NB_pipeline.predict(wine2))

wine=input("Tell me your wine description : ")
wine2= [wine] 
print(NB_pipeline.predict(wine2))

