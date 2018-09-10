# Import the libraries

import numpy as np
import pandas as pd
import nltk
import pickle
import re

#Import the dataset

from nltk.corpus import stopwords
from sklearn.datasets import load_files

nltk.download('stopwords')

reviews = load_files('txt_sentoken/')
X,y = reviews.data, reviews.target

# Storing as Pickle files

with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)

    
# Unpickling the dataset

with open('X.pickle','rb') as f:
    X=pickle.load(f)
        
with open('y.pickle','rb') as f:
    y=pickle.load(f)

# Creating the corpus
corpus = []
for i in range(len(X)):
    review = re.sub(r'\W',' ',str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ',review)
    review = re.sub(r'^[a-z]\s+',' ',review)
    review = re.sub(r'\s+',' ',review)
    corpus.append(review)
            
# Transfering into the BOW MODEL
from sklearn.feature_extraction.text import CountVectorizer
Vectorizer = CountVectorizer(max_features = 2000, min_df =3, max_df = 0.6, stop_words = stopwords.words('english'))
X = Vectorizer.fit_transform(corpus).toarray()    

# Transform the BOW Model into TF-IDF Model
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


# Transfering into the TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
Vectorizer = TfidfVectorizer(max_features = 2000, min_df =3, max_df = 0.6, stop_words = stopwords.words('english'))
X = Vectorizer.fit_transform(corpus).toarray()    


# Creating Test and Train Data

from sklearn.model_selection import train_test_split
text_train,text_test,sent_train,sent_test =train_test_split(X,y,test_size=0.2,random_state=0)

# Use Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(text_train,sent_train)

# Testing the model performance

from sklearn.metrics import confusion_matrix

sent_pred = classifier.predict(text_test)
cm = confusion_matrix(sent_test,sent_pred)

print(cm)

# Pickle the classifier

with open('classifier.pickle','wb') as f:
    pickle.dump(classifier, f)
    
# Pickle the Vectorizer
    
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(Vectorizer, f)

# Unpickling the classifier and Vectorizer
  
with open('classifier.pickle','rb') as f:
    cs = pickle.load(f)
    
with open('tfidfmodel.pickle','rb') as f:
    tfidf = pickle.load(f)
