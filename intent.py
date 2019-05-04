import sklearn
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  

target = []
data = []
with open('testSet-qualifiedBatch-fixed.txt') as file:
    for line in file:
        line_s = line.split("\t")
        target.append(line_s[0])
        data.append(line_s[1])

documents = data
y = target


tfid = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.9)  
X = tfid.fit_transform(data).toarray() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify = y)  

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train)  

y_fin = classifier.predict(X_test)  

print(confusion_matrix(y_test,y_fin))  
print(classification_report(y_test,y_fin))  
print(accuracy_score(y_test, y_fin))
