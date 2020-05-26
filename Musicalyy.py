# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:11:17 2020

@author: Dell
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
from nltk.corpus import wordnet
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re,string,unicodedata
from string import punctuation
from xgboost import XGBClassifier
#from sklearn.ensemble import AdaBoostClassifier
df=pd.read_csv("datasets/Musical_instruments_reviews.csv")
print(df.shape)
train_date=df['reviewTime'].str.split(' ',n=2,expand = True)
df['month']=train_date[1]
df['days']=train_date[0]
df['year']=train_date[2]
df=df.drop(['reviewTime'],axis=1)
#sns.distplot(df.overall,bins=20, kde=False,color='purple')
#plt.show()
print(df.head(2))
df['reviewText'] = df['reviewText'] + ' ' + df['summary']
del df['summary']
df['reviewText'].fillna("",inplace = True)
df['helpful'].fillna("0",inplace = True)

en=LabelEncoder()
df[['asin','helpful','reviewerID']]=df[['asin','helpful','reviewerID']].apply(en.fit_transform)
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
def clean_text(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            word = i.strip().lower()
            final_text.append(word)
    return " ".join(final_text) 
df['reviewText'] = df['reviewText'].apply(clean_text)
#sns.distplot(df.helpful,bins=20, kde=False,color='purple')
#plt.show()
# print(df.columns)
# unhelp=df['helpful'].nunique()
# un_sum=df['summary'].nunique()
#print(un_sum,unhelp)
cv=CountVectorizer()
#transformed train reviews
x=cv.fit_transform(df['reviewText'])
data_sample=df[0:10]
cv2=CountVectorizer()
x=cv2.fit_transform(data_sample['reviewText'])
print(x.shape)
df1=pd.DataFrame(x.toarray(),columns=cv2.get_feature_names())
print(df1.head())
dff=pd.concat([df,df1])
dff.update(dff[['years','write','wrong','worth','works','would','helpful','00', '100','1980', '1996', '20', '21', '30', '50',
'90', 'able', 'added', 'affordable', 'allowing', 'alone', 'amazon', 'amp', 'another', 'aroma', 'around', 'arrived', 'as',
'at', 'attached', 'attaches', 'avoid', 'back', 'bass', 'best', 'better', 'block', 'blocks', 'board', 'bonus', 'bought', 
'break', 'breath', 'but', 'buy', 'cable', 'cables', 'candy', 'cannot', 'cant', 'careful', 'carefully', 'carries', 'chain', 
'clamp', 'cloth', 'coaxing', 'coil', 'coloration', 'come', 'comes', 'connectors', 'constructed', 'cord', 'cost', 'crisp', 
'day', 'degree', 'despite', 'device', 'did', 'dif', 'disappointed', 'doesnt', 'done', 'double', 'eighties', 'either', 
'eliminate', 'end', 'enough', 'epiphone', 'even', 'exactly', 'expected', 'expensive', 'fact', 'fender', 'filter', 'filters',
'fit', 'for', 'found', 'frequencies', 'get', 'gets', 'getting', 'go', 'gold', 'good', 'goose', 'gooseneck', 'got', 'grape', 'great', 'guess', 'guitar', 'harm', 'heavy', 'here', 'high', 'hint', 'hold', 'honestly', 'hook', 'hurt', 'ii', 'input', 
'instructions', 'instrument', 'isnt', 'it', 'jack', 'jacks', 'jake', 'job', 'keep', 'last', 'learned', 'lets', 'lifetime', 'like', 'line', 'little', 'looks', 'lot', 'love', 'lowest', 'makes', 'marginally', 'market', 'may', 'metal', 'mic', 'might', 'mike', 'mine', 'money', 'monster', 'mount', 'much', 'mxl', 'neck', 'needed', 'needs', 'never', 'new', 'next', 'nice', 'night', 'nose', 'noticeable', 'old', 'one', 'ones', 'otherwise', 'output', 'pass', 'pay', 'payed', 'pedal', 'perfect', 'performs', 'planet', 'pleasing', 'plug', 'plus', 'pop', 'popping', 'pops', 'position', 'positioning', 'prevents', 'price', 'prices', 'pricing', 'primary', 'problems', 'produce', 'product', 'protects', 'put', 'putting', 'quite', 'read', 'realized', 'reason', 'record', 'recorded', 'recording', 'recordings', 'reduction', 'reminiscent', 'replace', 'requires', 'return', 'rig', 'run', 'sagging', 'sake', 'save', 'screen', 'screened', 'screens', 'secure', 'series', 'several', 'sheraton', 'simple', 'since', 'sing', 'small', 'smell', 'smelling', 'sound', 'sounds', 'stand', 'standard', 'stated', 'stay', 'still', 'stop', 'strat', 
'studio', 'supposed', 'thing', 'thought', 'time', 'to', 'top', 'try', 'up', 'update', 'use', 'used', 'using', 'vocals', 'voice', 'volume', 
'wanted', 'warranty', 'waves', 'well', 'went', 'windscreen', 'wins', 'work', 'working','asin','overall','reviewerID', 'reviewerName']].fillna(0))

# print(print(dff[dff["overall"].isnull()][null_columns]))
# cat_feature = [col for col in dff.columns if dff[col].dtypes == "O"]
# print(cat_feature)
#print(df['reviewText'])
# num_feature = [col for col in dff.columns if dff[col].dtypes != "O"]
# print(num_feature)

y=dff['overall']
x=dff.drop(['overall','reviewerName', 'reviewText', 'month', 'days', 'year','unixReviewTime'],axis=1)
null_columns=dff.columns[dff.isnull().any()]
print(null_columns)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
model = XGBClassifier()
# model=SVC(C=1,gamma=0.01)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))    
y_pred = model.predict(x_test)      

# evaluate predictions
accuracy = model.score(y_test, y_pred)
print("Accuracy: ",accuracy)

# #transformed test reviews
# cv_test_reviews=cv.transform(x_test)

# print('BOW_cv_train:',cv_train_reviews.shape)
# print('BOW_cv_test:',cv_test_reviews.shape)
# print(y_pred)
# print(y_test)
# pipeline = Pipeline([('clf',model)]) 

# params = {'clf__C':( 1, 5, 10, ), 
#           'clf__gamma':(0.001, 0.01, 0.1)} 


# gridModel = GridSearchCV(pipeline, params) 

# gridModel.fit(x_train, y_train) 

# print("Best Score : ", gridModel.best_score_)

# best = gridModel.best_estimator_.get_params()
# #print(best)

# for k in sorted(params.keys()): 
#     print(k, best[k])
# model=SVC(C=best['clf__C'],gamma=best['clf__gamma'])
