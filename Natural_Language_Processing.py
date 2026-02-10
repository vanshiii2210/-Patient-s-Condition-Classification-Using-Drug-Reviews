# Generated from: Natural_Language_Processing.ipynb
# Converted at: 2026-02-10T14:20:45.287Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# # **NATURAL LANGUAGE PROCESSING PROJECT**


# # **<631-  Patient's Condition Classification Using Drug Reviews>**


# **Business Objective:**
# 
# This is a sample dataset which consists of 161297 drug name, condition reviews and ratings from different patients and our goal is to examine how patients are feeling using the drugs their positive and negative experiences so that we can recommend him a suitable drug. By analyzing the reviews, we can understand the drug effectiveness and its side effects.
# The dataset provides patient reviews on specific drugs along with related conditions and a 10 star patient rating reflecting overall patient satisfaction.
# 
# So in this dataset, we can see many patients conditions but we will focus only on the below, classify the below conditions from the patients reviews
# 
# a. Depression
# 
# c. High Blood Pressure
# 
# d. Diabetes, Type 2
# 
# 
# **Attribute Information:**
# 
# 1. DrugName (categorical): name of drug
# 
# 2. condition (categorical): name of condition
# 
# 3. review (text): patient review
# 
# 4. rating (numerical): 10 star patient rating
# 
# 5. date (date): date of review entry
# 
# 6. usefulCount (numerical): number of users who found review useful
# 
# 


# # Packages


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import random

import nltk
nltk.download('all')#1

import spacy
nlp=spacy.load('en_core_web_sm')
import re
from nltk.stem import WordNetLemmatizer

!pip install textblob
from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
lab_enc=LabelEncoder()

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegressionCV
log_model=LogisticRegressionCV()

from sklearn.metrics import roc_auc_score,roc_curve

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')

from sklearn.metrics import classification_report,precision_recall_fscore_support,accuracy_score

from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

import xgboost as xgb

import lightgbm as lgb

import keras
import tensorflow as tf

from tensorflow.keras.layers import TextVectorization,Embedding

import wordcloud
from wordcloud import WordCloud,STOPWORDS

!pip install afinn
from afinn import Afinn

!pip install streamlit
import streamlit as st
import pickle

import getpass

!pip install streamlit pyngrok

from pyngrok import ngrok

# # Datasets


df=pd.read_excel('/content/drugsCom_raw.xlsx')

original=df.copy()

afin=Afinn()
afin=afin._dict
print(afin)

df.head()

# # Cleaning Data


rating_count=df.groupby('rating')['review'].agg(['count']).reset_index()
plt.bar(rating_count['rating'],rating_count['count'],color='grey',edgecolor='black')

plt.style.use('_classic_test_patch')

plt.title('Rating Count',size=20,color='red')
plt.xlabel('Ratings',color='blue')
plt.ylabel('Count of Ratings',color='blue')

for x in range(len(rating_count)):
  plt.text(x+1,rating_count['count'][x],rating_count['count'][x],ha='center',va='bottom')

df=df.drop(columns=['Unnamed: 0','date'])
df.head()

df.isnull().sum().sum()/len(df)*100

df.dropna(inplace=True)
df.isnull().sum()

df.duplicated().sum()

data=df[(df['condition']=='Depression') |(df['condition']=='High Blood Pressure') | (df['condition']=='Diabetes, Type 2')]
data.head()

data.info()

for x in data.columns:
  print(f'{x}\t {len(data[x].unique())}')
  print('-'*30)
  print('-'*30)

def cleaning(txt):
  rev=' '.join(re.findall('\w+',txt))
  rev_s=nlp(rev)
  clean=[x.lemma_ for x in rev_s if not x.is_stop
         and not x.is_punct and not x.is_digit
         and not x.is_bracket and not x.is_currency]
  return clean

data['review']=[cleaning(i) for i in data['review']]
data.head()#6

cleaned_data=data.copy()

stop=STOPWORDS.add('FullStack')
all_word=' '.join([' '.join(x) for x in data['review']])
word=WordCloud(background_color='black',max_words=100,stopwords=stop).generate(all_word)
def word_cloud(image):
  plt.figure(figsize=(10,10))
  plt.imshow(image)
  plt.axis('off')
  plt.savefig('wordcloud.jpeg')
word_cloud(word)

rate_count=data.groupby('rating')['review'].agg(['count']).reset_index()
plt.bar(rate_count['rating'],rate_count['count'],color='grey',edgecolor='black')

plt.style.use('_classic_test_patch')

plt.title('Rating Count',size=20,color='red')
plt.xlabel('Ratings',color='blue')
plt.ylabel('Count of Ratings',color='blue')

for x in range(len(rate_count)):
  plt.text(x+1,rate_count['count'][x],rate_count['count'][x],ha='center',va='bottom')

# # Sentiment Analysis


# ## Lexicon Approach


# ### Affin Dataset


def sent(txt:list=None):
  cnt=0
  if txt:
    for x in txt:
      cnt+=afin.get(x,0)
  return cnt

aff=data.copy()
aff['sent']=aff.review.apply(lambda x:sent(x))
aff.head()

aff.to_csv('test.csv')

aff['pos/neg']=aff.sent.apply(lambda x:'Pos' if x>0 else 'Neg' if x<0 else 'Neu')
aff.head()

# ### Text Blob


def blb(txt:list=None):
  sen=TextBlob(' '.join(txt)).sentiment.polarity
  return sen

blob=data.copy()
blob['sent']=blob.review.apply(lambda x:blb(x))
blob.head()

blob['pos/neg']=blob.sent.apply(lambda x:'Pos' if x>0 else 'Neg' if x<0 else 'Neu')
blob.head()

# ### Vader


def vad(txt):
  snt=SentimentIntensityAnalyzer()
  return snt.polarity_scores(' '.join(txt))['compound']

vader=data.copy()
vader['sent']=vader.review.apply(lambda x:vad(x))#1
vader.head()

vader['pos/neg']=vader.sent.apply(lambda x:'Pos' if x>0 else 'Neg' if x<0 else 'Neu')
vader.head()

# ## Visuals


plt.figure(figsize=(12,6))
for i,a,nm in zip([1, 2, 3],[aff, blob, vader],['Afinn','Text Blob','Vader']):
  plt.suptitle('Positive Negative Comparison',size=25,color='red')
  plt.subplot(1,3,i)
  a_pn=a.groupby('pos/neg')['review'].agg(['count']).reset_index().sort_values(by='pos/neg',ascending=False)
  plt.pie(a_pn['count'],labels=['Postive','Neutral','Negative'],colors=['lightgreen','silver','tomato'],wedgeprops={'linewidth':2,'edgecolor':'black'},textprops={'size':20},autopct='%.2f%%')
  plt.title(f'Sentiment Analysis using\n{nm}',size=20,color='red')
plt.tight_layout()

plt.figure(figsize=(12,10))
plt.suptitle('Sentiment Helpfullness',size=20,color='red')

plt.subplot(3,3,1)
plt.title('Afinn - Depression',size=10,color='blue')
x=aff[aff['condition']=='Depression']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,2)
plt.title('Afinn - High Blood Pressure',size=10,color='blue')
x=aff[aff['condition']=='High Blood Pressure']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,3)
plt.title('Afinn - Diabetes, Type 2',size=10,color='blue')
x=aff[aff['condition']=='Diabetes, Type 2']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,4)
plt.title('Blob - Depression',size=10,color='blue')
x=blob[blob['condition']=='Depression']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,5)
plt.title('Blob - High Blood Pressure',size=10,color='blue')
x=blob[blob['condition']=='High Blood Pressure']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,6)
plt.title('Blob - Diabetes, Type 2',size=10,color='blue')
x=blob[blob['condition']=='Diabetes, Type 2']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,7)
plt.title('Vader - Depression',size=10,color='blue')
x=vader[vader['condition']=='Depression']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,8)
plt.title('Vader - High Blood Pressure',size=10,color='blue')
x=vader[vader['condition']=='High Blood Pressure']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.subplot(3,3,9)
plt.title('Afinn - Diabetes, Type 2',size=10,color='blue')
x=vader[vader['condition']=='Diabetes, Type 2']
sns.scatterplot(data=x,y='sent',x='usefulCount',color='red')
plt.xlim(0,500)

plt.tight_layout()

def drug_sentiment(condition,drug):
  plt.figure(figsize=(12,6))
  plt.suptitle(f'{drug} for {condition}',size=20,color='red')
  for id,method,title in zip([1, 2, 3],[aff, blob, vader],['Afinn','Text Blob','Vader']):
    plt.subplot(1,3,id)
    df=method[(method['drugName']==drug) & (method['condition']==condition)]
    df1=df.groupby('pos/neg')['sent'].mean().round(2).reset_index()
    plt.bar(df1['pos/neg'],df1['sent'],color='grey',edgecolor='black')
    plt.title(f'{title} Method\nOverall {df['pos/neg'].mode()[0]} ({round(df['sent'].mean(),2)})',color='red',size=15)
    for x in range(len(df1)):
      plt.text(x,df1['sent'][x],df1['sent'][x],ha='center',va='bottom',size=20,color='blue')
  plt.tight_layout()

con=data['condition'].unique().tolist()
c=random.choice(con)
drg=data[data['condition']==c]
drg1=drg['drugName'].unique().tolist()
d=random.choice(drg1)

drug_sentiment(c,d)

# ## Recomend


def recomend(method,condition,n):
  name='Afinn' if method is aff else 'Text Blob' if method is blob else 'Vader'
  print(f'Sentiment Analysis Summary for {condition}\n\tUsing {name} Method')
  print('-'*30)
  print('-'*30)
  method1=method[method['condition']==condition]
  x=method1.groupby('drugName')['sent'].agg(['mean']).sort_values(by='mean',ascending=False)[:n]
  print(f'Best Recomendation For\n {condition}')
  print('-'*30)
  print(x)
  y=method1.groupby('drugName')['sent'].agg(['mean']).sort_values(by='mean',ascending=False)[-n:]
  print('-'*30)
  print('-'*30)
  print(f'Worst Recomendation For\n {condition}')
  print('-'*30)
  print(y)

method=random.choice([aff,blob,vader])
condition=np.random.choice(['Depression','Diabetes, Type 2','High Blood Pressure'])

recomend(method,condition,3)

# # Evaluation


# ## Data Preparation


original.drop(columns=['Unnamed: 0','date'],inplace=True)
df=original[(original['condition']=='Depression')|(original['condition']=='High Blood Pressure')|(original['condition']=='Diabetes, Type 2')]
df.head()

df['t_review']=[cleaning(i) for i in df['review']]
df['sent']=df.t_review.apply(lambda x:sent(x))#5

df.drop(columns=['t_review'],inplace=True)
df.head()

df['pos/neg']=df.sent.apply(lambda x:'Pos' if x>0 else 'Neg' if x<0 else 'Neu')
df.head()

df.head()

feature=df['review']
target=df['pos/neg']
print(feature.shape,target.shape)

target=pd.DataFrame(lab_enc.fit_transform(target))
target.head()

feature=tfidf.fit_transform(feature)
feature.shape

x_train,x_test,y_train,y_test=train_test_split(feature,target,train_size=0.80,random_state=1000)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

# # Models


# ### Logistic Regression


log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
print(classification_report(y_test,y_pred))#1

log_acc=accuracy_score(y_test,y_pred)

y_prob=log_model.predict_proba(x_test)
plt.figure(figsize=(15,5))
plt.suptitle('ROC Curves for all Classes',size=20,color='red')
for i,p in zip([0,1,2],['Negative','Neutral','Positive']):
  plt.subplot(1,3,i+1)
  lab=(y_test==i).astype(int)
  asc=roc_auc_score(lab,y_prob[:,i])
  fpr,tpr,_=roc_curve(lab,y_prob[:,i])
  plt.plot(fpr,tpr,label=f'Class {i} AUC = {asc:.3f}',lw=3,color='green')
  plt.plot([0,1],linestyle='--',color='red')
  plt.title(f'The ROC AUC Curve for {p}',color='blue',size=15)
  plt.legend()

# ### Support Vector Machine


svc=SVC(C=1,kernel='linear')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(classification_report(y_test,y_pred))

svc=SVC(C=1,kernel='sigmoid')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(classification_report(y_test,y_pred))#1

svc=SVC(C=1,kernel='poly')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(classification_report(y_test,y_pred))#2

svc=SVC(C=1,kernel='rbf')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(classification_report(y_test,y_pred))#3

svc_acc=accuracy_score(y_test,y_pred)

pca=PCA(n_components=2)
x_train_new=pca.fit_transform(x_train)
x_test_new=pca.transform(x_test)
svc=SVC(C=1,kernel='rbf')
svc.fit(x_train_new,y_train)
y_pred=svc.predict(x_test_new)
x_min,x_max,y_min,y_max=x_train_new[:,0].min()-0.1,x_train_new[:,0].max()+0.1,x_train_new[:,1].min()-0.1,x_train_new[:,1].max()+0.1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),np.arange(y_min,y_max,0.02))
dt=np.c_[np.ravel(xx),np.ravel(yy)]
z=svc.predict(dt)
z1=z.reshape(xx.shape)
plt.contourf(xx,yy,z1)
plt.scatter(x_train_new[:,0],x_train_new[:,1],c=y_train.values.ravel())

# ### Decision Tree


dt=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3)
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(classification_report(y_test,y_pred))

dt_acc=accuracy_score(y_test,y_pred)

plot_tree(dt,filled=True)

# ### Random Forest


rf=RandomForestClassifier(n_estimators=100,criterion='gini',max_features=1,max_depth=5,bootstrap=True)
rf.fit(x_train,y_train)
y_pred=rf.predict(x_test)
print(classification_report(y_test,y_pred))

rf_acc=accuracy_score(y_test,y_pred)

# ### Ada Boost


ada=AdaBoostClassifier(LogisticRegressionCV(),n_estimators=100)
ada.fit(x_train,y_train)
y_pred=ada.predict(x_test)
print(classification_report(y_test,y_pred))#13

ada_acc=accuracy_score(y_test,y_pred)

# ### Gradient Boost


grad=GradientBoostingClassifier(n_estimators=100,max_features=1)
grad.fit(x_train,y_train)
y_pred=grad.predict(x_test)
print(classification_report(y_test,y_pred))

grad_acc=accuracy_score(y_test,y_pred)

# ### Xtreme Gradient Boost


xg=xgb.XGBClassifier(n_estimatores=100,max_features=1)
xg.fit(x_train,y_train)
y_pred=xg.predict(x_test)
print(classification_report(y_test,y_pred))#1

xg_acc=accuracy_score(y_test,y_pred)

# ### Light Gradient Boost


lg=lgb.LGBMClassifier(n_estimators=100,max_features=1)
lg.fit(x_train,y_train)
y_pred=lg.predict(x_test)
print(classification_report(y_test,y_pred))#1

lg_acc=accuracy_score(y_test,y_pred)

# ### Bagging


bag=BaggingClassifier(LogisticRegressionCV(),n_estimators=100,bootstrap=True,bootstrap_features=True)
bag.fit(x_train,y_train)
y_pred=bag.predict(x_test)
print(classification_report(y_test,y_pred))#80

bag_acc=accuracy_score(y_test,y_pred)

# ### Artificial Neural Network


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

ann=tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=128,activation='relu',input_shape=[14165]))
ann.add(tf.keras.layers.Dense(units=64,activation='relu'))
ann.add(tf.keras.layers.Dense(units=32,activation='relu'))
ann.add(tf.keras.layers.Dense(units=3,activation='softmax'))
ann.summary()

ann.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

hist=ann.fit(x_train,y_train,epochs=20,validation_data=(x_test,y_test),batch_size=1000)

y_pred=ann.predict(x_test)

ann_loss,ann_acc=ann.evaluate(x_test,y_test)

plt.figure(figsize=(10,5))
plt.suptitle('Final Evaluation',size=15,color='red')

plt.subplot(1,2,1)
plt.title('Accuracy Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,21),hist.history['val_accuracy'],label='Testing Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['loss'],label='Training Loss')
plt.plot(range(1,21),hist.history['val_loss'],label='Testing Loss')
plt.legend()

# ### Reccurent Neural Network


feature=df['review']
target=df['pos/neg']
target=pd.DataFrame(lab_enc.fit_transform(target))

x_train,x_test,y_train,y_test=train_test_split(feature,target,train_size=0.80,random_state=1000)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

vect=TextVectorization(max_tokens=10000,output_mode='int',
output_sequence_length=10)
vect.adapt(x_train)

emb=Embedding(input_dim=14165,output_dim=128,input_length=10)

rnn=tf.keras.Sequential()
rnn.add(tf.keras.Input(shape=(1,),dtype=tf.string))
rnn.add(vect)
rnn.add(emb)
rnn.add(tf.keras.layers.SimpleRNN(units=128,activation='relu',return_sequences=True))
rnn.add(tf.keras.layers.SimpleRNN(units=64,return_sequences=True,activation='relu'))
rnn.add(tf.keras.layers.SimpleRNN(units=32,activation='relu'))
rnn.add(tf.keras.layers.Dense(units=3,activation='softmax'))
rnn.summary()

rnn.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

hist=rnn.fit(x_train.values,y_train.values,epochs=20,validation_data=(x_test.values,y_test.values),batch_size=1000)

y_pred=rnn.predict(x_test.values)

rnn_loss,rnn_acc=rnn.evaluate(x_test.values,y_test.values)

plt.figure(figsize=(10,5))
plt.suptitle('Final Evaluation',size=15,color='red')

plt.subplot(1,2,1)
plt.title('Accuracy Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,21),hist.history['val_accuracy'],label='Testing Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['loss'],label='Training Loss')
plt.plot(range(1,21),hist.history['val_loss'],label='Testing Loss')
plt.legend()

# ### Long Short Term Memory


lstm=tf.keras.Sequential()
lstm.add(tf.keras.Input(shape=(1,),dtype=tf.string))
lstm.add(vect)
lstm.add(emb)
lstm.add(tf.keras.layers.LSTM(units=128,activation='relu',return_sequences=True))
lstm.add(tf.keras.layers.LSTM(units=64,return_sequences=True,activation='relu'))
lstm.add(tf.keras.layers.LSTM(units=32,activation='relu'))
lstm.add(tf.keras.layers.Dense(units=3,activation='softmax'))
lstm.summary()

lstm.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

hist=lstm.fit(x_train.values,y_train.values,epochs=20,validation_data=(x_test.values,y_test.values),batch_size=1000)

y_pred=lstm.predict(x_test.values)

lstm_loss,lstm_acc=lstm.evaluate(x_test.values,y_test.values)

plt.figure(figsize=(10,5))
plt.suptitle('Final Evaluation',size=15,color='red')

plt.subplot(1,2,1)
plt.title('Accuracy Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,21),hist.history['val_accuracy'],label='Testing Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['loss'],label='Training Loss')
plt.plot(range(1,21),hist.history['val_loss'],label='Testing Loss')
plt.legend()

# ### Gated Reccured Unit


gru=tf.keras.Sequential()
gru.add(tf.keras.Input(shape=(1,),dtype=tf.string))
gru.add(vect)
gru.add(emb)
gru.add(tf.keras.layers.GRU(units=128,activation='relu',return_sequences=True))
gru.add(tf.keras.layers.GRU(units=64,return_sequences=True,activation='relu'))
gru.add(tf.keras.layers.GRU(units=32,activation='relu'))
gru.add(tf.keras.layers.Dense(units=3,activation='softmax'))
gru.summary()

gru.compile(optimizer=tf.optimizers.Adam(),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

hist=gru.fit(x_train.values,y_train.values,epochs=20,validation_data=(x_test.values,y_test.values),batch_size=1000)

y_pred=gru.predict(x_test.values)

gru_loss,gru_acc=gru.evaluate(x_test.values,y_test.values)

plt.figure(figsize=(10,5))
plt.suptitle('Final Evaluation',size=15,color='red')

plt.subplot(1,2,1)
plt.title('Accuracy Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['accuracy'],label='Training Accuracy')
plt.plot(range(1,21),hist.history['val_accuracy'],label='Testing Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.title('Loss Graph',size=10,color='blue')
plt.plot(range(1,21),hist.history['loss'],label='Training Loss')
plt.plot(range(1,21),hist.history['val_loss'],label='Testing Loss')
plt.legend()

# ## Comparison


comp={'Model':['Logistic Regression','Support Vector Machine','Decision Tree Calssifier','Random Forest Classifier','Ada Boost Classifier','Gradient Boost Classifier','Xtreme Gradient Boost Classifier','Light Gradient Boost Classifier','Bagging Classifier','Artificial Neural Network','Recurrent Neural Network','Long Short Term Memory','Gated Recurrent Unit'],'Accuracy':[round(log_acc*100,2),round(svc_acc*100,2),round(dt_acc*100,2),round(rf_acc*100,2),round(ada_acc*100,2),round(grad_acc*100,2),round(xg_acc*100,2),round(lg_acc*100,2),round(bag_acc*100,2),round(ann_acc*100,2),round(rnn_acc*100,2),round(lstm_acc*100,2),round(gru_acc*100,2)]}
comparison=pd.DataFrame(data=comp)
comparison.sort_values(by='Accuracy',ascending=False)

print(log_acc,svc_acc,dt_acc,rf_acc,ada_acc,grad_acc,xg_acc,lg_acc,bag_acc,ann_acc,rnn_acc,lstm_acc,gru_acc)
#0.9074937253495877 0.8924345643599857 0.5905342416636787 0.49264969523126567 0.8866977411258515 0.5080674076730011 0.8669774112585156 0.8809609178917175 0.8859806382215848 0.8917174339294434 0.7296522259712219 0.6471853852272034 0.6697741150856018

# # Deployment


# ## Condition Modeling


feat=df['review']
targ=df['condition']

le=LabelEncoder()
targ=pd.DataFrame(le.fit_transform(targ))
targ.head()

feat=tfidf.fit_transform(feat)
feat.shape

lm=LogisticRegressionCV()
lm.fit(feat,targ)

lab='labeling.pkl'
pickle.dump(le,open(lab,'wb'))
lab_m='lab_model.pkl'
pickle.dump(lm,open(lab_m,'wb'))

file='log_model.pkl'
pickle.dump(log_model,open(file,'wb'))
vect='tfidf.pkl'
pickle.dump(tfidf,open(vect,'wb'))

ngrok_key=getpass.getpass('Enter ngrok key => ')

# ## Final Deployment


%%writefile deploy.py

import streamlit as st
import pickle
import pandas as pd

model=pickle.load(open('log_model.pkl','rb'))
tfi=pickle.load(open('tfidf.pkl','rb'))
labe=pickle.load(open('labeling.pkl','rb'))
lab_mo=pickle.load(open('lab_model.pkl','rb'))

st.set_page_config(layout="wide")
st.title('Sentiment Analysis')
col_l,col_r=st.columns([3,1])

with col_l:
  input=st.text_area('Enter your Text here......',height=500)

label={0:'Negative',1:'Neutral',2:'Positive'}

with col_r:
  if st.button('Predict Sentiment'):
    vector=tfi.transform([input])
    pred=model.predict(vector)[0]
    pred_prob=model.predict_proba(vector)
    c_pred=lab_mo.predict(vector)[0]
    c_label=labe.inverse_transform([c_pred])[0]
    sent_label=label[pred]
    neg,neu,pos=pred_prob[0]

    if pred==2:
      st.write(f'Positive Sentiment')
      st.write(f'Most likely for {c_label}')
    if pred==1:
      st.write(f'Neutral Sentiment')
      st.write(f'Most likely for {c_label}')
    if pred==0:
      st.write(f'Negative Sentiment')
      st.write(f'Most likely for {c_label}')

    tab={'Sentiment':['Positive','Neutral','Negative'],'Probability':[round(pos*100,2),round(neu*100,2),round(neg*100,2)]}
    st.table(pd.DataFrame(tab))

port=8501
ngrok.set_auth_token(ngrok_key)
ngrok.connect(port).public_url

!rm -f logs.txt && streamlit run deploy.py >/content/logs.txt 2>&1

# Some sample reviews

# I have been taking Jardiance for just over a year - I have NOT experienced any side effects whatsoever - MORE importantly this drug has reduced my blood sugar levels to an average of 5 and has maintained this level, quite incredible as nothing else has worked previously - I have named Jardiance as the &quot;diabetic wonder drug&quot; although my Doctor keeps reminding me there is no such thing as a wonder drug - I disagree this is MY wonder drug - thank you Jardiance you have changed my life completely astonishing result.

# It has worked so far for me and I would recommend it to anyone suffering with depression.

# Using this for parasomnia, bph and bp. Kind of silver bullet. But my new doc tried to give me the brand name... Is it an extended release or just much more expensive? I only take it at night and dizziness hasnt been an issue. Seems to work as well as other bph meds that I have tried, my bp has also been reasonable. Seems to make sleepwalking more infrequent which is why I tried it.

# Started this last night. I have zero appetite. How is this possible? I love food, but I don&039;t really feel like eating. All I&039;ve had today is some grapes. I have a sandwich in my lunchbox, but I don&039;t have any appetite. I haven&039;t had any of the side effects so far. I started with the 1.2 dose. The &quot;pen&quot; is really easy to use compared to my intramuscular testosterone. However, this drug is ridiculously expensive. Even with good insurance, I would have to pay almost $400 a MONTH!!! Fortunately, another serious illness pushed me over my out-of-pocket annual. So, I filled the script for $0. Due to politics I may have no insurance next year. Anyway, so far, so good with Victoza.

# Worked beautifully in reducing my BP from 190/104 down to normotensive readings of 120/72. However, as with the other meds that were tried, I had intolerable side effects like GERD, abdominal bloating and malaise. My internal med PA-C changed me to Losartan (since the ARB class of drug relieved my hypertension) but this time without the HCT diuretic, and the results have been wonderful. Same BP effect without the high cost (luckily we trialed the Benicar HCT with professional free samples!), side effects, or need to excessively hydrate to balance the diuretic.

# This medication is amazing! After 3 days of being extremely sick, I started to feel amazing, I am now 1 month into it and am so happy all the time and have no depressive thoughts at all. It kind of blocks out any sad thoughts. Works perfectly for me.

# I've been on Metformin for six months. On the positive side, my A1C dropped from 8.4 to 6.2, which my doctor is thrilled about. However, the 'Metformin runs' are no joke. I have constant bloating and stomach cramps that make it hard to leave the house. It's a love-hate relationship; the drug does exactly what it's supposed to do for my blood sugar, but the GI side effects are making me consider switching to something else.

# Invokana was a disaster for me!  First on Invokamet, also bad!  Had GI problems with Invokamet.  Dr, switched me to Invokana.  Worse GI problems.  Developed anal fissures from the diarrhea and constipation.  Now have yeast infection.  Back on Metformin and exercise.  Numbers are better!

# I work nights so I took my 2nd dose of the day at 1am. By the time I saw the doctor at 8am, my blood pressure was 160/110. It doesn&#039;t work for me. I get no benefit, and most of the side effects.

# Venlafaxine made me, dizzy, gave me nausea and diarrhea. My anxiety increased because of the severe stomach distress this medication caused me.I felt all around lousy. Took medication for 2 months hoping the symptoms would subside. The problems just got worse.


# The first two weeks were absolute hell. I had increased anxiety, cold sweats, and couldn't sleep at all. I almost quit, but my therapist urged me to stick it out. Around week five, the clouds finally lifted. I don't feel 'numb,' I just feel like myself again. If you're just starting, please give it at least a month before you give up. It takes time for your brain to adjust.

# Absolute disaster. Took it at 10 PM, and by midnight I was 'sleep-eating' an entire box of cereal in the kitchen. My husband found me trying to fold the laundry while I was still out cold. I woke up feeling like a zombie with a massive headache. Never again. I'd rather stay awake than deal with the scary parasomnia this caused.

# It's okay. Doesn't work as well as the stronger stuff, but it's better than taking nothing at all. I haven't noticed any weird symptoms yet. I'll keep taking it for another week and see if anything changes.

# It's okay. Doesn't work as well as the stronger stuff, but it's better than taking nothing at all. I haven't noticed any weird symptoms yet. I'll keep taking it for another week and see if anything changes.