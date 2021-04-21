from nltk.corpus import stopwords
import logging
import util_func as uf
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk.data
from gensim.models import word2vec
from collections import Counter
remove_stopwords = True
clean_up = False
import texthero as hero
from sklearn.feature_extraction.text import CountVectorizer
# read the input data
train = pd.read_csv("./data/p2_train.csv")
test = pd.read_csv("./data/p2_test.csv")
y_train = train.type.values
y_test = test.type.values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

base_train = np.loadtxt('base_train.txt')
base_test = np.loadtxt('base_test.txt')

from textblob import TextBlob
# Get the polarity score using below function
polar_tr, polar_ts = [], []
for i in range(len(train)):
    polar_tr = polar_tr + [uf.get_textBlob_score(train['question'][i]) - uf.get_textBlob_score(train['response'][i])]

for i in range(len(test)):
    polar_ts = polar_ts + [uf.get_textBlob_score(test['question'][i]) - uf.get_textBlob_score(test['response'][i])]


train['r_ws'] = hero.remove_stopwords(train['response'])
train['r_ws'] = hero.clean(train['r_ws'])
# train['q_ws'] = hero.remove_stopwords(train['question'])
# train['q_ws'] = hero.clean(train['q_ws'])
agr, i_agr = [], 10
ans, i_ans = [], 20
att, i_att = [], 20
irr, i_irr= [], 20
agreed = Counter(" ".join(train[train['type']=='agreed']["r_ws"]).split()).most_common(i_agr)
answered = Counter(" ".join(train[train['type']=='answered']["r_ws"]).split()).most_common(i_ans)
attacked = Counter(" ".join(train[train['type']=='attacked']["r_ws"]).split()).most_common(i_att)
irrelevant = Counter(" ".join(train[train['type']=='irrelevant']["r_ws"]).split()).most_common(i_irr)

for i in range(i_agr):
    agr.append(agreed[i][0])
for i in range(i_ans):    
    ans.append(answered[i][0])
for i in range(i_att):
    att.append(attacked[i][0])
for i in range(i_irr):
    irr.append(irrelevant[i][0])

bag = agr+ans+att+irr
bag = set(bag)
vec2_train = np.zeros((len(train),len(bag)+1))

for i in range(len(train)):
    vec2_train[i,len(bag)]=polar_tr[i]
    txt = train['response'][i]
    k=0
    for j in bag:
        if j in txt:
            vec2_train[i,k]=1
        k+=1
vec2_test = np.zeros((len(test),len(bag)+1))
for i in range(len(test)):
    vec2_test[i,len(bag)]=polar_ts[i]
    txt = test['response'][i]
    k=0
    for j in bag:
        if j in txt:
            vec2_test[i,k]=1
        k+=1

X_train = np.concatenate((vec2_train, base_train), axis=1)
X_test = np.concatenate((vec2_test,base_test), axis=1)

from sklearn import svm

clf = svm.SVC()
clf.fit(X_train, y_train)  
predicted = clf.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("svm accuracy", accuracy_score(y_test, predicted))
print("svm f1_score", f1_score(y_test, predicted,average='micro'))
print("svm precision_score", precision_score(y_test, predicted,average='micro'))
print("svm recall_score", recall_score(y_test, predicted,average='micro'))

tf.keras.backend.clear_session()
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(32, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)
predicted = model.predict_classes(X_test)
print(classification_report(y_test, predicted))