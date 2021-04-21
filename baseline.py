from nltk.corpus import stopwords
import logging
import util_func as uf
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import nltk.data
from gensim.models import word2vec

from sklearn.feature_extraction.text import CountVectorizer

embed_fn = uf.embed_useT('./input/module/module_useT')
remove_stopwords = False
clean_up = False
sess = tf.Session()

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


# calculate features using USE
q_train_use = np.reshape(train.question.values,(len(train.question.values)))
r_train_use = np.reshape(train.response.values,(len(train.response.values)))

q_test_use = np.reshape(test.question.values,(len(test.question.values)))
r_test_use = np.reshape(test.response.values,(len(test.response.values)))

q_training_use = embed_fn(q_train_use)
r_training_use = embed_fn(r_train_use)

q_test_use = embed_fn(q_test_use)
r_test_use = embed_fn(r_test_use)

# calculate features using POS-tagging

nltk.download('universal_tagset')
q_training_pos = {}

for i in range(len(train)):
    text = nltk.word_tokenize(train['question'][i])
    q_training_pos[i] = []
    for j in range(len(text)):
        q_training_pos[i] =  q_training_pos[i] + [nltk.pos_tag(text)[j][0] + '_' + nltk.pos_tag(text)[j][1]]

q_sen = list(map(list, q_training_pos.values()))
r_training_pos = {}
for i in range(len(train)):
    text = nltk.word_tokenize(train['response'][i])
    r_training_pos[i] = []
    for j in range(len(text)):
        r_training_pos[i] =  r_training_pos[i] + [nltk.pos_tag(text)[j][0] + '_' + nltk.pos_tag(text)[j][1]]
r_sen = list(map(list, q_training_pos.values()))

num_features = 300
model_q = word2vec.Word2Vec(q_sen,\
                          min_count=5,
                         window=5,
                         size=num_features,
                         sample=6e-4, 
                         alpha=0.03, 
                         workers=2)

model_q.init_sims(replace=True)

model_r = word2vec.Word2Vec(r_sen,\
                          min_count=5,
                         window=5,
                         size=num_features,
                         sample=6e-4, 
                         alpha=0.03, 
                         workers=2)

model_r.init_sims(replace=True)

remove_stopwords = False
clean_up = False

q_vec_pos_tr = uf.getAvgFeatureVecs(q_sen, model_q, num_features)
r_vec_pos_tr = uf.getAvgFeatureVecs(r_sen, model_r, num_features)

q_test_pos={}
for i in range(len(test)):
    text = nltk.word_tokenize(test['question'][i])
    q_test_pos[i] = []
    for j in range(len(text)):
        q_test_pos[i] =  q_test_pos[i] + [nltk.pos_tag(text)[j][0] + '_' + nltk.pos_tag(text)[j][1]]

q_sen_ts = list(map(list, q_test_pos.values()))
r_test_pos = {}
for i in range(len(test)):
    text = nltk.word_tokenize(test['response'][i])
    r_test_pos[i] = []
    for j in range(len(text)):
        r_test_pos[i] =  r_test_pos[i] + [nltk.pos_tag(text)[j][0] + '_' + nltk.pos_tag(text)[j][1]]
r_sen_ts = list(map(list, q_test_pos.values()))

q_vec_pos_ts = uf.getAvgFeatureVecs(q_sen_ts, model_q, num_features)
r_vec_pos_ts = uf.getAvgFeatureVecs(r_sen_ts, model_r, num_features)

X_train = np.concatenate((q_vec_pos_tr,r_vec_pos_tr, q_training_use, r_training_use), axis=1)
X_test = np.concatenate((q_vec_pos_ts,r_vec_pos_ts, q_test_use, r_test_use), axis=1)

np.savetxt('base_train.txt', X_train)
np.savetxt('base_test.txt', X_test)
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