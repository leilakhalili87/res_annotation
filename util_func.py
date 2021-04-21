# Therefore, the performance is similar to the "bag of words" model.

from textblob import TextBlob
# Importing libraries
import numpy as np
import pandas as pd
# BeautifulSoup is used to remove html tags from the text
from bs4 import BeautifulSoup 
import re # For regular expressions
import nltk.data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# Stopwords can be useful to undersand the semantics of the sentence.
# Therefore stopwords are not removed while creating the word2vec model.
# But they will be removed  while averaging feature vectors.
from nltk.corpus import stopwords
from tqdm import tqdm
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize
import nltk
import tensorflow_hub as hub

def import_xlsx(file_path, header_name):
    init_file = pd.ExcelFile(file_path)
    data = {sheet_name: init_file.parse(sheet_name) for sheet_name in init_file.sheet_names}[header_name]
    return data

def review_sentences(review, tokenizer, remove_stopwords, clean_up):
    # This function splits a review into sentences
    # 1. Using nltk tokenizer
    raw_sentences = tokenizer.tokenize(review.strip(), realign_boundaries=True)
    sentences = []
    # 2. Loop for each sentence
    for raw_sentence in raw_sentences:
        if len(raw_sentence)>0:
            sentences.append(review_wordlist(raw_sentence,\
                                            remove_stopwords, clean_up))

    # This returns the list of lists
    return sentences


def review_wordlist(review, remove_stopwords, clean_up):
    # This function converts a text to a sequence of words.
    # Using punkt tokenizer for better splitting of a paragraph into sentences.
    if clean_up:
    # 1. Removing html tags
        review_text = BeautifulSoup(review).get_text()
    # 2. Removing non-letter.
        review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 3. Converting to lower case and splitting
        words = review_text.lower().split()
    else:
        words = review.split()

    # 4. Optionally remove stopwords
    if remove_stopwords:
        stops = set(stopwords.words("english"))     
        words = [w for w in words if not w in stops]
    
    return(words)

def featureVecMethod(words, model, num_features):
    # Function to average all word vectors in a paragraph
    # Pre-initialising empty numpy array for speed
    featureVec = np.zeros(num_features,dtype="float32")
    nwords = 0
    
    #Converting Index2Word which is a list to a set for better speed in the execution.
    index2word_set = set(model.wv.index2word)
    
    for word in  words:
        if word in index2word_set:
            nwords = nwords + 1
            featureVec = np.add(featureVec,model[word])
    
    # Dividing the result by number of words to get average
    featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    # Function for calculating the average feature vector
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    for review in reviews:
        # Printing a status message every 1000th review
        if counter%1000 == 0:
            print("Review %d of %d"%(counter,len(reviews)))
            
        reviewFeatureVecs[counter] = featureVecMethod(review, model, num_features)
        counter = counter+1
        
    return reviewFeatureVecs


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    vec = tfidf_vectorizer.fit_transform(data)
    return vec, tfidf_vectorizer


def create_corpus_new(df):
    corpus=[]
    for tweet in tqdm(df['sentence']):
        words=[word.lower() for word in word_tokenize(tweet)]
        corpus.append(words)
    return corpus  

def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
pos_code_map={'CC':'A','CD':'B','DT':'C','EX':'D','FW':'E','IN':'F','JJ':'G','JJR':'H','JJS':'I','LS':'J','MD':'K','NN':'L','NNS':'M',
'NNP':'N','NNPS':'O','PDT':'P','POS':'Q','PRP':'R','PRP$':'S','RB':'T','RBR':'U','RBS':'V','RP':'W','SYM':'X','TO':'Y','UH':'Z',
'VB':'1','VBD':'2','VBG':'3','VBN':'4','VBP':'5','VBZ':'6','WDT':'7','WP':'8','WP$':'9','WRB':'@'}

def pos_tags(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text_processed=tokenizer.tokenize(text)
    return "".join(convert(tag) for (word, tag) in nltk.pos_tag(text_processed))


def text_pos_inv_convert(text):
    return "-".join(inv_convert(c.upper()) for c in text)

def convert(tag):
    try:
        code=pos_code_map[tag]
    except:
        code='?'
    return code


def inv_convert(code):
    try:
        tag=code_pos_map[code]
    except:
        tag='?'
    return tag

def get_textBlob_score(sent):
    # This polarity score is between -1 to 1
    polarity = TextBlob(sent).sentiment.polarity
    return polarity