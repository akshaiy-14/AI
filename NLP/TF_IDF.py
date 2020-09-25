#BUILDING a TF_IDF MODEL

import nltk
import re
import heapq
import numpy as np
import math

# nltk.download('stopwords')
# from nltk.corpus import stopwords
# sw = stopwords.words('english')
# nltk.download('punkt')
paragraph = 'The Moon is a barren, rocky world without air and water. It has dark lava plain on its surface. The Moon is filled wit craters. It has no light of its own. It gets its light from the Sun. The Moo keeps changing its shape as it moves round the Earth. It spins on its axis in 27.3 days stars were named after the Edwin Aldrin were the first ones to set their foot on the Moon on 21 July 1969 They reached the Moon in their space craft named Apollo II. The sun is a huge ball of gases. It has a diameter of 1,392,000 km. It is so huge that it can hold millions of planets inside it. The Sun is mainly made up of hydrogen and helium gas. The surface of the Sun is known as the photosphere. The photosphere is surrounded by a thin layer of gas known as the chromospheres. Without the Sun, there would be no life on Earth. There would be no plants, no animals and no human beings. As, all the living things on Earth get their energy from the Sun for their survival. The Solar System consists of the Sun Moon and Planets. It also consists of comets, meteoroids and asteroids. The Sun is the largest member of the Solar System. In order of distance from the Sun, the planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto; the dwarf planet. The Sun is at the centre of the Solar System and the planets, asteroids, comets and meteoroids revolve around it.'
sentences = nltk.sent_tokenize(paragraph)

mod_sentences = []
for sentence in sentences:
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\d', ' ', sentence)
    sentence = re.sub(r'\s+[b-z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'[\']', '', sentence)
    sentence = sentence.lower()
    mod_sentences.append(sentence)
 
word2count = {}
wordlist = []

#GETTING ALL THE WORDS FROM EVERY SENTENCE
for sents in mod_sentences:
    words = nltk.word_tokenize(sents)
    wordlist.append(words)
bow = [val for words in wordlist for val in words]
  

#COUNTING THE FREQUENCY OF WORDS
for word in bow:
    if word not in word2count.keys():
        word2count[word] = 1
    else:
        word2count[word] += 1
                
frequent = heapq.nlargest(100, word2count)


idf = {}
for word in frequent:
    n_doc = 0
    for sent in mod_sentences:
        if word in nltk.word_tokenize(sent):
            n_doc += 1
        else:
            n_doc = n_doc
    idf[word] = math.log(len(mod_sentences)/n_doc)
    
    
idf_matrix = []
for k,v in idf.items():
    val = [k,v]
    idf_matrix.append(val)

idf_matrix = np.asarray(idf_matrix)

tf = {}
for word in frequent:
    tf_list = []
    for sentence in mod_sentences:
        cnt = 0
        for i in nltk.word_tokenize(sentence):
            if word == i:
                cnt =+ 1
            
        tf_part = cnt/len(nltk.word_tokenize(sentence))
        tf_list.append(tf_part)
    tf[word] = tf_list


tf_idf = {}
tf_idf_list = []
for word in tf.keys():
    tf_list = []
    for x in tf[word]:
        value = idf[word] * x
        tf_list.append(value)
    
    tf_idf_list.append(tf_list)
    tf_idf[word] = tf_list

tf_idf_list = np.transpose(np.asarray(tf_idf_list))
            
#DIRECT FORMULAE
from sklearn.feature_extraction.text import CountVectorizer as CVR
cvr = CVR(max_features = 100)
vec_cvr = cvr.fit_transform(mod_sentences).toarray()


from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(max_features = 100)
tfidf_vec = tfidf.fit_transform(mod_sentences).toarray()
          

