#BUILDING a BAG OF WORDS MODEL

import nltk
import re
import heapq
import numpy as np

#nltk.download('punkt')
paragraph = 'The Moon is a barren, rocky world without air and water. It has dark lava plain on its surface. The Moon is filled wit craters. It has no light of its own. It gets its light from the Sun. The Moo keeps changing its shape as it moves round the Earth. It spins on its axis in 27.3 days stars were named after the Edwin Aldrin were the first ones to set their foot on the Moon on 21 July 1969 They reached the Moon in their space craft named Apollo II. The sun is a huge ball of gases. It has a diameter of 1,392,000 km. It is so huge that it can hold millions of planets inside it. The Sun is mainly made up of hydrogen and helium gas. The surface of the Sun is known as the photosphere. The photosphere is surrounded by a thin layer of gas known as the chromospheres. Without the Sun, there would be no life on Earth. There would be no plants, no animals and no human beings. As, all the living things on Earth get their energy from the Sun for their survival. The Solar System consists of the Sun Moon and Planets. It also consists of comets, meteoroids and asteroids. The Sun is the largest member of the Solar System. In order of distance from the Sun, the planets are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto; the dwarf planet. The Sun is at the centre of the Solar System and the planets, asteroids, comets and meteoroids revolve around it.'
sentences = nltk.sent_tokenize(paragraph)

mod_sentences = []
for sentence in sentences:
    sentence = sentence.lower()
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\d', ' ', sentence)
    sentence = re.sub(r'\s+[b-z]\s+', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    mod_sentences.append(sentence)
 
word2count = {}
wordlist = []

for sents in mod_sentences:
    words = nltk.word_tokenize(sents)
    wordlist.append(words)
    
bow = [val for words in wordlist for val in words]
  
for word in bow:
    if word not in word2count.keys():
        word2count[word] = 1
    else:
        word2count[word] += 1

sorted_wordlist = sorted(word2count.items(), key = lambda x: x[1], reverse = True)
        
frequent = heapq.nlargest(100, word2count)

count_vec = []
for sent in mod_sentences:
    vector = []
    for word in frequent:
        if word in nltk.word_tokenize(sent):
            value = 1
            vector.append(value)
        else:
            value = 0
            vector.append(value)
    count_vec.append(vector)

count_vec = np.asarray(count_vec)
            



