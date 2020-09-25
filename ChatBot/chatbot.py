#CHATBOT

#DATA PRE-PROCESSING

#Import libraries
import numpy as np
import tensorflow as tf
import re
import time

#Importing the dataset
lines = open('C:/Users/RAMALINGAM/Desktop/ChatBot/cornell movie-dialogs corpus/movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('C:/Users/RAMALINGAM/Desktop/ChatBot/cornell movie-dialogs corpus/movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#Creating a dictionary for lines
id2dialog = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
     id2dialog[_line[0]] = _line[4]
     
#Creating a list for conversations
convs = []
for conv in conversations:
    _conv = conv.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_conv.split(','))
     
#Matching the questions with its corresponding answers
questions = []
answers = []
for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2dialog[conv[i]])
        answers.append(id2dialog[conv[i+1]])
        
#Initial cleaning of the list
def clean_1(text):
    text = text.lower()
    text = re.sub(r"i'm" , "i am" , text)
    text = re.sub(r"\'ve" , " have" , text)
    text = re.sub(r"\'s" , " is" , text)
    text = re.sub(r"\'d" , " would" , text)
    text = re.sub(r"\'ll" , " will" , text)
    text = re.sub(r"\'re" , " are" , text)
    text = re.sub(r"won't" , "will not" , text)
    text = re.sub(r"can't" , "can not" , text)
    text = re.sub(r"[+=?|_:;()\-\"'~@#$%.,/<>]" , "" , text)
    return text


#Cleaning the questions
clean_q = []
for quest in questions:
    clean_q.append(clean_1(quest))
    
#Cleaning the answers
clean_a = []
for ans in answers:
    clean_a.append(clean_1(ans))
    
#Counting the word occurrences in clean_q
word_countq = {}
for qstn in clean_q:
    for word in qstn.split():
        if word not in word_countq:
            word_countq[word] = 1
        else:
            word_countq[word] += 1
            
#Counting the word occurrences in clean_a
word_counta = {}
for ans in clean_a:
    for word in ans.split():
        if word not in word_counta:
            word_counta[word] = 1
        else:
            word_counta[word] += 1
            
#Assigning a number to the most frequent words in the list
thresh = 20
count = 0
tokenizeq = {}
for word, freq in word_countq.items():
    if freq >= thresh:
        tokenizeq[word] = count;
        count += 1

count = 0
tokenizea = {}
for word, freq in word_counta.items():
    if freq >= thresh:
        tokenizea[word] = count;
        count += 1
        
tokens = ['<PAD>' , '<EOS>' , '<OUT>' , '<SOS>']
for token in tokens:
    tokenizeq[token] = len(tokenizeq)
 
for token in tokens:
    tokenizea[token] = len(tokenizea)
    
#Inversing the tokenizea dictionary
inv_tokenizea = { w_i:w for w,w_i in tokenizea.items()}

#Adding EOS token to the clean_a
for i in range(len(clean_a)):
    clean_a[i] += ' <EOS>'
    
#Translating all the clean_q and clean_a to their int tokens and replacing less frequent words with OUT
intofclean_q = []
for qstn in clean_q:
    intval = []
    for word in qstn.split():
        if word in tokenizeq:
           intval.append(tokenizeq[word])
        else:
            intval.append(tokenizeq['<OUT>'])
    intofclean_q.append(intval)       
    
intofclean_a = []
for ans in clean_a:
    intval = []
    for word in ans.split():
        if word in tokenizea:
           intval.append(tokenizea[word])
        else:
            intval.append(tokenizea['<OUT>'])
    intofclean_a.append(intval) 
    
#Sorting Q and A based on size
sorted_clean_q = []
sorted_clean_a = []
for length in range(1,25):
    for i in enumerate(intofclean_q):
        if len(i[1]) == length:
            sorted_clean_q.append(i[1])
            sorted_clean_a.append(intofclean_a[i[0]])  #Answer for that sorted question.
            



# BUILDING THE SEQ2SEQ MODEL
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None,None], name = 'input')   
    targets = tf.placeholder(tf.int32, [None,None], name = 'target')  
    lr = tf.placeholder(tf.float32, name = 'learning_rate')  
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  #This is the drop factor
    return inputs, targets, lr, keep_prob

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size,1], tokenizeq['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size,-1], [1,1])
    preprocessed_tgts = tf.concat([left_side,right_side], 1)
    return preprocessed_tgts
    


    

        
    

    
        

        
    
    
    

    
    
    
    
    
    
    
    
    
    