import bs4 as bs
import urllib.request as urlreq
import re
import nltk
nltk.download('stopwords')

source = urlreq.urlopen('https://en.wikipedia.org/wiki/Heath_Ledger').read()
soup = bs.BeautifulSoup(source, 'lxml')  #parsing style

text = ''
for para in soup.find_all('p'):
    text += para.text

text = text.lower()
sentences = nltk.sent_tokenize(text)
mod_sentences = []

for sentence in sentences:
    sentence = re.sub(r'\[[0-9]\]', '', sentence)
    sentence = re.sub(r'\W', ' ', sentence)
    sentence = re.sub(r'\d+', ' ', sentence)
    sentence = re.sub(r'\s[a-zA-Z]\s', ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'^\W', '', sentence)
    mod_sentences.append(sentence)
    
stupid_text = re.sub(r'\[[0-9]\]', '', text)
stupid_text = re.sub(r'\W', ' ', stupid_text)
stupid_text = re.sub(r'\d+', ' ', stupid_text)
stupid_text = re.sub(r'\s[a-zA-Z]\s', ' ', stupid_text)
stupid_text = re.sub(r'\s+', ' ', stupid_text)
stupid_text = re.sub(r'^\W', '', stupid_text)
all_words = nltk.word_tokenize(stupid_text)
    
sw = nltk.corpus.stopwords.words('english')

word2count = {}
for sent in mod_sentences:
    for word in nltk.word_tokenize(sent):
        if word not in sw:
           if word in word2count.keys():
               word2count[word] += 1
           else:
               word2count[word] = 1
               
max_count =  max(word2count.values())              
for k in word2count.keys():
    word2count[k] = word2count[k]/max_count

sum_dict = {}    
for sentence in mod_sentences:
    sum = 0
    for word in nltk.word_tokenize(sentence):
        if word not in sw:
            sum += word2count[word]
    sum_dict[sentence] = sum
    
sorted_sum_dict = sorted(sum_dict.items(), key = lambda x: x[1], reverse = True)

print('------------------------------------------------------------------------------------------')
for i in range (0,3):
    print(sorted_sum_dict[i][0])
    print('\n')
        
        


    
