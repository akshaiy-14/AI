import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u'I am Akshaiy, a data science enthusiast')
for token in doc:
    print(token.text, token.pos_, token.pos)
 
doc2 = nlp(u'I am Akshaiy, a data science enthusiast from the U.S.A. I am a novice at the moment.')
for sentence in doc2.sents:
    print(sentence)
    
type(doc2)
print(doc2[9:])
type(doc2[9:])

doc3 = nlp(u'Tesla is planning on building a factory in Hong Kong for $5 million. The U.S army has very less manpower.')
for entity in doc3.ents:
    print(entity, entity.label_)
    
for chunks in doc3.noun_chunks:
    print(chunks)
    
for token in doc3:
    print(token.text, token.lemma, token.lemma_) #Lemmatization is much better than Stemming.
    
from spacy import displacy
displacy.serve(doc2, style = 'ent')

print(nlp.Defaults.stop_words) 
len(nlp.Defaults.stop_words)           #326 stopwords in spacy.
nlp.Defaults.stop_words.add('btw')     #adding btw to the list
nlp.vocab['btw'].is_stop               #True
nlp.Defaults.stop_words.remove('not')
