import re
import nltk
from sklearn.datasets import load_files 
from nltk.corpus import stopwords
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')

reviews = load_files('C:/Users/RAMALINGAM/Desktop/Learn/NLP/txt_sentoken/')
X, y = reviews.data, reviews.target

corpus = []
for i in range (0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\d+', ' ', review)
    review = re.sub(r'\s[a-zA-Z]\s', ' ', review)
    review = re.sub(r'^[a-zA-Z]\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
tfidf = TFIDF(max_features = 2000, min_df = 5, max_df = 0.5, stop_words = sw)
X = tfidf.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty = 'l2', C = 3.5)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train, y_train)
y_pred2 = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import accuracy_score as acc
cm1 = CM(y_test, y_pred)
print('Accuracy of logistic regression is ' + str(100*acc(y_test, y_pred)))
cm2 = CM(y_test, y_pred2)
print('Accuracy of RFC is ' + str(100*acc(y_test, y_pred2)))







