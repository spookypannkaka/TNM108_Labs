from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names

#print(data.target_names)

my_categories = ['rec.sport.baseball','rec.motorcycles','sci.space','comp.graphics']
train = fetch_20newsgroups(subset='train', categories=my_categories)
test = fetch_20newsgroups(subset='test', categories=my_categories)

#print(len(train.data))
#print(len(test.data))
#print(train.data[9])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train_counts=cv.fit_transform(train.data)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf=tfidf_transformer.fit_transform(X_train_counts)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train_tfidf, train.target)

docs_new = ['Pierangelo is a really good baseball player','Maria rides her motorcycle', 'OpenGL on the GPU is fast', 'Pierangelo rides his motorcycle and goes to play football since he is a good football player too.', 'Jupiter is my favorite planet']
X_new_counts = cv.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = model.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, train.target_names[category]))