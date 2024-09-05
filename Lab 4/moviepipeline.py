import sklearn
from sklearn.datasets import load_files

moviedir = r'C:\Users\lovis\Desktop\TNM108\Lab 4\movie_reviews'

# loading all files and set categories to pos and neg
movie = load_files(moviedir, shuffle=True)
categories = movie.target_names

# Split movie set into train and test
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, 
                                                          test_size = 0.20, random_state = 12)

# Build pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', MultinomialNB()),
])

text_clf.fit(docs_train, y_train)

'''Evaluate performance on test set'''
import numpy as np
predicted = text_clf.predict(docs_test)

print("multinomialBC accuracy ",np.mean(predicted == y_test))

# training SVM classifier
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])
text_clf.fit(docs_train, y_train)
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == y_test))

from sklearn import metrics
print(metrics.classification_report(y_test, predicted,
target_names=categories))

print(metrics.confusion_matrix(y_test, predicted))

from sklearn.model_selection import GridSearchCV
parameters = {
'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(docs_train, y_train)

#print(twenty_train.target_names[gs_clf.predict(['God is love'])[0]])

print(gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))