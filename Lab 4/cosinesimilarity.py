d1 = "The sky is blue."
d2 = "The sun is bright."
d3 = "The sun in the sky is bright."
d4 = "We can see the shining sun, the bright sun."
Z = (d1,d2,d3,d4)

# We have 4 documents (rows) and 11 columns (unique words)
import math
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(Z)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
cos_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix)
print(cos_similarity)

# Take the cos similarity of the third document (cos similarity=0.52)
angle_in_radians = math.acos(cos_similarity[0, 2])
print(math.degrees(angle_in_radians))