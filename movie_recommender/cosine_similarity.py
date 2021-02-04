
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


cv = CountVectorizer()

text = ['London Paris London', 'Paris Paris London']

count_matrix = cv.fit_transform(text)
similarity_scores = cosine_similarity(count_matrix)

#print(count_matrix.toarray())
print(similarity_scores)