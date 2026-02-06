import pandas as pd
import neattext.functions as nfx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load and preprocess dataset
df = pd.read_csv('UdemyCleanedTitle.csv')
df['Clean_title'] = df['course_title'].apply(nfx.remove_stopwords)
df['Clean_title'] = df['Clean_title'].apply(nfx.remove_special_characters)

# Vectorization
countvect = CountVectorizer()
cvmat = countvect.fit_transform(df['Clean_title'])

# Cosine Similarity Matrix
cosine_mat = cosine_similarity(cvmat)

# Save model components
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump((df, countvect, cosine_mat), f)

print("Model and data saved successfully.")
