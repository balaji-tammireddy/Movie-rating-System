import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

df = pd.read_csv('stemming.csv')

if df['Review'].dtype == object and isinstance(df['Review'].iloc[0], list):
    df['Review'] = df['Review'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(df['Review'])

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

tfidf_df.to_excel('tfidf_values.xlsx', index=False)

print("Process Complete")
