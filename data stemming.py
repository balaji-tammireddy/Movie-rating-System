import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def stem_tokenized_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    df_stemmed = df.apply(lambda col: col.map(lambda x: stem_tokens(eval(x)) if isinstance(x, str) else x))

    df_stemmed.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = 'cleaned_tokenized_data.csv'
    output_csv = 'stemmed_data.csv'
    stem_tokenized_dataset(input_csv, output_csv)
