import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_html_tags(text):
    if "<" in text and ">" in text:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    return text

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def clean_data(text):
    text = clean_html_tags(text)
    text = remove_special_characters(text)
    text = remove_stop_words(text)
    return text

def tokenize_text(text):
    return word_tokenize(text)

def clean_and_tokenize_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    df_cleaned = df.apply(lambda col: col.map(lambda x: clean_data(str(x)) if isinstance(x, str) else x))

    df_tokenized = df_cleaned.apply(lambda col: col.map(lambda x: tokenize_text(str(x)) if isinstance(x, str) else x))

    df_tokenized.to_csv(output_csv, index=False)

if __name__ == "__main__":
    input_csv = 'movie reviews 1 rating.csv'
    output_csv = 'cleaned_tokenized_data.csv'
    clean_and_tokenize_dataset(input_csv, output_csv)
