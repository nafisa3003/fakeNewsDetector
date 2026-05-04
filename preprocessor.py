import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def wordopt(text):
    if not isinstance(text, str):
        return ""
    # cleans text by removing special characters, html tags, links, and punctuation
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()


def preprocess(data_fake, data_true):
    # prepares data by merging, cleaning, and converting text to TF-IDF vectors
    data_merge = pd.concat([data_fake, data_true], axis=0)

    # Only drop columns if they exist
    cols_to_drop = ['title', 'subject', 'date', 'index']
    existing_cols = [c for c in cols_to_drop if c in data_merge.columns]
    data = data_merge.drop(existing_cols, axis=1)

    data = data.sample(frac=1, random_state=42)
    data.reset_index(inplace=True, drop=True)

    data['text'] = data['text'].apply(wordopt)

    x = data['text']
    y = data['class']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    # transforms text data into numerical TF-IDF features
    vectorization = TfidfVectorizer(max_features=50000)
    xv_train = vectorization.fit_transform(x_train)
    xv_test  = vectorization.transform(x_test)

    return xv_train, xv_test, y_train, y_test, vectorization