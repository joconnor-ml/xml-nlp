import pandas as pd
import string
from lxml import etree
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("all", quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

preprocessing = "lemmatize"  # "stem"

with open("MsgLog_2017-01-24-011347.xml", "rt") as f:
    s = f.read()

parser = etree.XMLParser(recover=True)
tree = etree.fromstring(s, parser=parser)

english_words = set(nltk.corpus.words.words('en'))
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
lt = WordNetLemmatizer()

for tag in ["Subject", "Body"]:
    texts = []
    for node in tree.iter(tag=tag):
        # remove punctuation
        content = node.text.translate(str.maketrans('','',string.punctuation))
        # remove digits
        content = content.translate(str.maketrans('','',string.digits))
        # lowercase
        content = content.lower()
        # tokenize
        word_list = word_tokenize(content)
        # remove stop words and words not in dictionary
        word_list = [word for word in word_list
                     if word in english_words and word not in stop_words]
        # stem or lemmatize
        if preprocessing == "stem":
            word_list = [ps.stem(word) for word in word_list]
        elif preprocessing == "lemmatize":
            word_list = [lt.lemmatize(word) for word in word_list]
    
        texts.append(word_list)

    text_strings = [" ".join(words) for words in texts]
    text_string = " ".join(text_strings)
  
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([text_string]).toarray().flatten()
    tfidf_terms = vect.get_feature_names()

    vect = CountVectorizer()
    count = vect.fit_transform([text_string]).toarray().flatten()
    count_terms = vect.get_feature_names()
    
    output = pd.DataFrame({"term": count_terms, "occurence": count})
    output.index = output["term"]
    output["TF-IDF"] = tfidf
    output[["occurence", "TF-IDF"]].to_csv("TF_{}.csv".format(tag))
