import string
from lxml import etree
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download("all", quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer

preprocessing = "lemmatize"  # "stem"

with open("MsgLog_2017-01-24-011347.xml", "rt") as f:
    s = f.read()
parser = etree.XMLParser(recover=True)
tree = etree.fromstring(s, parser=parser)

english_words = set(nltk.corpus.words.words('en'))
stop_words = set(stopwords.words('english'))

ps = PorterStemmer()
lt = WordNetLemmatizer()

s = s.translate(str.maketrans('','',string.punctuation))
s = s.translate(str.maketrans('','',string.digits))
s = s.lower()

for tag in ["Subject", "Body"]:
    texts = []
    for node in tree.iter(tag=tag):
        # remove punctuation
        content = node.text.translate(str.maketrans('','',string.punctuation))
        # remove digits
        content = content.translate(str.maketrans('','',string.digits))
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
    text_string = [" ".join(ts) for ts in text_strings] 

    vect = TfidfVectorizer()
    tfidf = vect.fit_transform([text_string])
    tfidf_terms = vect.get_feature_names()
    tfidf_words = pd.Series(tfidf.toarray().flatten(),
                            index=tfidf_terms)
    tfidf_words.to_csv("TF_{}.csv".format(tag))



