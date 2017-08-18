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
from sklearn.decomposition import TruncatedSVD


########### define optional parameters here ##########
preprocessing = "lemmatize"  # "stem"
n_words_to_select = 100
######################################################

# for writing output
def to_xml(df, name):
    with open(name, "wt") as f:
        for i, row in df.iterrows():
            f.write("<field term=\"{term}\"><occurence>{occurence}</occurence></field>\n" \
                    .format(term=i, occurence=row["occurence"]))


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
  
    # calculate TF-IDF scores
    tfidf_vect = TfidfVectorizer()
    # sum over all documents to get importance score for each word
    tfidf = tfidf_vect.fit_transform(text_strings).toarray().sum(axis=0)
    tfidf_terms = tfidf_vect.get_feature_names()

    # count terms in each document
    count_vect = CountVectorizer()
    count = count_vect.fit_transform(text_strings).toarray()
    # sum them up across all documents
    total_counts = count.sum(axis=0)
    count_terms = count_vect.get_feature_names()

    # perform LSA:
    lsa = TruncatedSVD(n_words_to_select)
    lsa.fit_transform(count)
    
    # get largest component from each LSA vector
    lsa_vectors = lsa.components_
    top_components = [count_terms[vector.argmax()] for vector in lsa_vectors]

    # put terms, counts and TF-IDF scores into a dataframe to make writing out the results easier
    output = pd.DataFrame({"term": count_terms, "occurence": total_counts})
    output.index = output["term"]
    output["TF-IDF"] = tfidf

    # pick top scoring TF-IDF terms, write to file (CSV and XML)
    output.nlargest(n_words_to_select, "TF-IDF")[["occurence"]].to_csv("TF_{}.csv".format(tag))
    to_xml(output.nlargest(n_words_to_select, "TF-IDF")[["occurence"]],
           "TF_{}.xml".format(tag))

    # select top LDA components, write to file (CSV and XML)
    output.loc[top_components, ["occurence"]].to_csv("LSA_{}.csv".format(tag))
    to_xml(output.loc[top_components, ["occurence"]],
           "LSA_{}.xml".format(tag))

