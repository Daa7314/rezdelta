#Import packages

#Statistical packages
import numpy as np
import pandas as pd
import requests
import statistics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Importing word document
import docx2txt
import docx


#word cloud
from pathlib import Path
import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud



#Natural language processing packages
import nltk
from nltk.probability import FreqDist
import spacy
from textblob import TextBlob
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity


file = "C:/Users/justd/Downloads/Dammi Akinwale - analyst_LSq.docx"
document = docx2txt.process(file)


#Import txt file JD
with open('C:/Users/justd/Downloads/JD.txt', encoding='utf-8') as f:
    jd_text = f.read()

#print(jd_text)


from sklearn.feature_extraction.text import CountVectorizer
data = [document, jd_text]
count_vectorizer = CountVectorizer()
vector_matrix = count_vectorizer.fit_transform(data)
vector_matrix


cosine_similarity_matrix = cosine_similarity(vector_matrix)
resulted = pd.DataFrame(cosine_similarity_matrix,['resume','jd_text'])


#Keyword extraction
nltk.download('stopwords')
from rake_nltk import Rake
rake_nltk_var = Rake()
text = jd_text
rake_nltk_var.extract_keywords_from_text(text)
keyword_extracted = rake_nltk_var.get_ranked_phrases()

print(x1)


# vectr = TfidfVectorizer()

# user = vectr.fit(keyword_extracted2)

# x2 = list(user.get_feature_names_out())
# print(x2)

x2.count('data')


d = 0
for a in x1:
    d += x2.count(a)
print(d)

print(len(x1))


#Elements in list not in resume/cv

missing_keywords = list(set(x1) - set(x2))

print(missing_keywords)