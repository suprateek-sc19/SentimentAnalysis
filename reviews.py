# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:33:38 2019

@author: Suprateek Chatterjee
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

#Training
tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def getCleanReview(review):
    
    review = review.lower()
    review = review.replace('<span data-hook="review-body" class="a-size-base review-text review-text-content"><span class="">'," ")
    review = review.replace("<br /><br />"," ")
    review = review.replace("<br>"," ")
    review = review.replace("</span>"," ")
    tokens = tokenizer.tokenize(review)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    
    cleaned_review = ' '.join(stemmed_tokens)
    
    return cleaned_review


df = pd.read_csv('Train.csv')

x = df.iloc[0:20000,0].values
Y = df.iloc[0:20000,1].values

le = LabelEncoder()
y = le.fit_transform(Y)

x_clean = [getCleanReview(i) for i in x]

cv = CountVectorizer()
x_vec = cv.fit_transform(x_clean).toarray()

mnb = MultinomialNB()
mnb.fit(x_vec,y)



#Testing
def getReview(review):
    
    review = review.lower()
    review = review.replace('<span data-hook="review-body" class="a-size-base review-text review-text-content"><span class="">'," ")
    review = review.replace("<br /><br />"," ")
    review = review.replace("<br>"," ")
    review = review.replace("</span>"," ")
    return review

test = pd.read_csv('reviews.csv')
x_test = test.iloc[0:,0]

test_clean = [getCleanReview(i) for i in x_test]
xt_vec = cv.transform(test_clean).toarray()

result = mnb.predict(xt_vec)
reslist = result.tolist()

for i in range(len(reslist)):
    if reslist[i]==0:
        reslist[i]="neg"
    else:
        reslist[i]="pos"
        


prediction = pd.DataFrame(reslist, columns=['label']).to_csv('prediction.csv')

#WordCloud
revs = pd.read_csv("reviews.csv")

r = revs.iloc[0:,0].values



rev = [getReview(i) for i in r]

comment_words = ' '
stopwords = set(STOPWORDS) 
# iterate through the csv file 
for val in rev: 
      
    # typecaste each val to string 
    val = str(val) 
  
    # split the value 
    tokens = val.split() 
      
    # Converts each token into lowercase 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    for words in tokens: 
        comment_words = comment_words + words + ' ' 
    
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
        
pred = pd.read_csv("prediction.csv")
p = pred.iloc[0:,1].values

if np.sum(p=="neg") > np.sum(p=="pos"):
    print("THE OVERALL SENTIMENT OF THIS PRODUCT IS NEGATIVE! YOU SHOULD ALSO LOOK AT OTHER ALTERNATIVES")
else:
    print("THE OVERALL SENTIMENT OF THIS PRODUCT IS POSITIVE! GO FOR IT")








