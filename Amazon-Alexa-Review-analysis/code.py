# --------------
# import packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# Load the dataset

df=pd.read_table(path)
# Converting date attribute from string to datetime.date datatype 
#print(df.head())
df['date']= df['date'].apply(lambda x: datetime.strptime(x, '%d-%b-%y'))
#print(df.head())

# calculate the total length of word

#df['verified_reviews_len']=len(df['verified_reviews'].apply(lambda x: x.split()))
#length=df['verified_reviews_len'].sum()
df['length']=df['verified_reviews'].apply(len)
#print(length)
#df.drop(['verified_reviews_len'],axis=1,inplace=True)
print(df.shape)






# --------------
## Rating vs feedback

# set figure size


# generate countplot
sns.countplot(x = 'rating', hue = 'feedback' , data = df)
#df.head()
# display plot


## Product rating vs feedback
sns.countplot(x = 'rating', hue = 'variation' , data = df)
# set figure size


# generate barplot


# display plot




# --------------
# import packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# declare empty list 'corpus'
corpus=[]

# for loop to fill in corpus
for i in range(0,3150):
    # retain alphabets
    review=re.sub('[^a-zA-Z]', ' ',df.iloc[i]['verified_reviews'])
    #print(review)
    # convert to lower case
    review=review.lower()

    # tokenize
    review=review.split()

    # initialize stemmer object
    ps=PorterStemmer()
    # perform stemming
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    # join elements of list
    review=' '.join(review)
    # add to 'corpus'
    corpus.append(review)

print(corpus)
    
    
# display 'corpus'



# --------------
# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Instantiate count vectorizer
cv=CountVectorizer (max_features =1500)

# Independent variable
X=cv.fit_transform(corpus)

# dependent variable
y=df['feedback']

# Counts
count=df['feedback'].value_counts()

# Split the dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size = 0.2,random_state = 0)

print(count)





# --------------
# import packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

# Insatntiate calssifier
rf=RandomForestClassifier(random_state=2)

# fit modelk on training data
rf_model=rf.fit(X_train,y_train)

# predict on test data
y_pred=rf_model.predict(X_test)

# calculate the accuracy score
score=accuracy_score(y_test,y_pred)

# calculate the precision
precision=precision_score(y_test,y_pred)

# display 'score' and 'precision'
print(score)
print(precision)



# --------------
# import packages
from imblearn.over_sampling import SMOTE

# Instantiate smote
smote=SMOTE()

# fit_sample onm training data
X_train,y_train=smote.fit_sample(X_train,y_train)

# fit modelk on training data
rf_model=rf.fit(X_train,y_train)

# predict on test data
y_pred=rf_model.predict(X_test)

# calculate the accuracy score
score=accuracy_score(y_test,y_pred)

# calculate the precision
precision=precision_score(y_test,y_pred)

# display 'score' and 'precision'
print(score)
print(precision)




