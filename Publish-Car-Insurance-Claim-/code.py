# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df= pd.read_csv(path)
df['INCOME']=df['INCOME'].str.replace('$','')
df['INCOME']=df['INCOME'].str.replace(',','')
df['HOME_VAL']=df['HOME_VAL'].str.replace('$','')
df['HOME_VAL']=df['HOME_VAL'].str.replace(',','')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace('$','')
df['BLUEBOOK']=df['BLUEBOOK'].str.replace(',','')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace('$','')
df['OLDCLAIM']=df['OLDCLAIM'].str.replace(',','')
df['CLM_AMT']=df['CLM_AMT'].str.replace('$','')
df['CLM_AMT']=df['CLM_AMT'].str.replace(',','')

#df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].replace('$','')
#df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=df[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']].replace(',','')
df.head()
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
count=y.value_counts()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=6)

# Code ends here


# --------------
# Code starts here

#X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=pd.to_numeric=X_train[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]

#X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]=pd.to_numeric=X_test[['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT']]
X_train['INCOME']=pd.to_numeric(X_train['INCOME'])
X_train['HOME_VAL']=pd.to_numeric(X_train['HOME_VAL'])
X_train['BLUEBOOK']=(pd.to_numeric(X_train['BLUEBOOK'])).astype(float)
X_train['OLDCLAIM']=(pd.to_numeric(X_train['OLDCLAIM'])).astype(float)
X_train['CLM_AMT']=(pd.to_numeric(X_train['CLM_AMT'])).astype(float)

X_test['INCOME']=pd.to_numeric(X_test['INCOME'])
X_test['HOME_VAL']=pd.to_numeric(X_test['HOME_VAL'])
X_test['BLUEBOOK']=(pd.to_numeric(X_test['BLUEBOOK'])).astype(float)
X_test['OLDCLAIM']=(pd.to_numeric(X_test['OLDCLAIM'])).astype(float)
X_test['CLM_AMT']=(pd.to_numeric(X_test['CLM_AMT'])).astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
print(X_train.dtypes)
# Code ends here


# --------------
# Code starts here
X_train.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)
X_train.isnull().sum()
X_test.dropna(subset = ['YOJ','OCCUPATION'],inplace=True)
X_test.isnull().sum()
y_train=y_train.drop(index=X_train.index)
y_test=y_test.drop(index=X_test.index)
X_train['AGE']=X_train['AGE'].fillna(X_train['AGE'].mean())
X_train['CAR_AGE']=X_train['CAR_AGE'].fillna(X_train['CAR_AGE'].mean())
X_train['INCOME']=X_train['INCOME'].fillna(X_train['INCOME'].mean())
X_train['HOME_VAL']=X_train['HOME_VAL'].fillna(X_train['HOME_VAL'].mean())

X_test['AGE']=X_test['AGE'].fillna(X_test['AGE'].mean())
X_test['CAR_AGE']=X_test['CAR_AGE'].fillna(X_test['CAR_AGE'].mean())
X_test['INCOME']=X_test['INCOME'].fillna(X_test['INCOME'].mean())
X_test['HOME_VAL']=X_test['HOME_VAL'].fillna(X_test['HOME_VAL'].mean())

print(y_train)
# Code ends here


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
le=LabelEncoder()

for i in range(0,X_train.shape[1]):
    if X_train.dtypes[i]=='object':
        X_train[X_train.columns[i]] = le.fit_transform(X_train[X_train.columns[i]])

for i in range(0,X_test.shape[1]):
    if X_test.dtypes[i]=='object':
        X_test[X_test.columns[i]] = le.fit_transform(X_test[X_test.columns[i]])

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 

model=LogisticRegression(random_state=6)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote=SMOTE(random_state=6)
X_train,y_train=smote.fit_sample(X_train,y_train)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model=LogisticRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
score=accuracy_score(y_test,y_pred)
# Code ends here


