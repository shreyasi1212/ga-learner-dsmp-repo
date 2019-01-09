# --------------
import pandas as pd
from sklearn.model_selection import train_test_split
#path - Path of file 
df=pd.read_csv(path)
# Code starts here
X=df.iloc[:,1:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)




# --------------
import numpy as np
from sklearn.preprocessing import LabelEncoder
# Code starts here
X_train['TotalCharges']=X_train['TotalCharges'].replace(' ',np.NaN)
X_test['TotalCharges']=X_test['TotalCharges'].replace(' ',np.NaN)
X_train['TotalCharges']=pd.to_numeric(X_train['TotalCharges'])
X_test['TotalCharges']=pd.to_numeric(X_test['TotalCharges'])
X_train['TotalCharges']=X_train['TotalCharges'].fillna((X_train['TotalCharges'].mean()))
X_test['TotalCharges']=X_test['TotalCharges'].fillna((X_test['TotalCharges'].mean()))
print(X_train.isnull().sum())
le=LabelEncoder()

for i in range(0,X_train.shape[1]):
    if X_train.dtypes[i]=='object':
        X_train[X_train.columns[i]] = le.fit_transform(X_train[X_train.columns[i]])

for i in range(0,X_test.shape[1]):
    if X_test.dtypes[i]=='object':
        X_test[X_test.columns[i]] = le.fit_transform(X_test[X_test.columns[i]])
#X_train=X_train.apply(LabelEncoder().fit_transform)
#X_test=X_test.apply(LabelEncoder().fit_transform)
cleanup_value =  {'No':0, 'Yes':1}
X_train['TotalCharges']=X_train['TotalCharges'].astype(float)
X_test['TotalCharges']=X_test['TotalCharges'].astype(float)
y_train=y_train.replace(cleanup_value)
y_test=y_test.replace(cleanup_value)
print(X_train['TotalCharges'])




# --------------
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

# Code starts here
ada_model=AdaBoostClassifier(random_state=0)
ada_model.fit(X_train,y_train)
y_pred=ada_model.predict(X_test)
ada_score=accuracy_score(y_test,y_pred)
ada_cm=confusion_matrix(y_test,y_pred)
ada_cr=classification_report(y_test,y_pred)


# --------------
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

#Parameter list
parameters={'learning_rate':[0.1,0.15,0.2,0.25,0.3],
            'max_depth':range(1,3)}

# Code starts here
xgb_model=XGBClassifier(random_state=0)
xgb_model.fit(X_train,y_train)
y_pred=xgb_model.predict(X_test)
xgb_score=accuracy_score(y_test,y_pred)
xgb_cm=confusion_matrix(y_test,y_pred)
xgb_cr=classification_report(y_test,y_pred)

clf_model=GridSearchCV(estimator=xgb_model,param_grid=parameters)
clf_model.fit(X_train,y_train)
y_pred=clf_model.predict(X_test)
clf_score=accuracy_score(y_test,y_pred)
clf_cm=confusion_matrix(y_test,y_pred)
clf_cr=classification_report(y_test,y_pred)

print(xgb_cr)
print(clf_cr)


