# --------------
# import the libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Code starts here
df=pd.read_csv(path)
print(df.head())
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
print(X.head())
print(y.head())
X_train,X_test,y_train,y_test =train_test_split (X,y,test_size=0.2,random_state=6)
# Code ends here


# --------------
import matplotlib.pyplot as plt


# Code starts here
plt.boxplot(X_train['bmi'])
q_value=X_train['bmi'].quantile(0.95)
print(y_train.value_counts())

# Code ends here


# --------------
import seaborn as sns
# Code starts here
relation=X_train.corr()
print(relation)
sns.pairplot(X_train)
# Code ends here


# --------------
import seaborn as sns
import matplotlib.pyplot as plt

# Code starts here
cols=['children','sex','region','smoker']
fig,axes= plt.subplots(2,2)
for i in range(0,2):
    for j in range(0,2):
        col=cols[ i * 2 + j]
        sns.countplot(X_train[col],hue=y_train, ax=axes[i,j])
   
# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# parameters for grid search
parameters = {'C':[0.1,0.5,1,5]}

# Code starts here
lr=LogisticRegression()
grid=GridSearchCV(lr,parameters)
grid.fit(X_train,y_train)
y_pred=grid.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
# Code ends here


# --------------
from sklearn.metrics import roc_auc_score
from sklearn import metrics
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Code starts here

#print(y_pred)
#predictions=pd.Series(y_pred,name='predicted')
#y_test=y_test.reset_index(drop=True)
#y_pred_proba=pd.concat([y_test,predictions],axis=1)
#y_pred_proba=y_pred_proba[y_pred_proba['insuranceclaim']==1]['predicted']
#print(y_pred_proba)
#y_pred_proba=y_test.value_counts()[1]/len(y_test)
#print(y_pred_proba)

# Code ends here

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
score=roc_auc_score(y_pred,y_test)
#logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
auc=metrics.auc(fpr, tpr)
#roc_auc=roc_auc_score(y_test.where(y_test==1),y_pred_proba)
plt.plot(fpr,tpr,label="Logistic model, auc="+str(auc))
y_pred_proba=grid.predict_proba(X_test)[:,1]
print(y_pred_proba)
roc_auc=roc_auc_score(y_test,y_pred_proba)
#fpr, tpr, thresholds = roc_curve(y_test, )
#plt.figure()
#plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
#plt.show()


