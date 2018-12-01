# --------------
from sklearn.linear_model import Ridge

# Code starts here
ridge=Ridge(random_state=0)
ridge.fit(X_train,y_train)
y_pred=ridge.predict(X_test)
r2_ridge=r2_score(y_test, y_pred)


# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df=pd.read_csv(path)
#df.head()
X= df.copy()
X.drop(['Price'],axis=1,inplace=True)
#print(X.head())
y=df['Price']
#print(y)
#Code starts here
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state=6)
corr=X_train.corr()
print(corr)


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor=LinearRegression()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
r2=r2_score(y_test,y_pred)


# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso=Lasso(random_state=0)
lasso.fit(X_train,y_train)
y_pred=lasso.predict(X_test)
r2_lasso=r2_score(y_test, y_pred)


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor=LinearRegression()
score=cross_val_score(regressor, X_train, y_train, cv=10)
mean_score=np.mean(score)
print(mean_score)


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model=make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
r2_poly=r2_score(y_test,y_pred)
print(r2)


