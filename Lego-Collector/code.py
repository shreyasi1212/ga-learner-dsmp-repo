# --------------
import pandas as pd
from sklearn.cross_validation import train_test_split
# code starts here
df=pd.read_csv(path)
print(df.head())
X=df[['ages','num_reviews','piece_count','play_star_rating','review_difficulty','star_rating','theme_name','val_star_rating','country']].copy()
y=df['list_price'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)
# code ends here



# --------------
# Code starts here

corr=X_train.corr()
print(corr)
corr[corr>0.75]
#corr2=corr[[(corr>0.75) or (corr < -0.75)]]
# Code ends here
X_train.drop(['play_star_rating', 'val_star_rating'], axis=1,inplace=True)
X_test.drop(['play_star_rating', 'val_star_rating'], axis=1,inplace=True)
print(X_train.info())
print(X_test.info())


# --------------
import matplotlib.pyplot as plt

# code starts here        
cols=X_train.columns.values
print(cols)
fig, axes=plt.subplots(nrows=3, ncols=3)

for i in range(0,3):
  for j in range(0,3):
    col= cols[i * 3 + j]
    axes[i,j].scatter(X_train[col],y_train)

plt.show()
# code ends here



# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
# Code starts here
regressor = LinearRegression()
#y_train_new=np.log(y_train)
# fit model on training data
regressor.fit(X_train, y_train)
# predict on test features
y_pred = regressor.predict(X_test)
mse=(mean_squared_error(y_test,y_pred))
print(mse)
r2=r2_score(y_test,y_pred)
print(r2)
# Code ends here


# --------------
# Code starts here
residual=y_test - y_pred
plt.hist(residual)



# Code ends here


