# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path
# Code starts here
# read the dataset
dataset=pd.read_csv(path)
# look at the first five columns
print(dataset.head())
# Check if there's any column which is not useful and remove it like the column id
dataset.isnull().sum()
print(dataset.info())
dataset.drop(['Id'],axis=1,inplace=True)
print(dataset.head())
# check the statistical description
print(dataset.describe())


# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt
#names of all the attributes 
cols=dataset.columns
#number of attributes (exclude target)
size=len(cols)
print(cols)
print(size)
#x-axis has target attribute to distinguish between classes
x=list(dataset.columns)[-1]
#y-axis shows values of an attribute
y=dataset.iloc[:,-1]
print(type(x))
print(y)
#Plot violin for all attributes
sns.violinplot(data=dataset.iloc[:,:-1])
    



# --------------
import numpy
threshold = 0.5
# no. of features considered after ignoring categorical variables
num_features = 10
# create a subset of dataframe with only 'num_features'
subset_train =dataset.iloc[:,:num_features]
cols=subset_train.columns
subset_train.head()
#Calculate the pearson co-efficient for all possible combinations
data_corr=subset_train.corr()
plt.figure(figsize=(12,8))
sns.heatmap(data_corr, cmap='viridis')
#c = data_corr[(abs(data_corr)>0.5) & (abs(data_corr)< 1)]
#corr_var_list = c.unstack().dropna()
#s_corr_list = corr_var_list.sort_values(kind='quicksort')
#print(corr_var_list.head())
#print(s_corr_list)

# Set the threshold and search for pairs which are having correlation level above threshold

# Sort the list showing higher ones first 
def get_redundant_pairs(df):
    #'''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
       for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr

corr_var_list=data_corr[(abs(data_corr)>0.5) & (abs(data_corr)< 1)].unstack()
labels_to_drop = get_redundant_pairs(data_corr)
corr_var_list=corr_var_list.drop(labels=labels_to_drop).dropna()
#print(c.head())
print(corr_var_list)
s_corr_list=corr_var_list.sort_values(ascending=False)
#get_top_abs_correlations(subset_train)
print(s_corr_list)
#Print correlations and column names




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)
#print(dataset.columns)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
numerical_feature_columns = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
categorical_feature_columns = list(set(X_train.columns) - set(numerical_feature_columns))
scaler=StandardScaler()
#scaler.fit(X_train.loc[:,numerical_feature_columns])
X_train_temp=X_train[numerical_feature_columns].copy()
X_test_temp=X_test[numerical_feature_columns].copy()
X_train_temp[numerical_feature_columns]=scaler.fit_transform(X_train_temp[numerical_feature_columns])
X_test_temp[numerical_feature_columns]=scaler.fit_transform(X_test_temp[numerical_feature_columns])
#
#scaled_features_train_df = preprocessing.scale(X_train_temp)
#print(X_train_temp.head())
#print(X_test_temp.head())
#print(type(X_train_temp))
#print(X_train.loc[:,categorical_feature_columns])
X_train1=pd.concat([X_train_temp,X_train.loc[:,categorical_feature_columns]],axis=1)
#print(scaled_features_train_df)
# Scales are not the same for all variables. Hence, rescaling and standardization may be necessary for some algorithm to be applied on it.
#print(categorical_feature_columns)
print(X_train1.head())
X_test1=pd.concat([X_test_temp,X_test.loc[:,categorical_feature_columns]],axis=1)
print(X_test1.head())

scaled_features_train_df=X_train1
#Standardized
#Apply transform only for non-categorical data
scaled_features_test_df=X_test1

#Concatenate non-categorical data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif

#model=f_classif(X,y)
skb=SelectPercentile(f_classif,percentile=20)
predictors=skb.fit_transform(X_train1,y_train)
#print(predictors)
# Write your solution here:
scores=list(predictors)
top_k_index  = skb.get_support(indices=True)
top_k_predictors= predictors[top_k_index]
print(top_k_predictors)
print(top_k_index)
print(scores)



# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features =clf.fit(X_train , y_train)
predictions_all_features=clf.predict(X_test)
score_all_features= accuracy_score(y_test,predictions_all_features )
#print(len(scaled_features_train_df.columns))
#print(len(skb.get_support()))
print(scaled_features_train_df.columns[skb.get_support()])
#print(X_new.head())

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]
#print(y_test)
model_fit_top_features  =clf1.fit(X_new , y_train)
predictions_top_features=clf1.predict(X_test_new)
#print(predictions_top_features)
score_top_features= accuracy_score(y_test,predictions_top_features )




