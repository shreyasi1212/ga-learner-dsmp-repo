# --------------
# import packages

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 



# Load Offers
offers=pd.read_excel(path,sheet_name=0)
transactions=pd.read_excel(path,sheet_name=1)
transactions['n']=1
df=offers.merge(transactions)
df.head()

# Load Transactions


# Merge dataframes


# Look at the first 5 rows



# --------------
# Code starts here
df.columns
# create pivot table
matrix=df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')
matrix.fillna(0,inplace=True)
matrix.reset_index(inplace=True)
print(matrix.head())
matrix.columns
# replace missing values with 0


# reindex pivot table


# display first 5 rows


# Code ends here


# --------------
# import packages
from sklearn.cluster import KMeans

# Code starts here

# initialize KMeans object
cluster=KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10 , random_state=0)
matrix['cluster']=cluster.fit_predict(matrix[matrix.columns[1:]])

# create 'cluster' column

print(matrix)

# Code ends here


# --------------
# import packages
from sklearn.decomposition import PCA

# Code starts here

# initialize pca object with 2 components

pca=PCA(n_components=2, random_state=0)
#print(pca.fit_transform(matrix[matrix.columns[1:]]))
# create 'x' and 'y' columns donoting observation locations in decomposed form
matrix['x']=pca.fit_transform(matrix[matrix.columns[1:]])[:,0]
matrix['y']=pca.fit_transform(matrix[matrix.columns[1:]])[:,1]
# dataframe to visualize clusters by customer names
clusters=matrix.iloc[:,[0, 33, 34, 35]]
#print(clusters)
# visualize clusters
clusters.plot.scatter(x='x',y='y',c='cluster',colormap='viridis')

# Code ends here


# --------------
# Code starts here

# merge 'clusters' and 'transactions'
data=clusters.merge(transactions)

# merge `data` and `offers`
data=offers.merge(data)
# initialzie empty dictionary
data.head()
champagne={}
# iterate over every cluster
for i in range(0,5):
    # observation falls in that cluster
    champagne[i]=0
    new_df=data[data['cluster']==i]
    # sort cluster according to type of 'Varietal'
    counts=new_df['Varietal'].value_counts(ascending=False)
    # check if 'Champagne' is ordered mostly
    print(i)
    print(counts.index[0])
    if (counts.index[0]=='Champagne'):
        champagne[i]=counts[0]
        # add it to 'champagne'
print(champagne)
cluster_champagne= max(champagne, key=lambda k: champagne[k])
print(cluster_champagne)

# get cluster with maximum orders of 'Champagne' 


# print out cluster number




# --------------
# Code starts here

# empty dictionary
discount={}

# iterate over cluster numbers
for i in range(0,5):
    # dataframe for every cluster
    new_df=data[data['cluster']==i]
    # average discount for cluster
    print(new_df['Discount (%)'].sum()/len(new_df))
    counts=new_df['Discount (%)'].sum()/len(new_df)
    # adding cluster number as key and average discount as value 
    discount[i]=counts



# cluster with maximum average discount
cluster_discount=max(discount, key=lambda k: discount[k])
print(cluster_discount)
# Code ends here


