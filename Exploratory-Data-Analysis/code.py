# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(path)
data.head()
#data['Rating'].hist()
mask1=data['Rating'] < 5
mask2=data['Rating'] ==5
#print(mask2)
data=data[mask1 | mask2]
#print(data)
data['Rating'].hist()

#Code starts here


#Code ends here


# --------------
# code starts here
total_null=data.isnull().sum()
percent_null=(total_null/data.isnull().count())
#print(percent_null)
missing_data=pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'] )
print(missing_data)
data.dropna(inplace=True)
print(data.head())

total_null_1=data.isnull().sum()
percent_null_1=(total_null_1/data.isnull().count())
#print(percent_null)
missing_data_1=pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'] )
print(missing_data_1)
# code ends here


# --------------
#Code starts here
data['Price'].value_counts()
data['Price']=data['Price'].str.replace('$','').astype('float')
ax=sns.regplot(x="Price", y="Rating", data=data)
ax.set_title('Rating vs Price [RegPlot]')
#Code ends here


# --------------
import seaborn as sns
#Code starts here
tips = sns.load_dataset("tips")
ax=sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)

ax.set_titles('Rating vs Category [BoxPlot]')
ax.set_xticklabels(rotation=90)


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace(',','')
data['Installs']=data['Installs'].str.replace('+','')
print(data.head())
data['Installs']=data['Installs'].astype('int')
data.head()

le = LabelEncoder()
data['Installs']=le.fit_transform(data['Installs'])


graph = sns.regplot(x="Installs", y="Rating" , data=data)
graph.set_title('Rating vs Installs [RegPlot]')
#Code ends here

#Code ends here



# --------------

#Code starts here
#split_genre=lambda x: x.split(';')[:0]
data['Genres'].unique()
data['Genres']=data['Genres'].apply(lambda x: x.split(';')[0])
print(data.head())
gr_mean=data.groupby(['Genres'],as_index=False)['Genres','Rating'].mean()
print(gr_mean.describe())
gr_mean=gr_mean.sort_values(by='Rating')
print(gr_mean.head(1))
print(gr_mean.tail(1))
print(gr_mean)
#Code ends here


# --------------

#Code starts here
#data.info()
data['Last Updated']=pd.to_datetime(data['Last Updated'])
max_date=data['Last Updated'].max()
data['Last Updated Days']= (max_date-data['Last Updated']).dt.days
print(data.head())
ax=sns.regplot(x="Last Updated Days", y="Rating", data=data)
ax.set_title('Rating vs Last Updated [RegPlot]')
#Code ends here


