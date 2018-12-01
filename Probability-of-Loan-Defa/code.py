# --------------
# code starts here

prob_lp=len(df[df['paid.back.loan']=='Yes'])/len(df)
prob_cs=len(df[df['credit.policy']=='Yes'])/len(df)
print(prob_lp)
print(prob_cs)
new_df=df[df['paid.back.loan']=='Yes']
prob_pd_cs = len(new_df[new_df['credit.policy']=='Yes'])/len(new_df)

print(prob_pd_cs)
bayes=(prob_pd_cs * prob_lp) /prob_cs
print(bayes)
# code ends here


# --------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# code starts here
df=pd.read_csv(path)
p_a=len(df[df['fico']>700])/len(df)
p_b=len(df[df['purpose'] == 'debt_consolidation'])/len(df)
print(p_a)
print(p_b)
df1=df[df['purpose'] == 'debt_consolidation']
p_a_b= (p_a * p_b)/p_a
result=(p_a_b==p_a)
print(result)
#print(df['purpose'] )
# code ends here


# --------------
# code starts here
df['purpose'].value_counts(ascending=False).plot(kind='bar')
df1=df[df['paid.back.loan']=='No']
df1['purpose'].value_counts(ascending=False).plot(kind='bar')
#plt.show()
# code ends here


# --------------
# code starts here

inst_median=df['installment'].median()
inst_mean=df['installment'].mean()
plt.hist(df['installment'])
plt.hist(df['log.annual.inc'])
#print(df.columns)
# code ends here


