# --------------
# code starts here


def month_year(value):
    return value / 12

loan_term = bank['Loan_Amount_Term'].apply(month_year)
#print(loan_term)
big_loan_term= len(loan_term.where(loan_term >= 25).dropna())
#big_loan_term1.dropna(inplace=True)
#ig_loan_term=len(big_loan_term1)
print(big_loan_term)
# code ends here


# --------------
# Code starts here
#print(banks.isnull().sum())
avg_loan_amount=bank.pivot_table(values='LoanAmount',index=['Gender','Married','Self_Employed'],aggfunc=np.mean)
print(avg_loan_amount)

# code ends here



# --------------
# code ends here
loan_groupby=bank.groupby('Loan_Status')['ApplicantIncome', 'Credit_History']
#bank.head()
mean_values=loan_groupby.mean()
print(mean_values)

# code ends here


# --------------
# code starts here
loan_approved_se=len(banks[(banks['Self_Employed']=='Yes') & (banks['Loan_Status']=='Y')])
loan_approved_nse=len(banks[(banks['Self_Employed']=='No') & (banks['Loan_Status']=='Y')])
loan_status=len(banks['Loan_Status'])
print(loan_approved_se)
print(loan_approved_nse)
print(loan_status)
percentage_se=(loan_approved_se/loan_status)*100
percentage_nse=(loan_approved_nse/loan_status)*100
print(percentage_se)
print(percentage_nse)
# code ends here


# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
bank=pd.read_csv(path,sep=',')
#print(bank)
categorical_var=bank.select_dtypes(include=['object'])
print(categorical_var)
numerical_var=bank.select_dtypes(include=['number'])
print(numerical_var)
# code starts here






# code ends here


# --------------
# code starts here
banks=bank.drop('Loan_ID',axis=1)
print(banks.isnull().sum())
#code ends here
bank_mode=banks.mode()
print(bank_mode)

bank.fillna(banks.mode().iloc[0], inplace=True)
banks=bank
banks.drop('Loan_ID',axis=1,inplace=True)
print(banks.isnull().sum())


