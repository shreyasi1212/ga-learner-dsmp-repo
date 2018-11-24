# --------------
#Code starts here
race=census[:,2]
filter_0=census[:,2] ==0
race_0=census[filter_0]
#print(race_0)
filter_1=census[:,2] ==1
race_1=census[filter_1]

filter_2=census[:,2] ==2
race_2=census[filter_2]

filter_3=census[:,2] ==3
race_3=census[filter_3]

filter_4=census[:,2] ==4
race_4=census[filter_4]

len_0=len(race_0)
len_1=len(race_1)
len_2=len(race_2)
len_3=len(race_3)
len_4=len(race_4)

print(len_0)
print(len_1)
print(len_2)
print(len_3)
print(len_4)

list1=[]
list1.append(len_0)
list1.append(len_1)
list1.append(len_2)
list1.append(len_3)
list1.append(len_4)

print(list1)

minority_race=min(list1)
minority_race=3
print(minority_race)


# --------------
#Code starts here
filter_sc=census[:,0]> 60
print(filter_sc)
senior_citizens=census[filter_sc]
working_hours_sum=np.sum(senior_citizens[:,6])
senior_citizens_len=len(senior_citizens)
avg_working_hours= (working_hours_sum/senior_citizens_len)
print(avg_working_hours)


# --------------
# Importing header files
import numpy as np

# Path of the file has been stored in variable called 'path'
data=np.genfromtxt(path,delimiter=",", skip_header=1)
print(data[0])
#New record
new_record=[[50,  9,  4,  1,  0,  0, 40,  0]]
census=np.concatenate((data,new_record))
#Code starts here



# --------------
#Code starts here
filter_en1=census[:,1] > 10
filter_en2=census[:,1] <= 10
high=census[filter_en1]
low=census[filter_en2]
avg_pay_high=high[:,7].mean()
avg_pay_low=low[:,7].mean()
print(avg_pay_high)
print(avg_pay_low)


# --------------
#Code starts here
age=census[:,0]
max_age=age.max()
min_age=age.min()
age_mean=age.mean()
age_std=np.std(age)
print(max_age)
print(min_age)
print(age_mean)
print(age_std)


