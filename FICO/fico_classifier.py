#!/usr/bin/env python
# coding: utf-8

# In[161]:


import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd


accepted_data = pd.read_csv("data/accepted_2007_to_2018Q4.csv", usecols = ['last_fico_range_high','last_fico_range_low','loan_amnt','dti','zip_code','addr_state','emp_length','loan_status'],nrows=100)
rejected_data = pd.read_csv("data/rejected_2007_to_2018Q4.csv",usecols = ['Amount Requested', 'Debt-To-Income Ratio', 'Zip Code', 'State', 'Employment Length','Risk_Score'],nrows =100)
num_of_accepted = len(accepted_data)
num_of_rejected = len(rejected_data)


accepted_data = accepted_data.dropna(axis=0)
rejected_data = rejected_data.dropna(axis=0)



# In[162]:


#Calculate the feature from accepted data equivalent to "risk factor" in rejected data : mean of fico scores
cols = ['last_fico_range_high','last_fico_range_low']
Fico_mean = accepted_data[cols].astype(float).mean(axis=1) 

#add the new feature to accepted data
accepted_data_tmp = accepted_data.copy()
accepted_data_tmp['fico_mean'] = Fico_mean
accepted_data_new = accepted_data_tmp[['loan_amnt','dti','zip_code','addr_state','emp_length','fico_mean','loan_status']]

#Categorize accepted data by the loan status: Fully Paid =1, Charged Off, Default =0, else = remove rows
accepted_data_new.loan_status = accepted_data_new.loan_status.replace(['Fully Paid', 'Charged Off' , 'Current'] , [1,0,0])

#Remove rows that have values other than 'Fully Paid', 'Charged Off' , 'Current'
values_valid = [0,1]
accepted_data_new = accepted_data_new[accepted_data_new.loan_status.isin(values_valid)]
#print(accepted_data_new.head())


# In[163]:


#Choosing the features from rejected-data
rejected_data_tmp = rejected_data.copy()
rejected_data_new = rejected_data_tmp[['Amount Requested', 'Debt-To-Income Ratio', 'Zip Code', 'State', 'Employment Length','Risk_Score']]

#Change the column headers to match the columns of accepted_data_new
rejected_data_new.columns = ['loan_amnt','dti','zip_code','addr_state','emp_length','fico_mean']


# In[164]:


#Dealing with strings in the features
rejected_data_new['dti'] = rejected_data_new.dti.str.extract(r'(\d+)', expand=True).astype(int)
rejected_data_new['zip_code'] = rejected_data_new.zip_code.str.extract(r'(\d+)', expand=True).astype(int)
rejected_data_new['emp_length'] = rejected_data_new.emp_length.str.extract(r'(\d+)', expand=True).astype(int)

accepted_data_new['zip_code'] = accepted_data_new.zip_code.str.extract(r'(\d+)', expand=True).astype(int)
accepted_data_new['emp_length'] = accepted_data_new.emp_length.str.extract(r'(\d+)', expand=True).astype(int)

#Extract size of each accepted or rejected data
num_of_accepted = len(accepted_data_new)
num_of_rejected = len(rejected_data_new)

#Remove the "loan_status" column for the concatenation of all data
accepted_tmp = accepted_data_new[['loan_amnt','dti','zip_code','addr_state','emp_length','fico_mean']]

#Concatenate all data to do the numerization of the categorical data for all homogeneously
all_data = pd.concat([accepted_tmp,rejected_data_new])

#Change the categorical feature "addr_state" to numerical feature
address_state = all_data['addr_state'].unique().tolist()
all_data['addr_state'] = all_data['addr_state'].apply( lambda x : address_state.index(x))

#print(all_data.head())


# In[165]:


#Separate accepted and rejected dat after homogenizing all features
processed_data_accepted = all_data[0:num_of_accepted]
processed_data_accepted = processed_data_accepted.astype('int')

processed_data_rejected = all_data[num_of_accepted:]
processed_data_rejected = processed_data_rejected.astype('int')

#target values for calssification: loan_status column
target = accepted_data_new.loan_status
target = target.astype('int')


# In[166]:


#Installation of Graphvis python package as well as Graphvis on the operating system itself is required 
#for the visualization step 

#For Windows after installing Graphvis package and app:
# import os
# os.environ['PATH'] += os.pathsep + r'C:\Program Files (x86)\Graphviz2.38\bin'

#Shuffle data
cv = ShuffleSplit(n_splits= 5, test_size=0.4, random_state=0)

#Design a decision tree
clf = tree.DecisionTreeClassifier()

#Perform cross validation and calculate the score 
cross_val_score(clf, processed_data_accepted, target, cv=cv)

#Fit the model on the data to later visualize the designed tree
clf_fitted = clf.fit(processed_data_accepted, target)


#Visualize the tree
##Comment out if graphviz not installed
dot_data = tree.export_graphviz(clf_fitted, out_file=None) 
graph = graphviz.Source(dot_data) 
graph
graph.render("accepted_data")


# In[8]:


#clf = clf.fit(X, Y)
#clf.predict_proba([[2., 2.]])


# In[ ]:




