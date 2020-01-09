import numpy as np
import scipy as sp
from strategic_code import configuration as Configuration
import pandas as pd
import random
import time
import pickle
import pandas as pd
import json as js
import click
from joblib import Parallel, delayed

from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV


@click.command()
@click.option('--accepted_data', required=True)
@click.option('--rejected_data', required=True)
@click.option('--output', required=True, help="output directory")
@click.option('--n', default=None, type=int, help="number of samples")
@click.option('--njobs', default=1, type=int, help="number of samples")
def experiment(accepted_data, rejected_data, output, n, njobs):
    print(n)
    accepted_data = pd.read_csv(accepted_data, usecols=[
                                'last_fico_range_high', 'last_fico_range_low', 'loan_amnt', 'dti', 'zip_code', 'addr_state', 'emp_length', 'loan_status'], nrows=n)
    rejected_data = pd.read_csv(rejected_data, usecols=[
                                'Amount Requested', 'Debt-To-Income Ratio', 'Zip Code', 'State', 'Employment Length', 'Risk_Score'], nrows=n)

    num_of_accepted = len(accepted_data)
    num_of_rejected = len(rejected_data)
    accepted_data = accepted_data.dropna(axis=0)
    rejected_data = rejected_data.dropna(axis=0)

    cols = ['last_fico_range_high', 'last_fico_range_low']
    Fico_mean = accepted_data[cols].astype(float).mean(axis=1)

    # add the new feature to accepted data
    accepted_data_tmp = accepted_data.copy()
    accepted_data_tmp['fico_mean'] = Fico_mean
    accepted_data_new = accepted_data_tmp[[
        'loan_amnt', 'dti', 'zip_code', 'addr_state', 'emp_length', 'fico_mean', 'loan_status']]

    # Categorize accepted data by the loan status: Fully Paid =1, Charged Off, Default =0, else = remove rows
    accepted_data_new.loan_status = accepted_data_new.loan_status.replace(
        ['Fully Paid', 'Charged Off', 'Default'], [1, 0, 0])

    # Remove rows that have values other than 'Fully Paid', 'Charged Off' , 'Current'
    values_valid = [0, 1]
    accepted_data_new = accepted_data_new[accepted_data_new.loan_status.isin(
        values_valid)]
    # print(accepted_data_new.head())

    # In[163]:

    # Choosing the features from rejected-data
    rejected_data_tmp = rejected_data.copy()
    rejected_data_new = rejected_data_tmp[[
        'Amount Requested', 'Debt-To-Income Ratio', 'Zip Code', 'State', 'Employment Length', 'Risk_Score']]

    # Change the column headers to match the columns of accepted_data_new
    rejected_data_new.columns = ['loan_amnt', 'dti',
                                 'zip_code', 'addr_state', 'emp_length', 'fico_mean']

    # Dealing with strings in the features
    rejected_data_new['dti'] = rejected_data_new.dti.str.extract(
        r'(\d+)', expand=True).astype(float)
    rejected_data_new['zip_code'] = rejected_data_new.zip_code.str.extract(
        r'(\d+)', expand=True).astype(float)
    rejected_data_new['emp_length'] = rejected_data_new.emp_length.str.extract(
        r'(\d+)', expand=True).astype(float)

    accepted_data_new['zip_code'] = accepted_data_new.zip_code.str.extract(
        r'(\d+)', expand=True).astype(float)
    accepted_data_new['emp_length'] = accepted_data_new.emp_length.str.extract(
        r'(\d+)', expand=True).astype(float)

    # Extract size of each accepted or rejected data
    num_of_accepted = len(accepted_data_new)
    num_of_rejected = len(rejected_data_new)

    # Remove the "loan_status" column for the concatenation of all data
    accepted_tmp = accepted_data_new[[
        'loan_amnt', 'dti', 'zip_code', 'addr_state', 'emp_length', 'fico_mean']]

    # Concatenate all data to do the numerization of the categorical data for all homogeneously
    all_data = pd.concat([accepted_tmp, rejected_data_new])

    # In[165]:
    address_state = all_data['addr_state'].unique().tolist()
    all_data['addr_state'] = all_data['addr_state'].apply(
        lambda x: address_state.index(x))

    # Separate accepted and rejected dat after homogenizing all features
    processed_data_accepted = all_data[0:num_of_accepted]
    processed_data_accepted = processed_data_accepted

    processed_data_rejected = all_data[num_of_accepted:]
    processed_data_rejected = processed_data_rejected

    # target values for calssification: loan_status column
    target = accepted_data_new.loan_status
    target = target.astype('int')

    parameters = {'max_leaf_nodes': [100,200, 300,500, 700, 1000,1500]}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters, n_jobs=njobs)
    _ = clf.fit(processed_data_accepted, target)
    
    from sklearn.metrics import f1_score, accuracy_score
    def evaluate(clf, X, y):
        return accuracy_score(y, clf.predict(X))
    print(evaluate(clf,processed_data_accepted, target))

    predictions_rejected = clf.predict_proba(
        processed_data_accepted[target == 0])
    rejected_probs = predictions_rejected[:, 1]
    c = (np.percentile(rejected_probs, 70))

    all_data__ = all_data.copy()
    cats = clf.best_estimator_.apply(all_data__)
    all_data__['cats'] = cats
    all_data__['target'] = target
    all_data__['complete_geo'] = all_data__['addr_state'].astype(
        str) + "_" + all_data__['zip_code'].astype(str)
    print(all_data.head())


    #results = Parallel(n_jobs=njobs)(delayed(lambda x: all_data__[all_data__['cats'] == x].copy())(j) for j in all_data__['cats'].unique())
    #temps = {}
    # for i, kk in zip(all_data__['cats'].unique(), results):
    #    temps[i] = kk
    dt_dist = np.percentile(all_data__['dti'],50)
    temps = {}
    for i in all_data__['cats'].unique():
        data = all_data__[all_data__['cats'] == i].copy()
        temps[i] = (all_data__[all_data__['cats'] == i].copy(), data["complete_geo"].unique(), data["dti"].mean(), data["emp_length"].mean())


    def compute_distances(temp_i, temp_j):
        dist = (
            1 - np.isin(temp_i[1],temp_j[1]).mean()) * 0.25
        dist += min(max(temp_i[2] -
                        temp_j[2], 0) / dt_dist, 1) * 0.5
        dist += (max(temp_j[3]
                     - temp_i[3], 0) / 10)*0.25
        return dist * 4.0

    distances = {}

    for i in list(temps.keys()):
        temp_i = temps[i]
        print(i)
        if i not in distances:
            distances[i] = {}
        # results = Parallel(n_jobs=njobs)(delayed(lambda x: compute_distances(
        #    temp_i, temps[j]))(j) for j in list(temps.keys()))
        # for j, d in zip(list(temps.keys()), results):
        #    distances[i][j] = d
        for j in list(temps.keys()):
            temp_j = temps[j]
            distances[i][j] = compute_distances(temp_i, temp_j)

    df_dist = pd.DataFrame(distances)
    df_dist = df_dist.reindex(sorted(df_dist.columns), axis=1)
    df_dist.to_csv(output + "_distances.csv")
    counts = all_data__.groupby(["cats"]).count()['loan_amnt']
    counts = counts / counts.sum()

    pd.DataFrame(counts).to_csv(output + "_px.csv")
    all_data__.groupby(["cats"]).mean()['target'].to_csv(
        output + "c_{c}_u.csv".format(c=c))


if __name__ == '__main__':
    experiment()
