import pandas as pd
import random,time,csv
import numpy as np
import math,copy,os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics


import sys
sys.path.append(os.path.abspath('.'))

from SMOTE import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples
from DataBalanceExtended import rebalance, buildRatioMap

from sklearn import preprocessing
dataset_orig = pd.read_csv('./data/adult.data.csv')

## Drop NULL values
dataset_orig = dataset_orig.dropna()

## Drop categorical features
dataset_orig = dataset_orig.drop(['workclass','fnlwgt','education','marital-status','occupation','relationship','native-country'],axis=1)

## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == ' Male', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != ' White', 0, 1)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == ' <=50K', 0, 1)


## Discretize age
dataset_orig['age'] = np.where(dataset_orig['age'] >= 70, 70, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 60 ) & (dataset_orig['age'] < 70), 60, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 50 ) & (dataset_orig['age'] < 60), 50, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 40 ) & (dataset_orig['age'] < 50), 40, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 30 ) & (dataset_orig['age'] < 40), 30, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 20 ) & (dataset_orig['age'] < 30), 20, dataset_orig['age'])
dataset_orig['age'] = np.where((dataset_orig['age'] >= 10 ) & (dataset_orig['age'] < 10), 10, dataset_orig['age'])
dataset_orig['age'] = np.where(dataset_orig['age'] < 10, 0, dataset_orig['age'])

protected_attribute = 'race'

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)


dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2,shuffle = True)

t = buildRatioMap([protected_attribute])

cols = list(t[0].keys()) + ['recall', 'far', 'precision', 'accuracy', 'F1 Score', 'aod', 'eod', 'SPD', 'DI']
rows = []

for i,j in enumerate(t):
    
    print(f"processing {i + 1} out of {len(t)}")
    
    temp = rebalance(dataset_orig_train, 'Adult', ['race', 'sex'], [protected_attribute], j)
    df = temp
    X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR
    
    new_row = list(j.values()) + [measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'),
                                    measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')]
    
    rows.append(new_row)


df = pd.DataFrame(rows,columns=cols)
df.to_csv('adult_race_comb', sep='\t', encoding='utf-8')