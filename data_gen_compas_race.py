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

## Load dataset
dataset_orig = pd.read_csv('./data/compas-scores-two-years.csv')



## Drop categorical features
## Removed two duplicate coumns - 'decile_score','priors_count'
dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()


## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)

protected_attribute = 'race'

## Rename class column
dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

# Here did not rec means 0 is the favorable lable
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=0,shuffle = True)


t = buildRatioMap([protected_attribute])

cols = list(t[0].keys()) + ['recall', 'far', 'precision', 'accuracy', 'F1 Score', 'aod', 'eod', 'SPD', 'DI']
rows = []

for i,j in enumerate(t):
    
    print(f"processing {i + 1} out of {len(t)}")
    
    temp = rebalance(dataset_orig_train, 'Compas', ['race', 'sex'], [protected_attribute], j)
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
df.to_csv('compas_race_comb.csv', encoding='utf-8')