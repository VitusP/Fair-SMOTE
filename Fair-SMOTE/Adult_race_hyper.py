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
sys.path.append(os.path.abspath('..'))

from SMOTE import smote
from Measure import measure_final_score,calculate_recall,calculate_far,calculate_precision,calculate_accuracy
from Generate_Samples import generate_samples
from DataBalance import rebalance
from sklearn import preprocessing

# dataset_orig
cr = [0.8]
f = [0.8]

dict = {
    'model': [],
    'recall+': [],
    'far-': [],
    'precision+': [],
    'accuracy+': [],
    'f1+': [],
    'aod-': [],
    'eod-': [],
    'spd-': [],
    'di-': [],
}
result = pd.DataFrame(dict)

current_cr = 0.8
current_f = 0.8
model =None
recall = None
far = None
precision = None
accuracy = None
f1 = None
aod = None
eod = None
spd = None
di = None
for i in range(10):
    ## Load dataset
    dataset_orig = pd.read_csv('../data/adult.data.csv')

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
    ##### ORIGINAL ##############
    # Check original scores
    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR
    model="Original_LSR"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]

    clf = RandomForestClassifier()
    model="Original_RFC"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]
    ###################################

    #### SMOTE #########################
    def apply_smote(df):
        df.reset_index(drop=True,inplace=True)
        cols = df.columns
        smt = smote(df)
        df = smt.run()
        df.columns = cols
        return df

    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=None,shuffle = True)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    train_df = X_train
    train_df['Probability'] = y_train

    train_df = apply_smote(train_df)

    y_train = train_df.Probability
    X_train = train_df.drop('Probability', axis = 1)

    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR
    model="SMOTE_LSR"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]
    
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=None,shuffle = True)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
    X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    train_df = X_train
    train_df['Probability'] = y_train

    train_df = apply_smote(train_df)

    y_train = train_df.Probability
    X_train = train_df.drop('Probability', axis = 1)
    clf = RandomForestClassifier()
    model="SMOTE_RFC"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]

    #################################
    ### FAIR-SMOTE ##################
    df = None
    dataset_orig_train, dataset_orig_test = None, None
    X_train, y_train = None, None
    X_test , y_test = None, None
    def regenerateFairSmote(): 
        global dataset_orig_train
        global dataset_orig_test
        global X_train 
        global y_train
        global X_test 
        global y_test

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, random_state=None, test_size=0.2,shuffle = True)

        # first one is class value and second one is protected attribute value
        zero_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)])
        zero_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)])
        one_zero = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)])
        one_one = len(dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)])

        print(zero_zero,zero_one,one_zero,one_one)
        maximum = max(zero_zero,zero_one,one_zero,one_one)
        if maximum == zero_zero:
            print("zero_zero is maximum")
        if maximum == zero_one:
            print("zero_one is maximum")
        if maximum == one_zero:
            print("one_zero is maximum")
        if maximum == one_one:
            print("one_one is maximum")

        zero_zero_to_be_incresed = maximum - zero_zero ## where both are 0
        one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
        one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

        print(zero_zero_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)

        df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
        df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

        df_zero_zero['race'] = df_zero_zero['race'].astype(str)
        df_zero_zero['sex'] = df_zero_zero['sex'].astype(str)


        df_one_zero['race'] = df_one_zero['race'].astype(str)
        df_one_zero['sex'] = df_one_zero['sex'].astype(str)

        df_one_one['race'] = df_one_one['race'].astype(str)
        df_one_one['sex'] = df_one_one['sex'].astype(str)

        df_zero_zero = generate_samples(zero_zero_to_be_incresed,df_zero_zero,'Adult', current_cr, current_f)
        df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Adult', current_cr, current_f)
        df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Adult', current_cr, current_f)

        df = df_zero_zero.append(df_one_zero)
        df = df.append(df_one_one)

        df['race'] = df['race'].astype(float)
        df['sex'] = df['sex'].astype(float)

        df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
        df = df.append(df_zero_one)

        X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
        X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

    regenerateFairSmote()
    clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR
    model="Fair_SMOTE_LSR"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]
    
    regenerateFairSmote()
    clf = LogisticRegression(C=0.615848211066026,max_iter=50,solver='liblinear') # LSR Optimized
    model="Fair_SMOTE_LSR_Optimized"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]
    
    regenerateFairSmote()
    clf = RandomForestClassifier(max_features=6, n_estimators=70)
    model="Fair_SMOTE_RFC_Optimized"
    recall=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall')
    far=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far')
    precision=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision')
    accuracy=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy')
    f1=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1')
    aod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod')
    eod=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod')
    spd=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD')
    di=measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI')
    result.loc[len(result.index)] = [model,abs(recall),abs(far),abs(precision),abs(accuracy),abs(f1),abs(aod),abs(eod),abs(spd),abs(di)]
    
result.sort_values(by = 'model').to_excel ("result_hyper/adult_race_hyper.xlsx", index = False, header=True)
