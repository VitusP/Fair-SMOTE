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

from sklearn.preprocessing import MinMaxScaler

# dataset_orig
cr = [0.8,0.6,0.4]
f = [0.8,0.6,0.4]

dict = {
    'cr_group': [],
    'f_group': [],
    'recall': [],
    'far': [],
    'precision': [],
    'accuracy': [],
    'f1': [],
    'aod': [],
    'eod': [],
    'spd': [],
    'di': [],
}
result = pd.DataFrame(dict)

current_cr = None
current_f = None
cr_group = []
f_group = []
recall = []
far = []
precision = []
accuracy = []
f1 = []
aod = []
eod = []
spd = []
di = []
for i in range(len(cr)):
    for j in range(len(f)):
        current_cr = cr[i]
        current_f = f[j]
        print("cr: ", current_cr, " AND f: ", current_f)

        n = 5
        for k in range(n):
            ## Load dataset
            dataset_orig = pd.read_csv('../data/compas-scores-two-years.csv')

            ## Drop categorical features
            ## Removed two duplicate coumns - 'decile_score','priors_count'
            dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date',
                                            'dob','age','juv_fel_count','decile_score',
                                            'juv_misd_count','juv_other_count','days_b_screening_arrest',
                                            'c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date',
                                            'c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree',
                                            'r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out',
                                            'violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date',
                                            'vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date',
                                            'v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody',
                                            'out_custody','start','end','event'],axis=1)

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


            protected_attribute = 'sex'

            ## Rename class column
            dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)

            # Here did not rec means 0 is the favorable lable
            dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 0, 1, 0)

            scaler = MinMaxScaler()
            dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)
            dataset_orig

            dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.2, random_state=None,shuffle = True)
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

            zero_one_to_be_incresed = maximum - zero_one ## where class is 0 attribute is 1
            one_zero_to_be_incresed = maximum - one_zero ## where class is 1 attribute is 0
            one_one_to_be_incresed = maximum - one_one ## where class is 1 attribute is 1

            print(zero_one_to_be_incresed,one_zero_to_be_incresed,one_one_to_be_incresed)
            
            df_zero_one = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 1)]
            df_one_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 0)]
            df_one_one = dataset_orig_train[(dataset_orig_train['Probability'] == 1) & (dataset_orig_train[protected_attribute] == 1)]

            df_zero_one['race'] = df_zero_one['race'].astype(str)
            df_zero_one['sex'] = df_zero_one['sex'].astype(str)


            df_one_zero['race'] = df_one_zero['race'].astype(str)
            df_one_zero['sex'] = df_one_zero['sex'].astype(str)

            df_one_one['race'] = df_one_one['race'].astype(str)
            df_one_one['sex'] = df_one_one['sex'].astype(str)


            df_zero_one = generate_samples(zero_one_to_be_incresed,df_zero_one,'Compas', current_cr, current_f)
            df_one_zero = generate_samples(one_zero_to_be_incresed,df_one_zero,'Compas', current_cr, current_f)
            df_one_one = generate_samples(one_one_to_be_incresed,df_one_one,'Compas', current_cr, current_f)

            df = df_zero_one.append(df_one_zero)
            df = df.append(df_one_one)

            df['race'] = df['race'].astype(float)
            df['sex'] = df['sex'].astype(float)

            df_zero_zero = dataset_orig_train[(dataset_orig_train['Probability'] == 0) & (dataset_orig_train[protected_attribute] == 0)]
            df = df.append(df_zero_zero)

            X_train, y_train = df.loc[:, df.columns != 'Probability'], df['Probability']
            X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

            clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR

            # print("recall :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
            # print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
            # print("precision :", measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
            # print("accuracy :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
            # print("F1 Score :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
            # print("aod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
            # print("eod :"+protected_attribute,measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))

            # print("SPD:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
            # print("DI:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))
            cr_group.append(current_cr)
            f_group.append(current_f)
            recall.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'recall'))
            far.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'far'))
            precision.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'precision'))
            accuracy.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'accuracy'))
            f1.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'F1'))
            aod.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'aod'))
            eod.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'eod'))
            spd.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'SPD'))
            di.append(measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, protected_attribute, 'DI'))

        # Calculate median
        result.loc[len(result.index)] = [current_cr,current_f,abs(np.median(recall)),abs(np.median(far)),abs(np.median(precision)),abs(np.median(accuracy)),abs(np.median(f1)),abs(np.median(aod)),abs(np.median(eod)),abs(np.median(spd)),abs(np.median(di))]
        print("=======================================================================")
        print("CR and F experiment value: ", current_cr, " and ", current_f )
        print("recall :", np.median(recall))
        print("far :",np.median(far))
        print("precision :",np.median(precision))
        print("accuracy :",np.median(accuracy))
        print("F1 Score :",np.median(f1))
        print("aod :"+protected_attribute,np.median(aod))
        print("eod :"+protected_attribute,np.median(eod))
        print("SPD:",np.median(spd))
        print("DI:",np.median(di))
        print("=======================================================================")
    print("=======================================================================")
    print("CR and F experiment value: ", cr_group, " and ", f_group )
    print("recall :", recall)
    print("far :",far)
    print("precision :",precision)
    print("accuracy :",accuracy)
    print("F1 Score :",f1)
    print("aod :"+protected_attribute,aod)
    print("eod :"+protected_attribute,eod)
    print("SPD:",spd)
    print("DI:",di)
    print("=======================================================================")
    print(result)
    result.to_excel ("result/compass_sex_crf.xlsx", index = False, header=True)
