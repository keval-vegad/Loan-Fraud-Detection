# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:20:17 2020

@author: keval
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('LoanStats_2017Q2.csv')

dataset1 = dataset[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','installment','grade','sub_grade','emp_title','emp_length','home_ownership',
                    'annual_inc','verification_status','issue_d','loan_status','pymnt_plan','purpose','title','zip_code','addr_state','dti','delinq_2yrs',
                    'earliest_cr_line','inq_last_6mths','mths_since_last_delinq','mths_since_last_record','open_acc','pub_rec','revol_bal','revol_util','total_acc',
                    'initial_list_status','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','total_rec_late_fee',
                    'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog',
                    'policy_code','application_type','annual_inc_joint','dti_joint','verification_status_joint','acc_now_delinq','tot_coll_amt',
                    'tot_cur_bal','open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m',
                    'max_bal_bc','all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m','acc_open_past_24mths','avg_cur_bal',
                    'bc_open_to_buy','bc_util','chargeoff_within_12_mths','delinq_amnt','mo_sin_old_il_acct','mo_sin_old_rev_tl_op','mo_sin_rcnt_rev_tl_op',
                    'mo_sin_rcnt_tl','mort_acc','mths_since_recent_bc','mths_since_recent_bc_dlq','mths_since_recent_inq','mths_since_recent_revol_delinq',
                    'num_accts_ever_120_pd','num_actv_bc_tl','num_actv_rev_tl','num_bc_sats','num_bc_tl','num_il_tl','num_op_rev_tl','num_rev_accts',
                    'num_rev_tl_bal_gt_0','num_sats','num_tl_120dpd_2m','num_tl_30dpd','num_tl_90g_dpd_24m','num_tl_op_past_12m','pct_tl_nvr_dlq',
                    'percent_bc_gt_75','pub_rec_bankruptcies','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit',
                    'total_il_high_credit_limit','revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths','sec_app_mort_acc',
                    'sec_app_open_acc','sec_app_revol_util','sec_app_open_il_6m','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths',
                    'sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog']]


dataset1 = dataset1.replace([' ','NULL'],np.nan)
print(dataset1)


dataset2 = dataset1.loc[:, dataset1.isin([' ','NULL',0]).mean() < 0.6] # drop column which has empty values more than 60%

print (dataset2)
dataset2.describe()
dataset2.head()

#Count the Null Columns
null_columns = dataset2.columns[dataset2.isnull().any()]
sum_num_missing = dataset2[null_columns].isnull().sum()
len(dataset2)
percentage_missing_values =  (sum_num_missing/len(dataset2))*100
print(percentage_missing_values>60)
percentage_missing_values[percentage_missing_values>60]
need_to_delete = percentage_missing_values>60
need_to_delete[need_to_delete==True]


#deleteing the column which has null values greater than 60%
dataset4 = dataset2.drop(['mths_since_last_record', 'mths_since_last_major_derog','annual_inc_joint','dti_joint','verification_status_joint',
          'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq','revol_bal_joint','sec_app_earliest_cr_line','sec_app_inq_last_6mths',
          'sec_app_mort_acc','sec_app_open_acc','sec_app_revol_util','sec_app_open_il_6m','sec_app_num_rev_accts','sec_app_chargeoff_within_12_mths',
          'sec_app_collections_12_mths_ex_med','sec_app_mths_since_last_major_derog'],axis=1)

#Now need to find rows with duplicate values
duplicate_rows_df = dataset4[dataset4.duplicated()]
dataset4.drop_duplicates().shape
dataset4 = dataset4.drop([105451, 105452, 105453, 105454]) 

print(dataset4)

#Now need to find rows with duplicate values
dataset4.T.drop_duplicates().T

dataset4 = dataset4.drop(['funded_amnt', 'funded_amnt_inv'], axis=1) # drope these column with same values

dataset4 = dataset4.drop(['out_prncp_inv'], axis=1)

dataset4 = dataset4.drop(['total_pymnt_inv'], axis=1)

#list unique value in column
dataset4.issue_d.unique()
# it has only these 3 value which wont so drop that column array(['Jun-17', 'May-17', 'Apr-17', nan], dtype=object)
dataset4 = dataset4.drop(['issue_d'], axis=1)

#list unique value in column
dataset4.pymnt_plan.unique()
len(dataset4['pymnt_plan'].unique().tolist())
len(dataset4[dataset4['pymnt_plan'] == 'n']) # there are 105454 entries of 'n'
dataset4 = dataset4.drop(['pymnt_plan'], axis=1)


dataset4.initial_list_status.unique()
len(dataset4['initial_list_status'].unique().tolist())
len(dataset4[dataset4['initial_list_status'] == 'f']) # there are 79488 entries of 'w' and 25963 values of 'f'


dataset4.next_pymnt_d.unique()
len(dataset4['next_pymnt_d'].unique().tolist())
len(dataset4[dataset4['next_pymnt_d'] == 'Oct-17'])# there are 101494 entries of 'Sep-17' and 6 values of 'Aug-17', 30 values of Oct-17'
dataset4 = dataset4.drop(['next_pymnt_d'], axis=1)





dataset4.verification_status.unique()
len(dataset4['verification_status'].unique().tolist())
len(dataset4[dataset4['verification_status'] == 'Source Verified'])# #keep it

dataset4.last_pymnt_d.unique()
len(dataset4['last_pymnt_d'].unique().tolist())
len(dataset4[dataset4['last_pymnt_d'] == 'Apr-17'])# Aug-17: 100535, Jul: 2602, Jun:1264. May:758, Apr:147
dataset4 = dataset4.drop(['last_pymnt_d'], axis=1)


dataset4.last_credit_pull_d.unique()
len(dataset4['last_credit_pull_d'].unique().tolist())
len(dataset4[dataset4['last_credit_pull_d'] == 'nan']) #Apr-17:1166, Jul:1049, Jun:1445, May:1560, Apr:1166, Mar:312
dataset4 = dataset4.drop(['last_credit_pull_d'], axis=1)


dataset4.policy_code.unique()
len(dataset4['policy_code'].unique().tolist())
len(dataset4[dataset4['policy_code'] == 'nan']) 
dataset4 = dataset4.drop(['policy_code'], axis=1)

dataset4.application_type.unique()
len(dataset4['application_type'].unique().tolist())
len(dataset4[dataset4['application_type'] == 'DIRECT_PAY']) #INDIVIDUAL:98619, JOINT:6813, DIRECT_PAY:19
dataset4 = dataset4.drop(['application_type'], axis=1)#Drop it


dataset4.percent_bc_gt_75.unique()
len(dataset4['percent_bc_gt_75'].unique().tolist())
len(dataset4[dataset4['percent_bc_gt_75'] == 'DIRECT_PAY']) #Keep it


dataset4.total_il_high_credit_limit.unique()
len(dataset4['total_il_high_credit_limit'].unique().tolist()) #Keep it

dataset4.term.unique()
len(dataset4['term'].unique().tolist())
len(dataset4[dataset4['term'] == ' 60 months']) #36 months: 77105, 60 months: 28346
dataset4 = dataset4.drop(['term'], axis=1)#drop it

dataset4 = dataset4.drop(['sub_grade'], axis=1)#drop it

dataset4.initial_list_status.unique()
len(dataset4[dataset4['initial_list_status'] == 'f'])#w:79488, f:25963
dataset4 = dataset4.drop(['initial_list_status'], axis=1)#drop it



dataset4_numeric  = dataset4.drop(['grade','emp_title','emp_length','home_ownership','verification_status','loan_status',
                                  'purpose','title','zip_code','addr_state','earliest_cr_line'],axis = 1)

dataset4_categorical = dataset4[['grade','emp_title','emp_length','home_ownership','verification_status','loan_status',
                                  'purpose','title','zip_code','addr_state','earliest_cr_line']]


pd.DataFrame(dataset4_numeric).std()

#drop column which has standard deviation less than 10
threshold = 10

dataset4_numeric = dataset4_numeric.drop(dataset4_numeric.std()[dataset4_numeric.std() < threshold].index.values, axis=1)


# dealing with the missing values and replacing it by median 
#from sklearn.impute import SimpleImputer
#
#imputer = SimpleImputer(strategy="median")
#
#imputer.fit(dataset4_numeric) 

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit_transform(dataset4_numeric)

dataset4_numeric.fillna(dataset4_numeric.mean(), inplace=True)


print(dataset4_numeric.isnull().values.sum())


# dealing with the missing values and replacing it by mod for categorical data 
print(dataset4_categorical.isnull().values.sum())

print(dataset4_categorical.isnull().sum())

"""
the chaining of method .value_counts() in the code below. This returns the frequency distribution of each category in the feature, 
and then selecting the top category, 
which is the mode, with the .index attribute
"""
dataset4_categorical = dataset4_categorical.fillna(dataset4_categorical['emp_title'].value_counts().index[0])
dataset4_categorical = dataset4_categorical.fillna(dataset4_categorical['emp_length'].value_counts().index[0])

print(dataset4_categorical.isnull().sum())


#WE NEED TO REDUCED COLUMN BY STANDARD DEVIATION!!!

#frequency distribution of categories within the feature
print(dataset4_categorical['grade'].value_counts())
print(dataset4_categorical['grade'].value_counts().count())

"""
Below is a basic template to plot a barplot of the frequency distribution of a categorical feature using the seaborn package, 
which shows the frequency distribution of the grade column. 
"""

import seaborn as sns
import matplotlib.pyplot as plt
grade_count = dataset4_categorical['grade'].value_counts()
sns.set(style="darkgrid")
sns.barplot(grade_count.index, grade_count.values, alpha=0.9)
plt.title('Frequency Distribution of GRADE')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('GRADE', fontsize=12)
plt.show()



#change column type into category
for col in ['grade', 'emp_title', 'emp_length', 'home_ownership', 'verification_status', 'loan_status','purpose','title','zip_code','addr_state',
            'earliest_cr_line']:
    dataset4_categorical[col] = dataset4_categorical[col].astype('category')
    
    
dataset4_categorical['grade'] = dataset4_categorical['grade'].cat.codes
dataset4_categorical['emp_title'] = dataset4_categorical['emp_title'].cat.codes

dataset4.emp_length.unique()
dataset4_categorical['emp_length'] = dataset4_categorical['emp_length'].cat.codes

dataset4_categorical['home_ownership'] = dataset4_categorical['home_ownership'].cat.codes
dataset4_categorical['verification_status'] = dataset4_categorical['verification_status'].cat.codes
dataset4_categorical['loan_status'] = dataset4_categorical['loan_status'].cat.codes
dataset4_categorical['purpose'] = dataset4_categorical['purpose'].cat.codes
dataset4_categorical['title'] = dataset4_categorical['title'].cat.codes
dataset4_categorical['zip_code'] = dataset4_categorical['zip_code'].cat.codes
dataset4_categorical['addr_state'] = dataset4_categorical['addr_state'].cat.codes
dataset4_categorical['earliest_cr_line'] = dataset4_categorical['earliest_cr_line'].cat.codes


dataset4_categorical.earliest_cr_line.unique()
len(dataset4_categorical['earliest_cr_line'].unique().tolist())


#column concating
df = pd.concat([dataset4_numeric, dataset4_categorical], axis=1)

from openpyxl.workbook import Workbook

X = df.loc[:,df.columns != 'loan_status'].values
y = df['loan_status'].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#-------------------------APPLYING KNN ALGO-------------------------------------------#
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN.fit(X_train, y_train)

# Predicting the Test set results
y_pred_KNN = classifier_KNN.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN = confusion_matrix(y_test, y_pred_KNN)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_KNN, X_train, y_train, cv=3, scoring="accuracy")
#Out[31]: array([0.97411095, 0.97368421,     ]) approx 97% of accuracy
import pickle

# # save the model to disk
# filename_KNN = 'KNN_model.sav'
# pickle.dump(classifier_KNN, open(filename_KNN, 'wb'))
 
# # some time later...
 
# # load the model from disk
# loaded_classifier_KNN = pickle.load(open(filename_KNN, 'rb'))
# result = loaded_classifier_KNN.score(X_test, y_test)
# print(result)

# # predict probabilities for test set
# yhat_probs = classifier_KNN.predict(X_test, verbose=0)
# # predict crisp classes for test set
# yhat_classes = classifier_KNN.predict_classes(X_test, verbose=0)
# # reduce to 1d array
# yhat_probs = yhat_probs[:, 0]
# yhat_classes = yhat_classes[:, 0]

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_KNN)
# Out[14]: 
# array([[    0,     9,     0,     0,     0,     0],
#        [    0, 19943,    21,     0,     0,     0],
#        [    0,   174,   601,     0,     0,     0],
#        [    0,   191,     0,     0,     0,     0],
#        [    0,    55,     0,     0,     0,     0],
#        [    0,    96,     1,     0,     0,     0]], dtype=int64)

from sklearn.metrics import classification_report
# print(classification_report(y_test,y_pred_KNN))
#               precision    recall  f1-score   support

#            0       0.00      0.00      0.00         9
#            1       0.97      1.00      0.99     19964
#            2       0.96      0.78      0.86       775
#            3       0.00      0.00      0.00       191
#            4       0.00      0.00      0.00        55
#            5       0.00      0.00      0.00        97

#     accuracy                           0.97     21091
#    macro avg       0.32      0.30      0.31     21091
# weighted avg       0.96      0.97      0.97     21091



# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# X_knn = df[['total_pymnt','loan_amnt']]
# Training the K-Means model on the dataset
from sklearn.cluster import KMeans 
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

from sklearn.metrics import classification_report
print(classification_report(y,y_kmeans))
#               precision    recall  f1-score   support

#            0       0.00      0.04      0.00        25
#            1       0.95      0.46      0.62     99850
#            2       0.04      0.00      0.01      3896
#            3       0.01      0.10      0.02       932
#            4       0.00      0.23      0.01       312
#            5       0.00      0.15      0.01       436

#     accuracy                           0.43    105451
#    macro avg       0.17      0.16      0.11    105451
# weighted avg       0.90      0.43      0.58    105451

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, c = 'orange', label = 'Cluster 6')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of loan status')
plt.xlabel('Annual Income')
plt.ylabel('loan status)')
plt.legend()
plt.show()


#feature selection
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


#Correlation with output variable
cor_target = abs(cor["loan_status"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.1]
relevant_features




X1 = df.loc[:,df.columns != 'loan_status']
y1 = df['loan_status']
#Adding constant column of ones, mandatory for sm.OLS model
import statsmodels.api as sm
X_1 = sm.add_constant(X1)
#Fitting sm.OLS model
model = sm.OLS(y1,X_1).fit()
model.pvalues

#Backward Elimination
cols = list(X1.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X1[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y1,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)

# ['loan_amnt', 'installment', 'revol_bal', 'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp', 
#  'total_rec_int', 'last_pymnt_amnt', 'mths_since_rcnt_il', 'il_util', 'max_bal_bc', 'avg_cur_bal', 
#  'bc_util', 'mo_sin_rcnt_rev_tl_op', 'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort', 
#  'total_il_high_credit_limit', 'grade', 'verification_status', 'purpose']

X_reduced = df[['loan_amnt', 'installment', 'revol_bal', 'total_acc', 'out_prncp', 'total_pymnt', 'total_rec_prncp',
       'total_rec_int', 'last_pymnt_amnt', 'mths_since_rcnt_il', 'il_util', 'max_bal_bc', 'avg_cur_bal',
       'bc_util', 'mo_sin_rcnt_rev_tl_op', 'percent_bc_gt_75', 'tot_hi_cred_lim', 'total_bal_ex_mort', 
       'total_il_high_credit_limit', 'grade', 'verification_status', 'purpose']].values
y_reduced = df['loan_status'].values


from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_reduced, y_reduced, test_size=0.20, random_state=42)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train1 = sc_X.fit_transform(X_train1)
X_test1 = sc_X.transform(X_test1)
    
# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_KNN1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_KNN1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_KNN1 = classifier_KNN1.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_KNN1 = confusion_matrix(y_test1, y_pred_KNN1)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_KNN1, X_train1, y_train1, cv=3, scoring="accuracy")
# Out[40]: array([0.97788051, 0.97784495, 0.97841394])

# #-------------------------APPLYING SVM ALGO-------------------------------------------#
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_SVM = SVC(kernel = 'linear', random_state = 0)
classifier_SVM.fit(X_train, y_train)

# Predicting the Test set results
y_pred_SVM = classifier_SVM.predict(X_test)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_SVM, X_train, y_train, cv=3, scoring="accuracy")
#Out[53]: array([0.98374822, 0.98385491, 0.98406828])


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_SVM)
# Out[45]: 
# array([[    5,     4,     0,     0,     0,     0],
#        [    0, 19949,    15,     0,     0,     0],
#        [    0,     0,   775,     0,     0,     0],
#        [    0,   191,     0,     0,     0,     0],
#        [    0,    55,     0,     0,     0,     0],
#        [    0,    79,     1,     0,     0,    17]], dtype=int64)

from sklearn.metrics import classification_report
# print(classification_report(y_test,y_pred_SVM))
#               precision    recall  f1-score   support

#            0       1.00      0.56      0.71         9
#            1       0.98      1.00      0.99     19964
#            2       0.98      1.00      0.99       775
#            3       0.00      0.00      0.00       191
#            4       0.00      0.00      0.00        55
#            5       1.00      0.18      0.30        97

#     accuracy                           0.98     21091
#    macro avg       0.66      0.46      0.50     21091
# weighted avg       0.97      0.98      0.98     21091


#-------------with reduced columns-------------------------#
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier_SVM1 = SVC(kernel = 'linear', random_state = 0)
classifier_SVM1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_SVM1 = classifier_SVM1.predict(X_test1)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_SVM1, X_train1, y_train1, cv=3, scoring="accuracy")
# Out[44]: array([0.98381935, 0.98392603, 0.98421053])


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_SVM)
# Out[45]: 
# array([[    5,     4,     0,     0,     0,     0],
#        [    0, 19949,    15,     0,     0,     0],
#        [    0,     0,   775,     0,     0,     0],
#        [    0,   191,     0,     0,     0,     0],
#        [    0,    55,     0,     0,     0,     0],
#        [    0,    79,     1,     0,     0,    17]], dtype=int64)

from sklearn.metrics import classification_report
# print(classification_report(y_test,y_pred_SVM))
#               precision    recall  f1-score   support

#            0       1.00      0.56      0.71         9
#            1       0.98      1.00      0.99     19964
#            2       0.98      1.00      0.99       775
#            3       0.00      0.00      0.00       191
#            4       0.00      0.00      0.00        55
#            5       1.00      0.18      0.30        97

#     accuracy                           0.98     21091
#    macro avg       0.66      0.46      0.50     21091
# weighted avg       0.97      0.98      0.98     21091


# #-------------------------APPLYING Decision Tree ALGO-------------------------------------------#
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT.fit(X_train, y_train)

# Predicting the Test set results
y_pred_DT = classifier_DT.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT = confusion_matrix(y_test, y_pred_DT)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_DT, X_train, y_train, cv=3, scoring="accuracy")
#array([0.9688478 , 0.96881223, 0.96984353])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_DT)
# Out[49]: 
# array([[    9,     0,     0,     0,     0,     0],
#        [    0, 19616,     2,   206,    75,    65],
#        [    0,     7,   766,     2,     0,     0],
#        [    0,   181,     0,     3,     2,     5],
#        [    0,    53,     0,     1,     0,     1],
#        [    0,    62,     1,     2,     2,    30]], dtype=int64)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_DT))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       0.98      0.98      0.98     19964
#            2       1.00      0.99      0.99       775
#            3       0.01      0.02      0.01       191
#            4       0.00      0.00      0.00        55
#            5       0.30      0.31      0.30        97

#     accuracy                           0.97     21091
#    macro avg       0.55      0.55      0.55     21091
# weighted avg       0.97      0.97      0.97     21091


#-----------------
# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_DT1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_DT1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_DT1 = classifier_DT1.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_DT1 = confusion_matrix(y_test1, y_pred_DT1)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_DT1, X_train1, y_train1, cv=3, scoring="accuracy")
#Out[49]: array([0.97002134, 0.96888336, 0.9693101 ])
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_DT)
# Out[49]: 
# array([[    9,     0,     0,     0,     0,     0],
#        [    0, 19616,     2,   206,    75,    65],
#        [    0,     7,   766,     2,     0,     0],
#        [    0,   181,     0,     3,     2,     5],
#        [    0,    53,     0,     1,     0,     1],
#        [    0,    62,     1,     2,     2,    30]], dtype=int64)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_DT))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00         9
#            1       0.98      0.98      0.98     19964
#            2       1.00      0.99      0.99       775
#            3       0.01      0.02      0.01       191
#            4       0.00      0.00      0.00        55
#            5       0.30      0.31      0.30        97

#     accuracy                           0.97     21091
#    macro avg       0.55      0.55      0.55     21091
# weighted avg       0.97      0.97      0.97     21091
##-----------------RandomForest----------------------------####
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF.fit(X_train, y_train)

# Predicting the Test set results
y_pred_RF = classifier_RF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF = confusion_matrix(y_test, y_pred_RF)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_RF, X_train, y_train, cv=3, scoring="accuracy")
#array([0.98495733, 0.9851707 , 0.98538407])


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_RF)
# Out[53]: 
# array([[    4,     0,     5,     0,     0,     0],
#        [    0, 19962,     2,     0,     0,     0],
#        [    0,     0,   775,     0,     0,     0],
#        [    0,   191,     0,     0,     0,     0],
#        [    0,    55,     0,     0,     0,     0],
#        [    0,    69,     1,     0,     0,    27]], dtype=int64)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RF))
#               precision    recall  f1-score   support

#            0       1.00      0.44      0.62         9
#            1       0.98      1.00      0.99     19964
#            2       0.99      1.00      0.99       775
#            3       0.00      0.00      0.00       191
#            4       0.00      0.00      0.00        55
#            5       1.00      0.28      0.44        97

#     accuracy                           0.98     21091
#    macro avg       0.66      0.45      0.51     21091
# weighted avg       0.97      0.98      0.98     21091

# ------------------------------------------------
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_RF1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_RF1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_RF1 = classifier_RF1.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_RF1 = confusion_matrix(y_test1, y_pred_RF1)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_RF1, X_train1, y_train1, cv=3, scoring="accuracy")
#Out[54]: array([0.98492176, 0.98492176, 0.9851707 ])


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_RF)
# Out[53]: 
# array([[    4,     0,     5,     0,     0,     0],
#        [    0, 19962,     2,     0,     0,     0],
#        [    0,     0,   775,     0,     0,     0],
#        [    0,   191,     0,     0,     0,     0],
#        [    0,    55,     0,     0,     0,     0],
#        [    0,    69,     1,     0,     0,    27]], dtype=int64)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_RF))
#               precision    recall  f1-score   support

#            0       1.00      0.44      0.62         9
#            1       0.98      1.00      0.99     19964
#            2       0.99      1.00      0.99       775
#            3       0.00      0.00      0.00       191
#            4       0.00      0.00      0.00        55
#            5       1.00      0.28      0.44        97

#     accuracy                           0.98     21091
#    macro avg       0.66      0.45      0.51     21091
# weighted avg       0.97      0.98      0.98     21091
##-----------------Naive Bayes----------------------
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_NB = GaussianNB()
classifier_NB.fit(X_train, y_train)

# Predicting the Test set results
y_pred_NB = classifier_NB.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB = confusion_matrix(y_test, y_pred_NB)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_NB, X_train, y_train, cv=3, scoring="accuracy")
#array([0.9413229 , 0.91987909, 0.93602418])

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
confusion_matrix(y_test,y_pred_NB)
# Out[56]: 
# array([[    7,     1,     1,     0,     0,     0],
#        [    0, 19047,     2,   238,   102,   575],
#        [    2,     4,   767,     2,     0,     0],
#        [    0,   173,     0,     3,     0,    15],
#        [    0,    51,     0,     1,     0,     3],
#        [    0,    81,     1,     1,     0,    14]], dtype=int64)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_NB))
#            precision    recall  f1-score   support

#            0       0.78      0.78      0.78         9
#            1       0.98      0.95      0.97     19964
#            2       0.99      0.99      0.99       775
#            3       0.01      0.02      0.01       191
#            4       0.00      0.00      0.00        55
#            5       0.02      0.14      0.04        97

#     accuracy                           0.94     21091
#    macro avg       0.47      0.48      0.47     21091
# weighted avg       0.97      0.94      0.95     21091


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier_NB1 = GaussianNB()
classifier_NB1.fit(X_train1, y_train1)

# Predicting the Test set results
y_pred_NB1 = classifier_NB1.predict(X_test1)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_NB1 = confusion_matrix(y_test1, y_pred_NB1)

#IMPLEMENTING CROSS-VALIDATION using cross_val_score() function 
from sklearn.model_selection import cross_val_score
cross_val_score(classifier_NB1, X_train1, y_train1, cv=3, scoring="accuracy")
# Out[55]: array([0.96337127, 0.9527027 , 0.96500711])

