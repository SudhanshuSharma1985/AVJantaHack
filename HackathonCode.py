# -*- coding: utf-8 -*-
"""


"""

import pandas as pd
import sklearn as sk
import numpy as np, os
from pandasql import *
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import os
import pickle
import numpy as np
import lightgbm as lgb
from matplotlib import pyplot
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
label_encoder = LabelEncoder()
from sklearn.model_selection import GridSearchCV
import seaborn as sns

                                        
os.chdir(r"C:\Sudhanshu\PythonTraining\AVhackathon")                                            

input_file_path="C:\Sudhanshu\PythonTraining\AVhackathon\\"
output_file_path="C:\Sudhanshu\PythonTraining\AVhackathon\\"

input_file_name="train_fNxu4vz.csv"
in_f=input_file_path+input_file_name

input_file_name1="test_fjtUOL8.csv"
in_f1=input_file_path+input_file_name1

#Reading csv data file

TrainData = pd.read_csv(in_f)
TrainData['Loan_Amount_Requested'] = TrainData['Loan_Amount_Requested'].str.replace(',','')
TrainData['Loan_Amount_Requested'] = TrainData['Loan_Amount_Requested'].astype(int)

TestData = pd.read_csv(in_f1)
TestData['Loan_Amount_Requested'] = TestData['Loan_Amount_Requested'].str.replace(',','')
TestData['Loan_Amount_Requested'] = TestData['Loan_Amount_Requested'].astype(int)
TrainData['type'] = 'Train'
TestData['type'] = 'Test'
TestData['Interest_Rate'] = np.nan


def IncomeBinning(bin):
    if 3000<=bin <= 10000:
        return '4k-10k'
    if 10000<bin <= 20000:
        return '10k-20k'
    if  20000<= bin <= 30000 :
        return '20k-30k'
    if 30000 < bin <= 45000:
        return '30k-45k'
    if  45000 < bin<= 63000:
        return '45k-63k'
    if  63000 < bin<=89000:
        return '63k-89k'
    if  89000 < bin:
        return '89k+'

def LoanAmountBinning(bin):
    if 500<=bin <= 8000:
        return '.5k-8k'
    if 8001<bin <= 12500:
        return '8k-12.5k'
    if  12501<= bin <= 20000 :
        return '12k-20k'
    if 20000 < bin:
        return '20+'



Data = pd.concat([TrainData,TestData],axis =0)


df = Data
#Data summary for all variables
column_name = df.columns.values.tolist()
data_summary=pd.DataFrame(columns=('VariableName','Count','Count_distinct','Count_missing'))
i=0     
for col_i in column_name:
#    Counting distinct value for each variables
#    x=pd.Series.value_counts(df[col_i]).to_frame()
#    distinct_count=x.size
    distinct_count=df[col_i].nunique()
    #    Counting missing value for each variables
    missing_count=df[col_i].isnull().sum()
    data_summary.loc[i]=[col_i,len(df.index),distinct_count,missing_count]
    i=i+1
#End of the loop
result=data_summary.sort_values('Count_missing',ascending=False)
result.to_excel(r"DataSummary.xlsx")

#Exporting summary to excel
out_file=output_file_path+'data_summary_dm_Referrer.xlsx'
result.to_excel(out_file,sheet_name='Data_summary')





#Creating data understanding for numeric variables
numeric=df.describe()
num_T=numeric.T
num_T.index.name='VariableName'
#Exporting numeric variable details to excel
out_file=output_file_path+'numeric_dm_Referrer.xlsx'
num_T.to_excel(out_file,sheet_name='numeric_summary')


#Creating data understanding for character variables

#Selecting only string data type variables
char_var_list=list(df.select_dtypes(include=['O']).columns)

char_var_list=(df.select_dtypes(include=['O']).columns).tolist()


#Creating excel file to add multiple sheets

out_file=output_file_path+'char_dm_Referrer.xlsx'
writer = pd.ExcelWriter(out_file, engine='openpyxl')
for col_i in char_var_list:
    char=pd.Series.value_counts(df[col_i],dropna=False).to_frame()
    char.index.name=col_i
    char.columns=['count']
#Since excel sheet name have limit of 31 character so truncating the variable name
    if len(col_i)>31:
        sheet_name=col_i[:31]
    else:
        sheet_name=col_i
    char.to_excel(writer,sheet_name=sheet_name)
        
writer.save()   

#End of character data understanding output

##EDA

Data['Interest_Rate'].value_counts(normalize=True)
sns.countplot(Data["Interest_Rate"])


plt.figure(figsize=(24, 6))
plt.subplot(131)
sns.countplot(Data['Home_Owner'],order = Data['Home_Owner'].value_counts(dropna=False).index)

plt.figure(figsize=(24, 6))
plt.subplot(132)
sns.countplot(Data['Income_Verified'],order = Data['Income_Verified'].value_counts(dropna=False).index)

plt.figure(figsize=(24, 6))
plt.subplot(133)
sns.countplot(Data['Gender'],order = Data['Gender'].value_counts(dropna=False).index)

plt.figure(figsize=(24, 6))
sns.countplot(Data['Purpose_Of_Loan'],order = Data['Purpose_Of_Loan'].value_counts(dropna=False).index)

##bivariate
sns.boxplot(x='Interest_Rate',y='Months_Since_Deliquency', data = Data)
plt.show()

##bivariate
sns.boxplot(x='Interest_Rate',y='Loan_Amount_Requested', data = Data)
plt.show()

##bivariate
sns.boxplot(x='Interest_Rate',y='Inquiries_Last_6Mo', data = Data)
plt.show()

##bivariate
sns.boxplot(x='Interest_Rate',y='Total_Accounts', data = Data)
plt.show()

##Impute the data

Data = pd.concat([TrainData,TestData],axis =0)


from sklearn_pandas import CategoricalImputer
imputer = CategoricalImputer()



# imputing the missing values from the column

Data['Home_Owner']=imputer.fit_transform(Data['Home_Owner'])
Data['Length_Employed']=imputer.fit_transform(Data['Length_Employed'])
Data['Months_Since_Deliquency'].fillna(0, inplace=True)
Data['Annual_Income'].fillna(Data['Annual_Income'].mean(), inplace=True)

Data['Debt_amount'] = Data['Debt_To_Income']*(Data['Annual_Income']/12)
Data['Debt_amount'].fillna(Data['Debt_amount'].mean(), inplace=True)

cat_df = Data.select_dtypes(include=['object']).copy()
cat_df.columns

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

for column in cat_df.columns:
   cat_df[column]=label_encoder.fit_transform(cat_df[column])
   
num_df = Data.select_dtypes(include=['int64','float64','int32']).copy()
num_df.columns
num_df= num_df.reset_index()
num_df =num_df.drop(['index'], axis=1)


#final_df=pd.concat([cat_df,num_df], axis=1)
cat_df = cat_df.reset_index()
cat_df = cat_df.drop(['index'], axis=1)

final_df=pd.concat([cat_df,num_df], axis=1)
import seaborn as sns
corr = final_df.corr()
sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.6:
            if columns[j]:
                columns[j] = False
selected_columns = final_df.columns[columns]
final_df1 = final_df[selected_columns]


Train_final = final_df1[final_df['type'] ==1]
Train_final = Train_final.drop(['type'], axis=1)
Train_final = Train_final.drop(['Loan_ID'], axis=1)


Test_final = final_df1[final_df['type'] ==0]
Test_final = Test_final.drop(['type'], axis=1)
Test_final = Test_final.drop(['Loan_ID'], axis=1)


x = Train_final.drop(['Interest_Rate'], axis=1)
y1= Train_final['Interest_Rate']
y1 = pd.DataFrame(y1)

X_train,X_test,Y_train,Y_test=train_test_split(x,y1,test_size=0.30,random_state=1)

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)



## Applying XGboost


estimator_xgb = XGBClassifier(
        objective= 'multi:softprob',
        nthread=4,
        seed=42)
    
#=============Hyperparamters============================
    
parameters_xgb = {
        "early_stopping_rounds":[10],
        "eval_set" : [[X_test, Y_test]],            
        'max_depth': range (2, 10, 1),
        'n_estimators': range(60, 100, 20),
        'learning_rate': [0.1, 0.01, 0.05],
        
                     }
    
    #============Hyperparameter tuning using Grid Search using AUC as Scoring metric===
    
grid_search_xgb = GridSearchCV(
        estimator=estimator_xgb,
        param_grid=parameters_xgb,
        n_jobs = 10,
        cv = 4,
        verbose=True)


from sklearn.gaussian_process import GaussianProcessClassifier
model = GaussianProcessClassifier()
model.fit(X_train, Y_train)
   
grid_search_xgb.fit(X_train, Y_train)

y_pred_xgb = grid_search_xgb.predict(X_test)  
    #selection = SelectFromModel(grid_search_xgb, threshold=0.3, prefit=True)
predictions_xgb = [round(value) for value in y_pred_xgb]
predictions_xgb=pd.DataFrame(predictions_xgb)
accuracy_xgb = accuracy_score(Y_test, predictions_xgb)    
auc_xgb=multiclass_roc_auc_score(Y_test,predictions_xgb)  


    
#=========================================Light GBM=====================================
    
def classifier_light_gbm(X_train,X_test,Y_train,Y_test,flag):
    
    estimator_light_gbm=lgb.LGBMClassifier()
    
    parameters_light_gbm={
                        'learning_rate': [ 0.1],
                        'num_leaves': [31],
                        'boosting_type' : ['gbdt'],
                        'objective' : ['multiclass']
                        }
    
    grid_search_gbm=GridSearchCV(
                estimator=estimator_light_gbm,
                param_grid=parameters_light_gbm,
                n_jobs = 10,
                cv = 10,
                verbose=True)
    
    grid_search_gbm.fit(X_train,Y_train)    
    #grid_search_gbm.best_estimator_.feature_importances_
    y_pred_gbm = grid_search_gbm.predict(X_test)  
    Prob = pd.DataFrame(grid_search_gbm.predict_proba(X_test)).iloc[:,1]
    predictions_gbm = [round(value) for value in y_pred_gbm]    
    accuracy_gbm = accuracy_score(Y_test, predictions_gbm)    
    auc_gbm=multiclass_roc_auc_score(Y_test, predictions_gbm)    
    print("Accuracy: %.2f%%" % (accuracy_gbm * 100.0))    
    precision_recall_f1_gbm=classification_report(Y_test, predictions_gbm)
    
    
    if flag ==3:
        return accuracy_gbm
    if flag ==2:
        return Prob
    if flag==1:
        return auc_gbm
    else:
        return pickle.dump(grid_search_gbm, open('Light_GBM_model.sav', 'wb'))
    

###===============MODEL STACKING==================================================#####



from mlxtend.classifier import StackingCVClassifier
#from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

xgb = XGBClassifier()
lgbm = LGBMClassifier()
rf = RandomForestClassifier()
#svc = SVC(kernel='rbf')

model_svc = SVC(kernel='rbf')
model_svc1 = CalibratedClassifierCV(model_svc) 

#abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
# Train Adaboost Classifer


stack = StackingCVClassifier(classifiers=(model_svc1,rf, lgbm, xgb),
                            meta_classifier=grid_search_xgb, cv=3,
                            use_features_in_secondary=True,
                            store_train_meta_features=True,
                            shuffle=False,
                            random_state=42)

# Create list to store classifiers
classifiers = {"supportvec": model_svc1,
               "RandomForest": rf,
               "LightGBM": lgbm,
               "Xgboost": xgb,
               "Stack": stack}

for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    
    # Fit classifier
    classifier.fit(X_train, Y_train)
        
    # Save fitted classifier
    classifiers[key] = classifier
    

# Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict(X_test)
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
Y_test = Y_test.reset_index()
Y_test = Y_test.drop(['index'], axis=1)
results["Target"] = Y_test

for key, counter in zip(classifiers, range(5)):
    # Get predictions
    y_pred = results[key]
    
    # Get AUC
    #auc = metrics.roc_auc_score(y_test, y_pred)
    auc = metrics.accuracy_score(Y_test, y_pred)
    print("Accuracy for %s is : %.2f%%" % (key,auc * 100.0))    


###Scoring the train data ####

ScoringData = Test_final

#Y_OOS = ScoringData[['Client Name','Overall_Status_CM']]
#X_OOS = ScoringData.drop(['Client Name'], axis=1)
#X_OOS = ScoringData.drop(['Client Name'], axis=1)
X_OOS = ScoringData.drop(['Interest_Rate'], axis=1)


  #independent columns
#y =y.reset_index()
#y = y.drop(['index'], axis=1)

#X_OOS_1 = pd.get_dummies(X_OOS)
#Y_OOS['Overall_Status_CM']= label_encoder.fit_transform(Y_OOS['Overall_Status_CM']) 
#Y_OOS1 = Y_OOS['Overall_Status_CM']

results1 = grid_search_xgb.predict(X_OOS)
results1 =pd.DataFrame(results1)

results1.to_excel(r"ScoredResults_v6.xlsx")




 































#===========================Score a new data=====================================

ScoringData = Test_final



X_OOS = ScoringData.drop(['Interest_Rate'], axis=1)  #independent columns
#y =y.reset_index()
#y = y.drop(['index'], axis=1)

results1 = grid_search_xgb.predict(X_OOS)
results1 =pd.DataFrame(results1)





###BEST SOLUTION


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
df_train = pd.read_csv("train_fNxu4vz.csv")
df_test = pd.read_csv("test_fjtUOL8.csv")

# Convert to numeric
df_train["Loan_Amount_Requested"] = df_train["Loan_Amount_Requested"].str.replace(",", "")
df_train["Loan_Amount_Requested"] = pd.to_numeric(df_train["Loan_Amount_Requested"])
df_test["Loan_Amount_Requested"] = df_test["Loan_Amount_Requested"].str.replace(",", "")
df_test["Loan_Amount_Requested"] = pd.to_numeric(df_test["Loan_Amount_Requested"])

# Fill NaN
df_train["Length_Employed"].fillna('NaN', inplace=True)
df_test["Length_Employed"].fillna('NaN', inplace=True)

df_train["Home_Owner"].fillna('NaN', inplace=True)
df_test["Home_Owner"].fillna('NaN', inplace=True)

df_train["Income_Verified"].fillna('NaN', inplace=True)
df_test["Income_Verified"].fillna('NaN', inplace=True)

df_train["Purpose_Of_Loan"].fillna('NaN', inplace=True)
df_test["Purpose_Of_Loan"].fillna('NaN', inplace=True)

df_train["Gender"].fillna('NaN', inplace=True)
df_test["Gender"].fillna('NaN', inplace=True)

# Drop loan ids
df_train = df_train.drop(["Loan_ID"], axis=1)
loan_ids = df_test["Loan_ID"].values
df_test = df_test.drop(["Loan_ID"], axis=1)

# Fill NaN with mean
df_train["Annual_Income"].fillna(df_train["Annual_Income"].mean(), inplace=True)
df_test["Annual_Income"].fillna(df_test["Annual_Income"].mean(), inplace=True)

# Assumption: If it is NaN, then user has no deliquency, so set with 0
df_train["Months_Since_Deliquency"].fillna(0, inplace=True)
df_test["Months_Since_Deliquency"].fillna(0, inplace=True)

# New feature
df_train["Number_Invalid_Acc"] = df_train["Total_Accounts"] - df_train["Number_Open_Accounts"]
df_test["Number_Invalid_Acc"] = df_test["Total_Accounts"] - df_test["Number_Open_Accounts"]

# New feature
df_train["Number_Years_To_Repay_Debt"] = df_train["Loan_Amount_Requested"]/df_train["Annual_Income"]
df_test["Number_Years_To_Repay_Debt"] = df_test["Loan_Amount_Requested"]/df_test["Annual_Income"]

df_train.head()

X_train, Y = df_train.drop(["Interest_Rate"], axis=1).values, df_train["Interest_Rate"].values
X_test = df_test.values

X_train.shape, Y.shape, X_test.shape

##Perform Validation
kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()
for train, test in kfold.split(X_train):
    x_train, x_test = X_train[train], X_train[test]
    y_train, y_test = Y[train], Y[test]
    
    model = CatBoostClassifier(random_state=27, max_depth=4,  devices="0:1", n_estimators=1000, verbose=500)
    model.fit(x_train, y_train, cat_features=[1, 2, 4, 5, 11])
    preds = model.predict(x_test)
    score = f1_score(y_test, preds, average="weighted")
    scores.append(score)
    print(score)
print("Average: ", sum(scores)/len(scores))


model = CatBoostClassifier(random_state=27, devices="0:1", n_estimators=1000, max_depth=4, verbose=500)
model.fit(X_train, Y, cat_features=[1, 2, 4, 5, 11])
preds1 = model.predict_proba(X_test)

#Check Feature Importance

feat_imp = pd.Series(model.feature_importances_, index=df_train.drop(["Interest_Rate"], axis=1).columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))

#LIGHTGBM
#Pre-processing specific to LightGBM


df_train = pd.read_csv("train_fNxu4vz.csv")
df_test = pd.read_csv("test_fjtUOL8.csv")

# Convert to numeric
df_train["Loan_Amount_Requested"] = df_train["Loan_Amount_Requested"].str.replace(",", "")
df_train["Loan_Amount_Requested"] = pd.to_numeric(df_train["Loan_Amount_Requested"])
df_test["Loan_Amount_Requested"] = df_test["Loan_Amount_Requested"].str.replace(",", "")
df_test["Loan_Amount_Requested"] = pd.to_numeric(df_test["Loan_Amount_Requested"])

# Fill NaN
df_train["Length_Employed"].fillna('NaN', inplace=True)
df_test["Length_Employed"].fillna('NaN', inplace=True)

df_train["Home_Owner"].fillna('NaN', inplace=True)
df_test["Home_Owner"].fillna('NaN', inplace=True)

df_train["Purpose_Of_Loan"].fillna('NaN', inplace=True)
df_test["Purpose_Of_Loan"].fillna('NaN', inplace=True)

df_train["Gender"].fillna('NaN', inplace=True)
df_test["Gender"].fillna('NaN', inplace=True)

# Drop loan ids
df_train = df_train.drop(["Loan_ID"], axis=1)
loan_ids = df_test["Loan_ID"].values
df_test = df_test.drop(["Loan_ID"], axis=1)

# Label Encode
le = LabelEncoder()
df_train["Length_Employed"] = le.fit_transform(df_train["Length_Employed"])
df_test["Length_Employed"] = le.transform(df_test["Length_Employed"])

df_train["Home_Owner"] = le.fit_transform(df_train["Home_Owner"])
df_test["Home_Owner"] = le.transform(df_test["Home_Owner"])

df_train["Income_Verified"] = le.fit_transform(df_train["Income_Verified"])
df_test["Income_Verified"] = le.transform(df_test["Income_Verified"])

df_train["Purpose_Of_Loan"] = le.fit_transform(df_train["Purpose_Of_Loan"])
df_test["Purpose_Of_Loan"] = le.transform(df_test["Purpose_Of_Loan"])

df_train["Gender"] = le.fit_transform(df_train["Gender"])
df_test["Gender"] = le.transform(df_test["Gender"])

# Fill NaN with mean
df_train["Annual_Income"].fillna(df_train["Annual_Income"].mean(), inplace=True)
df_test["Annual_Income"].fillna(df_test["Annual_Income"].mean(), inplace=True)

# Assumption: If it is NaN, then user has no deliquency, so set with 0
df_train["Months_Since_Deliquency"].fillna(0, inplace=True)
df_test["Months_Since_Deliquency"].fillna(0, inplace=True)

# New feature
df_train["Number_Invalid_Acc"] = df_train["Total_Accounts"] - df_train["Number_Open_Accounts"]
df_test["Number_Invalid_Acc"] = df_test["Total_Accounts"] - df_test["Number_Open_Accounts"]

# New feature
df_train["Number_Years_To_Repay_Debt"] = df_train["Loan_Amount_Requested"]/df_train["Annual_Income"]
df_test["Number_Years_To_Repay_Debt"] = df_test["Loan_Amount_Requested"]/df_test["Annual_Income"]

df_train.head()

X_train, Y = df_train.drop(["Interest_Rate"], axis=1).values, df_train["Interest_Rate"].values
X_test = df_test.values

X_train.shape, Y.shape, X_test.shape

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

kfold, scores = KFold(n_splits=5, shuffle=True, random_state=0), list()
for train, test in kfold.split(X_train):
    x_train, x_test = X_train[train], X_train[test]
    y_train, y_test = Y[train], Y[test]
    
    num_class1, num_class2, num_class3 = Counter(y_train)[1], Counter(y_train)[2], Counter(y_train)[3]
    sm = SMOTE(random_state=27, sampling_strategy={1: int(2.0*num_class1), 2: int(1.6*num_class2), 3: int(1.6*num_class3)})
    x_train, y_train = sm.fit_resample(x_train, y_train)
    
    model = LGBMClassifier(random_state=27, max_depth=6, n_estimators=400)
    model.fit(x_train, y_train, categorical_feature=[1, 2, 4, 5, 11])
    preds = model.predict(x_test)
    score = f1_score(y_test, preds, average="weighted")
    scores.append(score)
    print(score)
print("Average: ", sum(scores)/len(scores))

##Make final prediction using Lightgbm

# We apply SMOTE on all classes, thus increasing total sample size of each class
# This generalizes the decision boundary
num_class1, num_class2, num_class3 = Counter(Y)[1], Counter(Y)[2], Counter(Y)[3]
sm = SMOTE(random_state=27, sampling_strategy={1: int(2.0*num_class1), 2: int(1.6*num_class2), 3: int(1.6*num_class3)})
X_train_, Y_ = sm.fit_resample(X_train, Y)

model = LGBMClassifier(random_state=27, max_depth=6, n_estimators=400)
model.fit(X_train_, Y_, categorical_feature=[1, 2, 4, 5, 11])
preds2 = model.predict_proba(X_test)

feat_imp = pd.Series(model.feature_importances_, index=df_train.drop(["Interest_Rate"], axis=1).columns)
feat_imp.nlargest(30).plot(kind='barh', figsize=(8,10))

##Ensembling
preds = list()
cb_weight=0.4 # Catboost
lb_weight=0.6 # LGBM

for i, j in zip(preds1, preds2):
    xx = [(cb_weight * i[0]) + (lb_weight * j[0]),
          (cb_weight * i[1]) + (lb_weight * j[1]),
          (cb_weight * i[2]) + (lb_weight * j[2])]
    preds.append(xx)
print(preds[:10])

for i, j in zip(preds1, preds2):
    xx = [(cb_weight * i[0]) + (lb_weight * j[0]),
          (cb_weight * i[1]) + (lb_weight * j[1]),
          (cb_weight * i[2]) + (lb_weight * j[2])]
    preds.append(xx)
print(preds[:10])
preds[1]

preds=np.argmax(preds,axis=1)+1

df_submit = pd.DataFrame({'Loan_ID': loan_ids, 'Interest_Rate': preds}) # Ensemble submission
df_submit.to_csv("submit1.csv", index=False)
