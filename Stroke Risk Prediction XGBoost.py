# -*- coding: utf-8 -*-
"""
@author: Charles Ma
ETH Zurich MTEC Master's Thesis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
import scipy.stats as stat
import seaborn as sns # for some plots
import xgboost as xgb
import time
import pickle
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay, make_scorer, roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Select the CSV file
d1 = pd.read_csv("ORIGINAL_stroke_research_final_V2.csv", header = 0)

d1 = d1[['hspid', 'gender9', 'age1', 'hi_bp', 'low_bp', 'BMI', 'glucose', 'CHO2', 'diag_flag']]
# diag_flag = 1 means stroke patient, 0 means normal
# gender9 = 1 means male, 0 means female
print(d1['diag_flag'].value_counts()) 
# there are 830 normal patients and 52 stroke patients, heavy class imbalance
X = d1[['hspid', 'gender9', 'age1', 'hi_bp', 'low_bp', 'BMI', 'glucose', 'CHO2']].copy()
Y = d1['diag_flag']
X.columns = ['HSPID', 'Gender', 'Age', 'Syst BP', 'Dias BP', 'BMI', 'Glucose', 'Total Cholesterol']

# Split data into training and test set, 80% in training, 20% in testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5991)
# Get only relevant values for the model, excluding the ID column
xtrain_data = x_train.iloc[:,1:8]  
xtest_data = x_test.iloc[:,1:8]


# Define function for hyperparameter tuning
def xgb_opt (sampler=""):
    model = XGBClassifier(objective='binary:logistic', eval_metric = 'auc', use_label_encoder=False ,random_state=5991)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=5991)
    steps = [('scaler', StandardScaler())]

    # resource: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
    # learning rate: makes model more robust, typically from 0.01 to 0.2, could also start from 0.001? 
    # min_child_weight: used to control overfitting, but too high can underfit
    # max_depth: used to control overfitting, higher value can overfit, typically from 3 to 10, depends on input vars
    # max_leaf_nodes: if defined, then max_depth is ignored
    # gamma: larger value makes model more conservative, 20 usually very high
    # max_delta_step: if 0 then no constraint, higher values makes model more conservative, typically not used but can help if class imbalanced
    # subsample: lower means more conservative, higher could lead to underfitting, typically from 0.5 to 1
    # colsample_bytree: subsample ratio of columns when constructing each tree, typically from 0.5 to 1
    # colsample_bylevel: subsample ratio of columns for each level, I WON'T USE THIS B/C I DON'T HAVE MANY COLUMNS TO START WITH
    # ALSO I'M ALREADY TUNING subsample AND colsample_bytree
    # lambda: L2 regularization term, ridge regression, reduces overfitting
    # alpha: L1 regularization term
    # n_estimators: default 100, how many trees are built, larger can overfit
    xgb_params={'model__max_depth': Integer(3, 10),
            'model__learning_rate': Real(0.01, 1,'log-uniform'),
            'model__subsample': Real(0.5, 1.0, 'uniform'),
            'model__colsample_bytree': Real(0.5, 1.0, 'uniform'),
            'model__min_child_weight': Integer(0, 10),
            'model__alpha': Real(0.001, 50, 'log-uniform'),
            'model__lambda': Real(0.001, 50, 'log-uniform'),
            'model__gamma': Real(0.0001, 1, 'log-uniform'),
            'model__n_estimators': Integer(50, 300)
            }
    
    if sampler == "":
        # scale_pos_weight: used for unbalanced classes, A typical value to consider: sum(negative instances) / sum(positive instances)
        # FOR MY DATA, TYPICAL scale_pos_weight IS 830/52 = 15.96
        xgb_params['model__scale_pos_weight'] = Real(1.0, 17.0)
    if sampler.lower() == "smotetomek":
        steps.append(('sampler', SMOTETomek(random_state= 5991)))
    if sampler.lower() == "smoteenn":
        steps.append(('sampler', SMOTEENN(random_state= 5991)))
    if sampler.lower() == "adasyn":
        steps.append(('sampler', ADASYN(random_state= 5991)))
    
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    
    bayesopt = BayesSearchCV(pipeline, xgb_params, scoring='roc_auc', cv=skf, n_iter=100, random_state=5991)
    return bayesopt


model1 = xgb_opt() # no imbalancing sampler
model2 = xgb_opt(sampler="SMOTETomek")
model3 = xgb_opt(sampler="SMOTEENN")
model4 = xgb_opt(sampler="ADASYN")


# Define function for results
def model_info(fittedmodel, xtest, ytest, sampler):
    print("Best Params: %s" % str(model1fit.best_params_))
    feature_importances = fittedmodel.best_estimator_.named_steps["model"].feature_importances_    
        
    y_pred = fittedmodel.predict(xtest)
    y_predscore = fittedmodel.predict_proba(xtest)[:,1]
    
    c_reportdict = classification_report(ytest, y_pred, output_dict=True)
    c_report = classification_report(ytest, y_pred, output_dict=False) # get a copy of classification report that's not dict
    print("Classification Report on Test Set:")
    print(c_report)
    
    c_matrix = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(ytest, y_pred))
    
    fpr, tpr, thresholds = roc_curve(ytest, y_predscore)
    roc_auc = auc(fpr, tpr)
    roc_plot = plt.figure()
    plt.title(f'Receiver Operating Characteristic for XGBoost with {sampler}', figure=roc_plot)
    plt.plot(fpr, tpr, color='red', label='AUC = %0.4f' % roc_auc, figure=roc_plot)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], linestyle='--', figure=roc_plot)
    plt.axis('tight')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    
    return feature_importances, y_pred, y_predscore, c_report, c_reportdict, c_matrix, roc_auc, roc_plot 


starttime = time.time() # measure time 
model1fit = model1.fit(xtrain_data, y_train) # no imbalancing sampler
endttime = time.time()
print("Time elapsed: {}".format(endttime - starttime)) # Normal baseline took 5 min 34 sec
model1_info = model_info(model1fit, xtest_data, y_test, "No Sampler")
disp1 = model1_info[5]
disp1.plot()  # show the confusion matrix
disp1.ax_.set_title("No Sampler XGBoost Confusion Matrix")

starttime = time.time()
model2fit = model2.fit(xtrain_data, y_train) # SMOTETomek sampler, took 4 min 40 sec
endttime = time.time()
print("Time elapsed: {}".format(endttime - starttime))
model2_info = model_info(model2fit, xtest_data, y_test, "SMOTETomek")
disp2 = model2_info[5]
disp2.plot()  
disp2.ax_.set_title("SMOTETomek XGBoost Confusion Matrix")

starttime = time.time()
model3fit = model3.fit(xtrain_data, y_train) # SMOTEENN sampler, took 4 min 41 sec
endttime = time.time()
print("Time elapsed: {}".format(endttime - starttime))
model3_info = model_info(model3fit, xtest_data, y_test, "SMOTEENN")
disp3 = model3_info[5]
disp3.plot()  
disp3.ax_.set_title("SMOTEENN XGBoost Confusion Matrix")

starttime = time.time()
model4fit = model4.fit(xtrain_data, y_train) # ADASYN sampler, took 5 min 20 sec
endttime = time.time()
print("Time elapsed: {}".format(endttime - starttime))
model4_info = model_info(model4fit, xtest_data, y_test, "ADASYN")
disp4 = model4_info[5]
disp4.plot()  
disp4.ax_.set_title("ADASYN XGBoost Confusion Matrix")

# save results
pickle.dump(model1_info, open("xgb_model1_info.pickle", "wb"))
pickle.dump(model1fit, open("xgb_model1fit.pickle", "wb")) 
pickle.dump(model2_info, open("xgb_model2_info.pickle", "wb"))
pickle.dump(model2fit, open("xgb_model2fit.pickle", "wb")) 
pickle.dump(model3_info, open("xgb_model3_info.pickle", "wb"))
pickle.dump(model3fit, open("xgb_model3fit.pickle", "wb")) 
pickle.dump(model4_info, open("xgb_model4_info.pickle", "wb"))
pickle.dump(model4fit, open("xgb_model4fit.pickle", "wb")) 

# load results
model1fit = pickle.load(open("xgb_model1fit.pickle", "rb"))
model2fit = pickle.load(open("xgb_model2fit.pickle", "rb"))
model3fit = pickle.load(open("xgb_model3fit.pickle", "rb"))
model4fit = pickle.load(open("xgb_model4fit.pickle", "rb"))


# Feature importance from SHAP values: the average of the marginal contributions across all permutations. 
import shap
explainer = shap.TreeExplainer(model1fit.best_estimator_.named_steps["model"])
scaledxtest = model1fit.best_estimator_.named_steps["scaler"].transform(xtest_data)  # scale the x test set
shap_values = explainer.shap_values(scaledxtest)
shap.summary_plot(shap_values, xtest_data, show=False)
plt.title("XGBoost with No Sampler Shapley Value Plot")

explainer2 = shap.TreeExplainer(model2fit.best_estimator_.named_steps["model"])
scaledxtest2 = model2fit.best_estimator_.named_steps["scaler"].transform(xtest_data)
shap_values2 = explainer2.shap_values(scaledxtest2)
shap.summary_plot(shap_values2, xtest_data, show=False)
plt.title("XGBoost with SMOTETomek Shapley Value Plot")

explainer3 = shap.TreeExplainer(model3fit.best_estimator_.named_steps["model"])
scaledxtest3 = model3fit.best_estimator_.named_steps["scaler"].transform(xtest_data)
shap_values3 = explainer3.shap_values(scaledxtest3)
shap.summary_plot(shap_values3, xtest_data, show=False)
plt.title("XGBoost with SMOTEENN Shapley Value Plot")    

explainer4 = shap.TreeExplainer(model4fit.best_estimator_.named_steps["model"])
scaledxtest4 = model4fit.best_estimator_.named_steps["scaler"].transform(xtest_data)
shap_values4 = explainer4.shap_values(scaledxtest4)
shap.summary_plot(shap_values4, xtest_data, show=False)
plt.title("XGBoost with ADASYN Shapley Value Plot")


# Below code summarizes results for each model as one dataframe
# macro avg: treats all classes equally, bigger penalisation when model does not perform well with minority classes (good against imbalance)
xgbscores = []
scoredict1 = model1_info[4]['macro avg']
del scoredict1['support']
scoredict1['AUC ROC score'] = model1_info[6]
xgbscores.append(scoredict1)

scoredict2 = model2_info[4]['macro avg']
del scoredict2['support']
scoredict2['AUC ROC score'] = model2_info[6]
xgbscores.append(scoredict2)

scoredict3 = model3_info[4]['macro avg']
del scoredict3['support']
scoredict3['AUC ROC score'] = model3_info[6]
xgbscores.append(scoredict3)

scoredict4 = model4_info[4]['macro avg']
del scoredict4['support']
scoredict4['AUC ROC score'] = model4_info[6]
xgbscores.append(scoredict4)

xgbscoresdf = pd.DataFrame(xgbscores)
xgbscoresdf.index = ['No Sampler', 'SMOTETomek', 'SMOTEENN', 'ADASYN']
# RECALL: True Positives/(True Positives + False Negatives). Appropriate when minimizing false negatives is the focus
# how many of the total numbers of positive instances were correctly classified
# PRECISION: True Positives/(True Positives + False Positives). Appropriate when minimizing false positives is the focus.
# the percentage of the instances classified in the positive class that is actually right
# F1-SCORE:  F1 score is defined as the harmonic mean of precision and recall

# False negatives are devastating for stroke prediction
# SMOTEENN (model 3) shows the best performance
model3fit = pickle.load(open("xgb_model3fit.pickle", "rb"))


# Prepare dataframes for percentile rankings
# get prediction prob scores for training data
y_trainscore = model3fit.predict_proba(xtrain_data)[:,1]  
# copy x_train dataframe to new dataframe
train_dfscores = x_train.copy()
# add the stroke labels back to x train data
train_dfscores['Stroke_Label'] = y_train
# add the scores back to complete x_train data
train_dfscores['Pred_Score'] = y_trainscore 
# change Score column to numeric because it wasn't for some reason
train_dfscores['Pred_Score'] = pd.to_numeric(train_dfscores['Pred_Score'])
# sort data from largest to smallest score
train_dfscores = train_dfscores.sort_values(['Pred_Score'], ascending=False) 
train_dfscores.to_pickle('xgb_train_dfscores.pkl')  # this dataframe contains only training data info
train_dfscores = pickle.load(open("xgb_train_dfscores.pkl", "rb"))

y_testscore = model3fit.predict_proba(xtest_data)[:,1]
test_dfscores = x_test.copy()
test_dfscores['Stroke_Label'] = y_test
test_dfscores['Pred_Score'] = y_testscore
test_dfscores['Pred_Score'] = pd.to_numeric(test_dfscores['Pred_Score'])
test_dfscores = test_dfscores.sort_values(['Pred_Score'], ascending=False)
test_dfscores.to_pickle('xgb_test_dfscores.pkl')  # this dataframe contains only test data info
test_dfscores = pickle.load(open("xgb_test_dfscores.pkl", "rb"))


# Create percentiles from 0 to 99 for training set
train_dfscores['Centile'] = pd.cut(train_dfscores['Pred_Score'].rank(method='max', pct=True), bins=100, labels=False)

# get the indices where each centile begins in the training set
unique_centile_index = np.unique(train_dfscores['Centile'], return_index=True)[1] 
# get the predicted scores from the training set where each centile begins 
train_centile_bins = train_dfscores['Pred_Score'].iloc[unique_centile_index].to_numpy()
# create array from 0 to 99, going by 1
percentiles1 = np.arange(1, 100, 1)  
# generate percentiles based on the training set
test_dfscores['Centile2'] = pd.cut(test_dfscores['Pred_Score'], bins=train_centile_bins, labels=percentiles1)
test_dfscores["Centile2"] = pd.to_numeric(test_dfscores['Centile2'])

# create percentile ranks
train_dfscores['Centile_Rank'] = train_dfscores['Centile'] + 1
test_dfscores['Centile_Rank'] = test_dfscores['Centile2'] + 1


# Create cross tables to help make the net lift table manually on Excel
#crosstab_train = pd.crosstab(train_dfscores['Centile'], train_dfscores['Stroke_Label'])
crosstab_train = pd.crosstab(train_dfscores['Centile_Rank'], train_dfscores['Stroke_Label'])
crosstab_train['Total'] = crosstab_train[0] + crosstab_train[1]

#crosstab_test = pd.crosstab(test_dfscores['Centile2'], test_dfscores['Stroke_Label'])
crosstab_test = pd.crosstab(test_dfscores['Centile_Rank'], test_dfscores['Stroke_Label'])
crosstab_test['Total'] = crosstab_test[0] + crosstab_test[1]


# CREATING VARIABLE FLAGS
# average age in dataset is 44, median and mode age is 42, it is skewed to the right, so we have relatively less old ppl 
# average age of normal ppl is 43.2, std dev is 12.1. Avg plus 1 std dev is 55, which I will define as risk flag boundary. 
# average systolic BP in dataset is 128.8, median 125, mode 124. I will define 140 and above as risk flag
# average diastolic BP is 80.2, median 79, mode 77. I will define 90 and above as risk flag
# BP values at or above 140/90 is hypertension according to WHO https://www.who.int/news-room/questions-and-answers/item/noncommunicable-diseases-hypertension
# avg BMI is 24, median 23.8, mode 22.9. I will define 30 and above as risk flag. 
# BMI above 30 is defined as obesity class I according to WHO https://www.euro.who.int/en/health-topics/disease-prevention/nutrition/a-healthy-lifestyle/body-mass-index-bmi
# avg glucose is 5.7, median 5.4, mode 5.5. Defined range from hospital wellness report is 3.89 to 6.11
# avg total chol is 4.5, median 4.4, mode 4.1. Defined range from hospital wellness report is 2.8 to 5.2 

# only include normal people, exclude stroke patients
train_dfscores2 = train_dfscores.loc[train_dfscores['Stroke_Label'] == 0] 
test_dfscores2 = test_dfscores.loc[test_dfscores['Stroke_Label'] == 0]  

train_dfscores2['Age_Flag'] = np.where(train_dfscores2['Age'] >= 55, 1, 0)
train_dfscores2['Sys_BP_Flag'] = np.where(train_dfscores2['Syst BP'] >= 140, 1, 0)
train_dfscores2['Dias_BP_Flag'] = np.where(train_dfscores2['Dias BP'] >= 90, 1, 0)
train_dfscores2['BMI_Flag'] = np.where(train_dfscores2['BMI'] >= 30, 1, 0)
train_dfscores2['Glucose_Flag'] = np.where(train_dfscores2['Glucose'] >= 6.11, 1, 0)
train_dfscores2['Total_Cho_Flag'] = np.where(train_dfscores2['Total Cholesterol'] >= 5.2, 1, 0)
flag_columns = ['Age_Flag', 'Sys_BP_Flag', 'Dias_BP_Flag', 'BMI_Flag', 'Glucose_Flag', 'Total_Cho_Flag']
train_dfscores2['Flag_Total'] = train_dfscores2[flag_columns].sum(axis=1)
print(train_dfscores2['Flag_Total'].value_counts())

test_dfscores2['Age_Flag'] = np.where(test_dfscores2['Age'] >= 55.3, 1, 0)
test_dfscores2['Sys_BP_Flag'] = np.where(test_dfscores2['Syst BP'] >= 140, 1, 0)
test_dfscores2['Dias_BP_Flag'] = np.where(test_dfscores2['Dias BP'] >= 90, 1, 0)
test_dfscores2['BMI_Flag'] = np.where(test_dfscores2['BMI'] >= 30, 1, 0)
test_dfscores2['Glucose_Flag'] = np.where(test_dfscores2['Glucose'] >= 6.11, 1, 0)
test_dfscores2['Total_Cho_Flag'] = np.where(test_dfscores2['Total Cholesterol'] >= 5.2, 1, 0)
test_dfscores2['Flag_Total'] = test_dfscores2[flag_columns].sum(axis=1)
print(test_dfscores2['Flag_Total'].value_counts())

# convert dataframes into excel files
train_dfscores2.to_excel('training_setDF_with_flags.xlsx',index = False, header=True)
test_dfscores2.to_excel('test_setDF_with_flags.xlsx',index = False, header=True)


# SENSITIVITY ANALYSIS
# patient 1807231436: age 59, BP 144/97, BMI 28.36, Glucose 7.15, Cholesterol 4.66, score 0.7747, 97 percentile rank
# try changing BP to 124/80, BMI 23, Glucose 5.6
subject1 = test_dfscores.loc[test_dfscores['HSPID'] == '1807231436']
subject1_adjusted = subject1.copy()
subject1_adjusted = subject1_adjusted.iloc[:,1:8]
subject1_adjusted['Syst BP'] = 124
subject1_adjusted['Dias BP'] = 80
subject1_adjusted['BMI'] = 23
subject1_adjusted['Glucose'] = 5.6
subject1_adjusted_score = model3fit.predict_proba(subject1_adjusted)[:,1]  # score dropped to 0.5503, 86 percentile rank
#stat.percentileofscore(test_dfscores['Pred_Score'], subject1_adjusted_score, kind='strict') 

# patient 1905270508: age 79, BP 167/103, BMI 26.83, Glucose 6.06, Cholesterol 3.05, score 0.7385, 94 percentile rank
# try changing BP to 135/85, BMI 24.5, 
subject2 = test_dfscores.loc[test_dfscores['HSPID'] == '1905270508']
subject2_adjusted = subject2.copy()
subject2_adjusted = subject2_adjusted.iloc[:,1:8]
subject2_adjusted['Syst BP'] = 135
subject2_adjusted['Dias BP'] = 85
subject2_adjusted['BMI'] = 24.5
subject2_adjusted_score = model3fit.predict_proba(subject2_adjusted)[:,1]  # score dropped to 0.5065, 84 percentile rank

# patient 1804171181: age 66, BP 130/74, BMI 26.84, Glucose 6.84, Cholesterol 5.57, score 0.7225, 94 percentile
# try changing to BMI 23.5, Glucose 5.6, Cholesterol 4.3
subject3 = test_dfscores.loc[test_dfscores['HSPID'] == '1804171181']
subject3_adjusted = subject3.copy()
subject3_adjusted = subject3_adjusted.iloc[:,1:8]
subject3_adjusted['BMI'] = 23.5
subject3_adjusted['Glucose'] = 5.6
subject3_adjusted['Total Cholesterol'] = 4.3
subject3_adjusted_score = model3fit.predict_proba(subject3_adjusted)[:,1]  # score dropped to 0.5104, 85 percentile rank
