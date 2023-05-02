import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score

# 5 - eyes open, means when they were recording the EEG signal of the brain the patient had their eyes open
# 4 - eyes closed, means when they were recording the EEG signal the patient had their eyes closed
# 3 - Yes they identify where the region of the tumor was in the brain and recording the EEG activity from the healthy brain area
# 2 - They recorder the EEG from the area where the tumor was located
# 1 - Recording of seizure activity

epilepsy_df = pd.read_csv(r'C:\Users\tania\Documents\school\spring23\cs584\final\Epileptic Seizure Recognition.csv')
# check for any empty rows
epilepsy_df.isnull().sum()

# extract data into X and y sets
X = epilepsy_df.iloc[:,1:179].values
y = epilepsy_df.iloc[:,179].values
# make into binary (seizure vs not seizure)
y[y > 1] = 0
# split 80% training 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# models:
# logistic regression
# svm
# naive bayes
# k-nn

print("Training...")
logReg = LogisticRegression(max_iter=1000)
logReg.fit(X_train, y_train)
sv = SVC()
sv.fit(X_train, y_train)
gNB = GaussianNB()
gNB.fit(X_train, y_train)
kNN = KNeighborsClassifier()
kNN.fit(X_train, y_train)

print("Testing...")
y_pred_logReg = logReg.predict(X_test)
y_pred_sv = sv.predict(X_test)
y_pred_gNB = gNB.predict(X_test)
y_pred_kNN = kNN.predict(X_test)

# get results and metrics
acc_logReg = accuracy_score(y_test, y_pred_logReg)
fScore_logReg = f1_score(y_test, y_pred_logReg, average="binary")
acc_sv = accuracy_score(y_test, y_pred_sv)
fScore_sv = f1_score(y_test, y_pred_sv, average="binary")
acc_gNB = accuracy_score(y_test, y_pred_gNB)
fScore_gNB = f1_score(y_test, y_pred_gNB, average="binary")
acc_kNN = accuracy_score(y_test, y_pred_kNN)
fScore_kNN = f1_score(y_test, y_pred_kNN, average="binary")

# print table w/ metrics
table = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machine', 'Naive Bayes', 'k-Nearest Neighbors'],
    'Accuracy': [acc_logReg, acc_sv, acc_gNB, acc_kNN],
    'F-Score':[fScore_logReg, fScore_sv, fScore_gNB, fScore_kNN]
    })
print(table)

# print roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_logReg)
auc = round(roc_auc_score(y_test, y_pred_logReg), 5)
plt.plot(fpr, tpr, label="Logistic Regression, AUC="+str(auc))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_sv)
auc = round(roc_auc_score(y_test, y_pred_sv), 5)
plt.plot(fpr, tpr, label="Support Vector Model, AUC="+str(auc))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_gNB)
auc = round(roc_auc_score(y_test, y_pred_gNB), 5)
plt.plot(fpr, tpr, label="Naive Bayes, AUC="+str(auc))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_kNN)
auc = round(roc_auc_score(y_test, y_pred_kNN), 5)
plt.plot(fpr, tpr, label="k-Nearest Neighbors, AUC="+str(auc))
plt.title("ROC Curves of Models")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()