# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 09:56:33 2018

@author: Zhipe
"""
import timeit
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets,metrics
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE
np.random.seed(0)
# Load the data set
data = np.loadtxt('C:/Users/Zhipe/Desktop/Learning/ML/MLDSBAAIAssignment_1/ML-DSBA-AI-Assignment_1/data.csv' , delimiter=',')
data1 = np.loadtxt('C:/Users/Zhipe/Desktop/Learning/ML/MLDSBAAIAssignment_1/ML-DSBA-AI-Assignment_1/test.csv' , delimiter=',')  
#load 1st column 
y = data[:,0:1]
y1 =data1[:,0:1]
# load columns 2 − end 
X = data[:,1:data.shape[1]]
X1=data1[:,1:data1.shape[1]]


X, y= load_iris(return_X_y=True)

# Create classifiers
#lr = LogisticRegression(penalty='l2', dual=False, class_weight=None, solver='lbfgs')
#lr.fit(X,y)
#lr.predict(X)
#lr.score(X,y)
classifier = OneVsRestClassifier(LogisticRegression(),n_jobs=2)                            
classifier.fit(X, y)
y_score=classifier.decision_function(X1)
y_score=y_score.reshape(-1,1)
y_score
np.shape(y_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])

# Compute  ROC curve and ROC area
fpr["F166"], tpr["F166"], _ = roc_curve(y1.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["F166"], tpr["F166"])

##############################################################################
# Plot ROC curves for the multiclass problem



# Plot all ROC curves
plt.figure()
plt.plot(fpr["F166"], tpr["F166"],
         label='Without feature selwction ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic to LogisticRegression')
plt.legend(loc="lower right")
plt.show()
##############################################################################
#examine the area under curve
metrics.roc_auc_score(y1.ravel(), y_score.ravel())
#the running time required for training
timeit.timeit(stmt="from sklearn.multiclass import OneVsRestClassifier"
                   "classifier = OneVsRestClassifier(LogisticRegression(),n_jobs=2)" ,number=1000000)

##############################################################################
# Apply feature selection using the recursive feature elimination (RFE)

selector20=RFE(LogisticRegression(),20,step=1)
selector20=selector20.fit(X,y)
selector20.support_
y20_score=selector20.decision_function(X1)
y20_score=y_score.reshape(-1,1)
y20_score
np.shape(y20_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y20_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])

selector40=RFE(LogisticRegression(),40,step=1)
selector40=selector40.fit(X,y)
y40_score=selector40.decision_function(X1)
y40_score=y40_score.reshape(-1,1)
y40_score
np.shape(y40_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y40_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])

selector60=RFE(LogisticRegression(),60,step=1)
selector60=selector60.fit(X,y)
y60_score=selector60.decision_function(X1)
y60_score=y60_score.reshape(-1,1)
y60_score
np.shape(y60_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y60_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])

selector80=RFE(LogisticRegression(),80,step=1)
selector80=selector80.fit(X,y)
y80_score=selector80.decision_function(X1)
y80_score=y80_score.reshape(-1,1)
y80_score
np.shape(y80_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y80_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])

selector100=RFE(LogisticRegression(),100,step=1)
selector100=selector100.fit(X,y)
y100_score=selector100.decision_function(X1)
y100_score=y100_score.reshape(-1,1)
y100_score
np.shape(y100_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y100_score[:, 0:1])

roc_auc[1] = auc(fpr[1], tpr[1])
selector150=RFE(LogisticRegression(),150,step=1)
selector150=selector150.fit(X,y)
y150_score=selector150.decision_function(X1)
y150_score=y150_score.reshape(-1,1)
y150_score
np.shape(y150_score)
# plot ROC curves
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[1], tpr[1], _ = roc_curve(y1[:, 0:1], y150_score[:, 0:1])
roc_auc[1] = auc(fpr[1], tpr[1])
fpr["F20"], tpr["F20"], _ = roc_curve(y1.ravel(), y20_score.ravel())
roc_auc["F20"] = auc(fpr["F20"], tpr["F20"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["F20"], tpr["F20"],
         label='20 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F20"]),
         color='green', linestyle=':', linewidth=4)
fpr["F40"], tpr["F40"], _ = roc_curve(y1.ravel(), y40_score.ravel())
roc_auc["F40"] = auc(fpr["F40"], tpr["F40"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["F40"], tpr["F40"],
         label='40 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F40"]),
         color='navy', linestyle=':', linewidth=4)
fpr["F60"], tpr["F60"], _ = roc_curve(y1.ravel(), y60_score.ravel())
roc_auc["F60"] = auc(fpr["F60"], tpr["F60"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["F60"], tpr["F60"],
         label='20 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F60"]),
         color='aqua', linestyle=':', linewidth=4)
fpr["F80"], tpr["F80"], _ = roc_curve(y1.ravel(), y60_score.ravel())
roc_auc["F80"] = auc(fpr["F80"], tpr["F60"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["F80"], tpr["F80"],
         label='20 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F80"]),
         color='aqua', linestyle=':', linewidth=4)
fpr["F100"], tpr["F100"], _ = roc_curve(y1.ravel(), y100_score.ravel())
roc_auc["F100"] = auc(fpr["F100"], tpr["F100"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["F100"], tpr["F100"],
         label='20 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F100"]),
         color='darkorange', linestyle=':', linewidth=4)
fpr["F150"], tpr["F150"], _ = roc_curve(y1.ravel(), y150_score.ravel())
roc_auc["F150"] = auc(fpr["F150"], tpr["F150"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["F150"], tpr["F150"],
         label='150 features ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["F150"]),
         color='cornflowerblue', linestyle=':', linewidth=4)
#examine the area under curve
metrics.roc_auc_score(y1.ravel(), y20_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector20=RFE(LogisticRegression(),20,step=1)"
                   "selector20=selector20.fit(X,y)" ,number=1000000)
metrics.roc_auc_score(y1.ravel(), y40_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector40=RFE(LogisticRegression(),40,step=1)"
                   "selector40=selector40.fit(X,y)" ,number=1000000)
metrics.roc_auc_score(y1.ravel(), y60_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector60=RFE(LogisticRegression(),60,step=1)"
                   "selector60=selector60.fit(X,y)" ,number=1000000)
metrics.roc_auc_score(y1.ravel(), y80_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector80=RFE(LogisticRegression(),80,step=1)"
                   "selector80=selector80.fit(X,y)" ,number=1000000)
metrics.roc_auc_score(y1.ravel(), y100_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector100=RFE(LogisticRegression(),100,step=1)"
                   "selector100=selector100.fit(X,y)" ,number=1000000)
#examine the area under curve
metrics.roc_auc_score(y1.ravel(), y150_score.ravel())
#the running time required for training
timeit.timeit(stmt="selector150=RFE(LogisticRegression(),150,step=1)"
                   "selector150=selector150.fit(X,y)" ,number=1000000)



