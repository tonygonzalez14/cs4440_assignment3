# -------------------------------------------------------------------------
# AUTHOR: Tony Gonzalez
# FILENAME: roc_curve.py
# SPECIFICATION: ROC Curve implementation
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

# read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv('cheat_data.csv', sep=',', header=0)
data_training = np.array(df.values)  # no ID column to skip

# transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
# Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [0, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
# be converted to a float.
X = []
for row in data_training:
    features = []
    # Refund: Yes = 1, No = 0 
    if row[0] == 'Yes':
        features.append(1)
    else:
        features.append(0)
    
    # one-hot encode 
    marital_status = row[1]
    if marital_status == 'Single':
        features.extend([1, 0, 0])
    elif marital_status == 'Divorced':
        features.extend([0, 1, 0])
    else:  # Married
        features.extend([0, 0, 1])
    
    # convert taxable income to float 
    taxable_income = row[2].replace('k', '')
    features.append(float(taxable_income))
    
    X.append(features)

# transform the original training classes to numbers and add them to the vector y. For instance Yes = 1, No = 0, so Y = [1, 1, 0, 0, ...]
y = []
for row in data_training:
    if row[3] == 'Yes':
        y.append(1)
    else:
        y.append(0)

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3)

# generate random thresholds for a no-skill prediction (random classifier)
ns_probs = np.random.random(len(testy))

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples (scores)
dt_probs = clf.predict_proba(testX)

# keep probabilities for the positive outcome only
dt_probs = dt_probs[:, 1]  # extract probabilities for class 1 (Yes)

# calculate scores by using both classifiers (no skilled and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')

# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')

# show the legend
pyplot.legend()

# show the plot
pyplot.show()