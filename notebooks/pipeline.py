import os
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.svm import LinearSVC
os.chdir('/Users/electron/Documents/mm/notebooks')
from eff_stats import *
from elo_ranks import *
from make_seeds import *
#import historical_matchups
#import elo

# Pipeline for integrating features

#### Generate features
	# basid seed
seeds = make_seeds()

	# efficiency stats
adf = eff_stats()

	# elo rankings
elo_obj = elo() #it's a class, so you have to instantiate
elo_df = elo_obj.elo_ranks()

# Merge into master dataset - then move to julia....merging is wrong because of double-sidedness
full = pd.merge(left=seeds, right=adf, how='left', on=['Season', 'WTeamID', 'LTeamID', 'Result'])
full = pd.merge(left=full, right=elo_df, how='left', on=['Season', 'WTeamID', 'LTeamID', 'Result'])
full = full.dropna()

# Create cross-validation setup (integrate some ideas from Lopez del Prado in purged k-fold CV)
ntrain = 1572
X_full = full.drop(['Result', 'WTeamID', 'LTeamID', 'Season'], axis=1).values
y_full = full.Result.values
X_full, y_full = shuffle(X_full, y_full)
X_train = X_full[:ntrain, :]
y_train = y_full[:ntrain]

# Create validation dataset
X_val = X_full[ntrain:, :]
y_val = y_full[ntrain:]

# DO SOME MODELLING WITH TRAINING DATA:::
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# SVM
clf = LinearSVC(random_state=0, tol=1e-5, penalty='l2', max_iter=5000)
#clf = CalibratedClassifierCV(clf) 
n_sims = 200
y_pred_svm = cross_val_predict(clf, X_train, y_train, cv=3, method='predict_proba')
#accuracy_score(y_train, y_pred_svm)
#y_pred_svm_prob = [ [x[0], 1-x[0]] for x in y_pred_svm]
log_loss(y_true = y_train, y_pred = y_pred_svm)

# LOGIT
logreg = LogisticRegression()
params = {'C': np.logspace(start=-5, stop=3, num=9)}
clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
clf.fit(X_train, y_train)
preds = [ [x[0], 1-x[0]] for x in clf.predict_proba(X_train)]
log_loss(y_true = y_train, y_pred = preds)

# NOT UNTIL THE VERY END!!!
preds = [ [x[0], 1-x[0]] for x in clf.predict_proba(X_test)]
log_loss(y_true = y_test, y_pred = preds)


# Create final matchups, write to CSV


# Go to R, then run the KAGGLE NCAA algo to simulate the final probs