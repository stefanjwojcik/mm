# THE GOAL OF THIS: COMPARE USING THE CDF AS THE OUTCOME COMPARED TO THE BINARY

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from scipy import stats
from scipy.stats import norm

data_dir = '/Users/electron/Documents/mm/notebooks/data'
df_seeds = pd.read_csv('DataFiles/NCAATourneySeeds.csv')
df_tour = pd.read_csv('DataFiles/NCAATourneyCompactResults.csv')

def seed_to_int(seed):
	#Get just the digits from the seeding. Return as int
	s_int = int(seed[1:3])
	return s_int

df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label

# Create smoothed outcome
df_tour['point_cdf'] = df_tour.WScore - df_tour.LScore

df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)


df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed

######
df_wins = pd.DataFrame()
df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['point_cdf'] = df_concat['point_cdf']
df_wins['Result'] = 1

#########
df_losses = pd.DataFrame()
df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['point_cdf'] = -df_concat['point_cdf']
df_losses['Result'] = 0

########
df_predictions = pd.concat((df_wins, df_losses))
df_predictions['point_cdf_'] = norm.cdf(df_predictions.point_cdf, scale = 5000)

# BASELINE MODEL
X_train = df_predictions.SeedDiff.values.reshape(-1,1)
y_train = df_predictions.point_cdf_.values
y_train_bin = df_predictions.Result.values
X_train, y_train, y_train_bin = shuffle(X_train, y_train, y_train_bin)

#logreg = LogisticRegression()
#params = {'C': np.logspace(start=-5, stop=3, num=9)}
#clf = GridSearchCV(logreg, params, scoring='neg_log_loss', refit=True)
#clf.fit(X_train, y_train_bin)

logreg = LogisticRegression()
logreg = logreg.fit(X_train, y_train_bin)
preds = [ [x[0], 1-x[0]] for x in logreg.predict_proba(X_train)]
y = [x for x in y_train_bin]
log_loss(y_true = y, y_pred = preds)

# ALTERNATIVE MODEL: LINEAR MODEL

# make a linear model, then use the predictions in a logit
linreg = LinearRegression()
linreg = linreg.fit(X_train, y_train)
linpreds = norm.cdf(linreg.predict(X_train), scale = 100)
preds = [ [x, 1-x] for x in linpreds]
y = [x for x in y_train_bin]
log_loss(y_true = y, y_pred = preds)

#########

df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage1.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


 X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    diff_seed = t1_seed - t2_seed
    X_test[ii, 0] = diff_seed

 preds = clf.predict_proba(X_test)[:,1]

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds
df_sample_sub.head()

df_sample_sub.to_csv('logreg_seed_starter.csv', index=False)

