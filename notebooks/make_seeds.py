"""
This file is responsible for creating basic ncaa seeds and for creating the outcome 'result'
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def seed_to_int(seed):
	#Get just the digits from the seeding. Return as int
	s_int = int(seed[1:3])
	return s_int

#data_dir = '../input/'
def make_seeds():
	print("loading data..")
	df_seeds = pd.read_csv('data/DataFiles/NCAATourneySeeds.csv')
	df_tour = pd.read_csv('data/DataFiles/NCAATourneyCompactResults.csv')

	df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
	df_seeds.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label

	df_tour.drop(labels=['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], inplace=True, axis=1)

	print("creating seeds...")
	df_winseeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
	df_lossseeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})
	df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'WTeamID'])
	df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'LTeamID'])
	df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed

	df_wins = pd.DataFrame()
	df_wins = df_concat[['Season',  'WTeamID',  'LTeamID', 'SeedDiff']]
	df_wins['Result'] = 1

	df_losses = pd.DataFrame()
	df_losses = df_concat[['Season',  'WTeamID',  'LTeamID']]
	df_losses['SeedDiff'] = -df_concat['SeedDiff']
	df_losses['Result'] = 0

	df_predictions = pd.concat((df_wins, df_losses))
	print("done")
	return(df_predictions)
