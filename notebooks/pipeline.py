import os
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

	# historic match-ups: what was the score differential the last time this couple met in the tourney?

	# elo rankings
elo_obj = elo() #it's a class, so you have to instantiate
elo_df = elo_obj.elo_ranks()

# Merge into master dataset - then move to julia....merging is wrong because of double-sidedness
full = pd.merge(left=seeds, right=adf, how='left', on=['Season', 'WTeamID', 'LTeamID', 'Result'])
full = pd.merge(left=full, right=elo_df, how='left', on=['Season', 'WTeamID', 'LTeamID', 'Result'])
full = full.dropna()

# Create cross-validation setup (integrate some ideas from Lopez del Prado in purged k-fold CV)


# Create final matchups, write to CSV


# Go to R, then run the KAGGLE NCAA algo to simulate the final probs