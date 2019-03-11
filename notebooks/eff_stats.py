## FROM : https://www.kaggle.com/lnatml/feature-engineering-with-advanced-stats
"""
This file is responsible for creating 'advanced' features related to team efficiencies
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from math import pi


def eff_stats(df_path = 'data/DataFiles/NCAATourneyDetailedResults.csv'):
	print("loading...")
	df = pd.read_csv(df_path)

	#Points Winning/Losing Team
	df['WPts'] = df.apply(lambda row: 2*row.WFGM + row.WFGM3 + row.WFTM, axis=1)
	df['LPts'] = df.apply(lambda row: 2*row.LFGM + row.LFGM3 + row.LFTM, axis=1)

	#Calculate Winning/losing Team Possesion Feature
	wPos = df.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
	lPos = df.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)
	#two teams use almost the same number of possessions in a game
	#(plus/minus one or two - depending on how quarters end)
	#so let's just take the average
	df['Pos'] = (wPos+lPos)/2

	print("computing offensive/defensive rating...")
	#Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
	df['WOffRtg'] = df.apply(lambda row: 100 * (row.WPts / row.Pos), axis=1)
	df['LOffRtg'] = df.apply(lambda row: 100 * (row.LPts / row.Pos), axis=1)
	#Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
	df['WDefRtg'] = df.LOffRtg
	df['LDefRtg'] = df.WOffRtg
	#Net Rating = Off.Rtg - Def.Rtg
	df['WNetRtg'] = df.apply(lambda row:(row.WOffRtg - row.WDefRtg), axis=1)
	df['LNetRtg'] = df.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)
	                         
	#Assist Ratio : Percentage of team possessions that end in assists
	df['WAstR'] = df.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
	df['LAstR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
	print("computing turnovers...")
	#Turnover Ratio: Number of turnovers of a team per 100 possessions used.
	#(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
	df['WTOR'] = df.apply(lambda row: 100 * row.WTO / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
	df['LTOR'] = df.apply(lambda row: 100 * row.LTO / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
	                    
	#The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
	df['WTSP'] = df.apply(lambda row: 100 * row.WPts / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)
	df['LTSP'] = df.apply(lambda row: 100 * row.LPts / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)
	#eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable 
	df['WeFGP'] = df.apply(lambda row:(row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)      
	df['LeFGP'] = df.apply(lambda row:(row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)   
	#FTA Rate : How good a team is at drawing fouls.
	df['WFTAR'] = df.apply(lambda row: row.WFTA / row.WFGA, axis=1)
	df['LFTAR'] = df.apply(lambda row: row.LFTA / row.LFGA, axis=1)
	                         
	#OREB% : Percentage of team offensive rebounds
	df['WORP'] = df.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
	df['LORP'] = df.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
	#DREB% : Percentage of team defensive rebounds
	df['WDRP'] = df.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
	df['LDRP'] = df.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)                                      
	#REB% : Percentage of team total rebounds
	df['WRP'] = df.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
	df['LRP'] = df.apply(lambda row: (row.LDR + row.LOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1) 


	df.drop(['WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF'], axis=1, inplace=True)
	df.drop(['LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'], axis=1, inplace=True)

	# NEED TO MAKE THIS COMPATIBLE WITH THE REST OF THE DATA: TAKE DIFFS AND CONCATENATE
	df_concat = pd.DataFrame()
	vars_to_add =  set([x[1:] for x in df.columns[11:]])
	for var in vars_to_add:
		df_concat['Diff_'+var] = df['W'+var]-df['L'+var]
	# 
	df_win = df[['Season', 'WTeamID', 'LTeamID']]
	df_win = pd.concat([df_win, df_concat], axis=1)
	df_win['Result'] = 1

	df_loss = df[['Season', 'WTeamID', 'LTeamID']]
	df_loss = pd.concat([df_loss, -df_concat], axis=1)
	df_loss['Result'] = 0

	df_out = pd.concat((df_win, df_loss))

	print("done")
	return(df_out)
