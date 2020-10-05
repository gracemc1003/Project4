#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 14:31:25 2020

@author: gracemcmonagle
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import random

filepath = '/Users/gracemcmonagle/Desktop/School/Fall 2020/EECS 731/Project 4/data/nfl_games.csv'
rawData = pd.read_csv(filepath, delimiter = ',')

rawData['ScoreDif'] = rawData['score1'] - rawData['score2']



#given a team, predict the number of points they will score in a game
games1 = pd.DataFrame(rawData, columns = ['season', 'playoff', 'team1', 'elo1', 'ScoreDif', 'score1']).rename(columns={'team1':'team', 'elo1':'elo', 'score1':'score'})
games1['home'] = 1
games2 = pd.DataFrame(rawData, columns = ['season', 'playoff', 'team2', 'elo2', 'ScoreDif', 'score2']).rename(columns={'team2':'team', 'elo2':'elo', 'score2':'score'})
games2['home'] = 0

games = pd.concat([games1, games2])
teams = list(games.team.unique())

#%%
plt.hist(games['score'])
plt.xlabel('Score')
plt.ylabel('Count')

#count number of games by each team
#we see that a lot of teams have only a few games, for our purposes we want our teams to have at least 100 games in order for us to make a prediction
noGames = games['team'].value_counts()
noGames = pd.DataFrame(noGames.where(noGames > 100).dropna())
incTeams = list(noGames.index)

games = games[games['team'].isin(incTeams)]

#%% Linear Regression for team

def predictScoreLR(team):
    teamL = [team]
    teamData = games[games['team'].isin(teamL)]
    X = teamData.drop(columns = ['team', 'score', 'season', 'playoff'])
    y = teamData['score']
    data = []
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .2)
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        data.append(reg.score(x_test,y_test))
    return data
    
team = random.choice(incTeams)
scores = predictScoreLR(team)

plt.plot(list(range(100)), scores)
plt.title('100 R2 values for Linear Regression for ' + team)
plt.show()

#%%

def predictScoreGB(team):
    teamL = [team]
    teamData = games[games['team'].isin(teamL)]
    X = teamData.drop(columns = ['team', 'score', 'season', 'playoff'])
    y = teamData['score']
    data = []
    for i in range(100):
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .2)
        reg = GradientBoostingRegressor(n_estimators = 200, random_state=0, max_features = 3)
        reg.fit(x_train,y_train)
        score = reg.score(x_test, y_test)
        data.append(score)
    return data
    
team = random.choice(incTeams)
scores = predictScoreGB(team)

plt.plot(list(range(100)), scores)
plt.title('100 R2 values for Gradient Boosting for ' + team)
plt.show()
