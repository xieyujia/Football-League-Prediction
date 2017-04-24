#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 12:42:37 2017

@author: yujia
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from time import time
import warnings

from utils_ppmf import *

start = time()
## Fetching data
#Connecting to database
path = ""  #Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

#Defining the number of jobs to be run in parallel during grid search
n_jobs = 1 #Insert number of parallel jobs here

rows = [ "season", "stage", "date", "match_api_id", "home_team_api_id", 
        "away_team_api_id", "home_team_goal", "away_team_goal"]

#Fetching required data tables
num_year = 7
match_data = [0 for i in range(num_year)]
for i in range(num_year):
    conmmand = "SELECT * FROM Match WHERE league_id='1729' and season='" + str(2014-i)+ "/" + str(2015-i) +"';"
    match_data[i] = pd.read_sql(conmmand, conn)

test_data = pd.read_sql("SELECT * FROM Match WHERE league_id='1729' and season='2015/2016';", conn)

#Reduce match data to fulfill run time requirements    
for i in range(num_year):   
    match_data[i].dropna(subset = rows, inplace = True)

test_data.dropna(subset = rows, inplace = True)


team = pd.read_sql("SELECT distinct home_team_api_id AS team_id FROM Match WHERE league_id='1729' and season='2014/2015';", conn)
team = team.sort(columns='team_id')
n_team= len(team)
team_list = [team.values[i][0] for i in range(n_team)]

matrix_list = []
for i in range(num_year):
    _matrix1, _matrix2 = matrix_construct(match_data[i], team_list)
    matrix_list.append(_matrix1)
    matrix_list.append(_matrix2)
matrix_array = np.asarray(matrix_list)

matrix = np.zeros([n_team, n_team])
for i in range(n_team):
    for j in range(n_team):
        goal_list = []
        for k in range(num_year):
            if math.isnan(matrix_array[k][i][j]) != True:               
                goal_list.append(matrix_array[k][i][j])
        matrix[i,j] = sum(goal_list)/len(goal_list)      


test1,test2 = matrix_construct(test_data, team_list)

R = ppmf_core(matrix, n_team)

#right_rate = test_accuracy(matrix, R, n_team)
#print(right_rate)

right_rate1 = test_accuracy(test1, R, n_team)
right_rate2 = test_accuracy(test2, R, n_team)
right_rate = (right_rate1+right_rate2)/2

print('The accuracy for rediction is',right_rate)


