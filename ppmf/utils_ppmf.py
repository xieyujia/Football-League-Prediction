#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:05:21 2017

@author: yujia
"""
import numpy as np
import pymc3 as pm
import theano
import scipy as sp

def matrix_construct(match_data, team_list):
    n_team=len(team_list)
    matrix = np.zeros([n_team, n_team])
    for team_home_index in range(n_team):
        match_home_team=match_data.loc[match_data['home_team_api_id'] == team_list[team_home_index]]
        for team_away_index in range(team_home_index+1, n_team):
            this_match=match_home_team.loc[match_home_team['away_team_api_id'] == team_list[team_away_index]]
            if len(this_match)==0:
                #print("Game of",team_home_index, team_away_index,'does not exist')
                matrix[team_home_index, team_away_index]= None
                matrix[team_away_index, team_home_index]= None
            else:
                matrix[team_home_index, team_away_index]= this_match['home_team_goal'].values[0]
                matrix[team_away_index, team_home_index]= this_match['away_team_goal'].values[0]

    matrix2 = np.zeros([n_team, n_team])
    for team_away_index in range(n_team):
        match_away_team=match_data.loc[match_data['away_team_api_id'] == team_list[team_away_index]]
        for team_home_index in range(team_away_index+1, n_team):
            this_match=match_away_team.loc[match_away_team['home_team_api_id'] == team_list[team_home_index]]
            if len(this_match)==0:
                #print("Game of",team_home_index, team_away_index,'does not exist')
                matrix2[team_home_index, team_away_index]= None
                matrix2[team_away_index, team_home_index]= None
            else:
                matrix2[team_home_index, team_away_index]= this_match['home_team_goal'].values[0]
                matrix2[team_away_index, team_home_index]= this_match['away_team_goal'].values[0]
    return matrix,matrix2
    
def ppmf_core(matrix, n_team):
    alpha_u = alpha_v = 1/np.var(matrix)
    alpha = np.ones((n_team,n_team)) * 2  # fixed precision for likelihood function
    dim = 10  # dimensionality
    #num_sample = 200
    
    with pm.Model() as pmf:
        pmf_U = pm.MvNormal('U', mu=0, tau=alpha_u * np.eye(dim),
                            shape=(n_team, dim), testval=np.random.randn(n_team, dim)*.01)
        pmf_V = pm.MvNormal('V', mu=0, tau=alpha_v * np.eye(dim),
                            shape=(n_team, dim), testval=np.random.randn(n_team, dim)*.01)
     #   pmf_R = pm.Normal('R', mu=theano.tensor.dot(pmf_U, pmf_V.T),
      #                    tau=alpha, observed=matrix)
        pmf_R = pm.Poisson('R', mu=theano.tensor.dot(pmf_U, pmf_V.T),
                           observed=matrix)
        # Find mode of posterior using optimization
        start = pm.find_MAP(fmin=sp.optimize.fmin_powell)  # Find starting values by optimization
    
    #    step = pm.NUTS(scaling=start)
    #    trace = pm.sample(num_sample, step, start=start)
    #
    #U_all = trace['U']
    #U = sum(U_all,0)/num_sample
    #V_all = trace['V']
    #V = sum(V_all,0)/num_sample
    #
    #R = U.dot(V.T)
    
    U = start['U']
    V = start['V']
    R = U.dot(V.T)
    return R
    
def test_accuracy(matrix, R, n_team):
    result_original = np.zeros([n_team, n_team])
    result_predicted = np.zeros([n_team, n_team])
    for team_home_index in range(n_team):
        for team_away_index in range(team_home_index+1, n_team):
            if matrix[team_home_index, team_away_index]== None:
                result_original[team_home_index, team_away_index]= None
                result_original[team_away_index, team_home_index]= None
            else:
                if matrix[team_home_index, team_away_index] >matrix[team_away_index, team_home_index]:
                    result_original[team_home_index, team_away_index]= 1
                    result_original[team_away_index, team_home_index]= -1
                else:
                    if matrix[team_home_index, team_away_index]==matrix[team_away_index, team_home_index]:
                        result_original[team_home_index, team_away_index]= 0
                        result_original[team_away_index, team_home_index]= 0
                    else:
                        result_original[team_home_index, team_away_index]= -1
                        result_original[team_away_index, team_home_index]= 1
        
                if R[team_home_index, team_away_index] >R[team_away_index, team_home_index]:
                    result_predicted[team_home_index, team_away_index]= 1
                    result_predicted[team_away_index, team_home_index]= -1
                else:
                    if R[team_home_index, team_away_index]==R[team_away_index, team_home_index]:
                        result_predicted[team_home_index, team_away_index]= 0
                        result_predicted[team_away_index, team_home_index]= 0
                    else:
                        result_predicted[team_home_index, team_away_index]= -1
                        result_predicted[team_away_index, team_home_index]= 1                                             
    
    #result = result_original - result_predicted
    
    #right_rate = 1 - np.count_nonzero(result)/n_team/n_team
    predict_result=[]                                        
    for team_home_index in range(n_team):
        for team_away_index in range(team_home_index+1, n_team):
            if matrix[team_home_index, team_away_index] != None:
                if result_original[team_home_index, team_away_index]==result_predicted[team_home_index, team_away_index]:
                    predict_result.append(1)
                else:
                    predict_result.append(0)
    #result = result_original - result_predicted
    
    right_rate = np.count_nonzero(predict_result)/len(predict_result)
    
    
    return right_rate