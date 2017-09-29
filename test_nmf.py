## Importing required libraries
## need the master developing branch of scikit-learn
## and pull request: https://github.com/scikit-learn/scikit- learn/pull/8474
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from time import time
from sklearn.decomposition import NMF

import warnings

# warnings.simplefilter("ignore")


def find_common_team_ids(team_ids_1, team_ids_2):
    # common_set = set(train_team_ids).intersection(set(test_team_ids))
    return np.intersect1d(team_ids_1, team_ids_2)


def compute_approx_diagonal(matrix):
    # print(matrix.shape[0])
    diag = np.array([])
    for i in range(0,matrix.shape[0]):
        rowSum = np.sum(matrix[i,:])
        colSum = np.sum(matrix[:,i])
        diag_i = (2 * rowSum * colSum) / (2 * np.sum(matrix) - rowSum - colSum)
        diag = np.concatenate((diag, [diag_i]))
    # print(diag)
    return diag


def generate_goal_df(df1, df2, matches):
    for idx, match in matches.iterrows():
        home_team = match['home_team_api_id']
        away_team = match['away_team_api_id']
        # note we neglect teams not appearing in BOTH 09 and 10
        if (home_team in df1.columns) & (away_team in df1.columns):
            # since the column & row labels are sorted ascent after np.intersect1d
            # we can easily always let upper right df matrix to store home goals
            if (home_team < away_team):
                df1.loc[match['away_team_api_id'], match['home_team_api_id']] = \
                    match['away_team_goal']
                df1.loc[match['home_team_api_id'], match['away_team_api_id']] = \
                    match['home_team_goal']
            else:
                df2.loc[match['away_team_api_id'], match['home_team_api_id']] = \
                    match['home_team_goal']
                df2.loc[match['home_team_api_id'], match['away_team_api_id']] = \
                    match['away_team_goal']

    # diag1 = compute_approx_diagonal(df1.values)
    # # print(diag1)
    for i in range(0, df1.values.shape[0]):
        df1.iloc[i,i] = np.nan
    #
    # diag2 = compute_approx_diagonal(df2.values)
    # # print(diag2)
    for i in range(0, df2.values.shape[0]):
        df2.iloc[i,i] = np.nan

    # for i in range(0, train_num_teams):
    #     for j in range(0, train_team_ids):
    #         if i < j:
    #             df.iloc[i,j] = matches.loc[matches['home_team_api_id'] == common_teams[i] && ]


def generate_result_vector_from_matrix(df1, df2):
    r = np.array([])
    #
    # # convert to values (2D-array)
    # df1 = df1.values
    # df2 = df2.values
    for i in range(0, df1.shape[0]):
        for j in range(i+1, df1.shape[1]):
            diff = df1[i,j] - df1[j,i]
            if abs(diff) < 0.1:
                result = 0
            else:
                result = np.sign(diff)
            # result = np.sign(df1[i,j] - df1[j,i])
            r = np.concatenate((r, [result]))

    for i in range(0, df2.shape[0]):
        for j in range(i + 1, df2.shape[1]):
            diff = df2[i,j] - df2[j,i]
            if abs(diff) < 0.1:
                result = 0
            else:
                result = np.sign(diff)
            # result = np.sign(df2[i, j] - df2[j, i])
            r = np.concatenate((r, [result]))

    return r


def compute_nmf(matrix):
    model = NMF(n_components=7, solver='mu', init='random', max_iter=2000)
    W = model.fit_transform(matrix)
    H = model.components_
    print(np.linalg.norm(matrix - W.dot(H), 'fro') / np.linalg.norm(matrix, 'fro'))
    return W.dot(H)


start = time()
## Fetching data
# Connecting to database
path = "./input/"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

matches = pd.read_sql("SELECT * FROM Match;", conn)

col_to_include = ["league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
                  "away_team_api_id", "home_team_goal", "away_team_goal"]
# eliminate dont care columns
matches = matches[col_to_include]
# substract England Premier League matches for 08~09 and 09~10, note that league_id is an integer
matches = matches.loc[matches['league_id'] == 1729]
matches.dropna(subset=col_to_include, inplace=True)
match_train = matches.loc[matches['season'] == '2008/2009']
match_test = matches.loc[matches['season'] == '2009/2010']

# 2 seasons, total of 380 matches per season
print(match_train.shape[0])
print(match_test.shape[0])

# we don't train and predict the matches involving teams noe appearing in both seasons
train_team_ids = match_train['home_team_api_id'].unique()
test_team_ids = match_test['home_team_api_id'].unique()
common_teams = find_common_team_ids(train_team_ids, test_team_ids)
print(common_teams)
# 17/20 teams appear in both seasons
common_num_teams = common_teams.shape[0]

# each pair of teams has two matches in one season
# match outcomes with smaller team id being home teams
df_train_1 = pd.DataFrame(np.zeros((common_num_teams, common_num_teams)), \
                  index=common_teams, \
                  columns=common_teams)
# match outcomes with larger team id being home teams
df_train_2 = pd.DataFrame(np.zeros((common_num_teams, common_num_teams)), \
                  index=common_teams, \
                  columns=common_teams)
generate_goal_df(df_train_1, df_train_2, match_train)


df_test_1 = pd.DataFrame(np.zeros((common_num_teams, common_num_teams)), \
                  index=common_teams, \
                  columns=common_teams)
df_test_2 = pd.DataFrame(np.zeros((common_num_teams, common_num_teams)), \
                  index=common_teams, \
                  columns=common_teams)
generate_goal_df(df_test_1, df_test_2, match_test)

# print(df_train_1)
# print(df_test_1)

alpha = 0.1
matrix_train_1 = df_train_1.values * (1-alpha) + df_train_2.values * alpha
matrix_train_2 = df_train_2.values * (1-alpha) + df_train_1.values * alpha

approx_matrix_train_1 = compute_nmf(matrix_train_1)
approx_matrix_train_2 = compute_nmf(matrix_train_2)
print(matrix_train_2)

# print(approx_matrix_train_1)

pred = generate_result_vector_from_matrix(approx_matrix_train_1, approx_matrix_train_2)
real = generate_result_vector_from_matrix(df_test_1.values, df_test_2.values)
print(pred.shape)
print(real.shape)
# print(pred.sum())
# print(real.sum())

# match_train = match_train[match_train['home_team_api_id'] != 8549]
# match_train = match_train[match_train['home_team_api_id'] != 8659]
# match_train = match_train[match_train['home_team_api_id'] != 10261]
# match_train = match_train[match_train['away_team_api_id'] != 8549]
# match_train = match_train[match_train['away_team_api_id'] != 8659]
# match_train = match_train[match_train['away_team_api_id'] != 10261]
#
#
# print(np.setxor1d(train_team_ids, common_teams))
# match_train['result'] = np.sign(match_train['home_team_goal'] - match_train['away_team_goal'])
# print(match_train['result'].sum())
#
# get_bookkeeper_reselt()
#
# print(matches.head())
#
print(accuracy_score(pred, real))
