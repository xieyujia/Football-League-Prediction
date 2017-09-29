#final version, I believe...
## Importing required libraries
import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from time import time
# from imblearn.over_sampling import RandomOverSampler
from xml.dom.minidom import parse
import xml.dom.minidom


def time_discount_vector(length):
    time_discount_temp = range(length)
    time_discount = np.power(0.9, time_discount_temp)
    return time_discount

def get_match_label(match):
    """ Derives a label for a given match. """
    # Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.Series()
    label['match_api_id'] = match['match_api_id']

    # Identify match label
    if home_goals > away_goals:
        label['label'] = "Win"
    if home_goals == away_goals:
        label['label'] = "Draw"
    if home_goals < away_goals:
        label['label'] = "Defeat"

    # Return label (match_api_id; label)
    return label


def get_fifa_stats(match, player_stats):
    """ Aggregates fifa stats for a given match. """

    # Define variables
    match_id = int(match.match_api_id)
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]

    overall_ratings = np.array([])
    for player in players:
        # Get player ID
        player_id = match[player]

        # Get player stats
        stats = player_stats[player_stats.player_api_id == player_id]

        # Identify current stats
        current_stats = stats[stats.date < date].sort_values(by='date', ascending=False).iloc[0]

        # get overall rating for every player, this cannot be nan since we "dropna" in main()
        overall_ratings = np.concatenate((overall_ratings, [current_stats["overall_rating"]]))

    colNames = np.core.defchararray.add(players, '_overall_rating')
    player_stats_new = pd.Series(overall_ratings, index=colNames)
    player_stats_new['match_api_id'] = match_id
    # print(player_stats_new)
    return player_stats_new


def get_fifa_data(matches, player_stats, path=None, data_exists=False):
    """ Gets fifa data for all matches. """

    # Check if fifa data already exists
    if data_exists == True:

        fifa_data = pd.read_pickle(path)

    else:

        print("Collecting fifa data for each match...")
        start = time()

        # Apply get_fifa_stats for each match
        fifa_data = matches.apply(lambda x: get_fifa_stats(x, player_stats), axis=1)

        end = time()
        print("Fifa data collected in {:.1f} minutes".format((end - start) / 60))

    # Return fifa_data
    # print(fifa_data)
    return fifa_data


def get_most_recent_matches_helper(total_matches, date, x):
    # sort the matches given, output the most recent x if possible
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]
        # Check for error in data
        if last_matches.shape[0] > x:
            print("Error in obtaining matches")
    return last_matches


def get_most_recent_same_season_matches_helper(total_matches, date, season, x):
    # sort the matches given, output the most recent x if possible
    try:
        last_matches = total_matches[(total_matches.date < date)
                                     & (total_matches.season.str.contains(season))].sort_values(by='date', ascending=False).iloc[0:x, :]
    except:
        last_matches = total_matches[(total_matches.date < date)
                                     & (total_matches.season.str.contains(season))].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]
        # Check for error in data
        if last_matches.shape[0] > x:
            print("Error in obtaining matches")
    return last_matches


def get_last_ashome_matches(matches, date, season, team, x=5):
    """ Get the last x matches of a given team as home. """

    # Filter team matches from matches
    total_matches = matches[matches['home_team_api_id'] == team]

    # print(total_matches)

    last_matches = get_most_recent_same_season_matches_helper(total_matches, date, season, x)

    # omitting all with less than 5 recent matches
    # if last_matches.shape[0] != x:
    #     return last_matches.iloc[0:0, :]

    # print(last_matches)
    # Return last matches
    return last_matches


def get_last_asaway_matches(matches, date, season, team, x=5):
    """ Get the last x matches of a given team as away. """

    # Filter team matches from matches
    total_matches = matches[matches['away_team_api_id'] == team]

    last_matches = get_most_recent_same_season_matches_helper(total_matches, date, season, x)

    # Return last matches
    return last_matches


def get_last_competing_matches(matches, date, home_team, away_team, x=2):
    """ Get the last x competing matches between two given teams (in the same location). """

    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    last_matches = get_most_recent_matches_helper(home_matches, date, x)
    return last_matches


def get_last_reverse_competing_matches(matches, date, home_team, away_team, x=2):
    """ Get the last x competing matches between two given teams (in the ohter team's location). """
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    last_matches = get_most_recent_matches_helper(away_matches, date, x)
    return last_matches


def get_goals(matches, isAway=False):
    """ Get the average goals from a set of matches. """
    # the set of matches must share the same home_team or away_team

    num_of_matches = matches.shape[0]
    # print(matches.away_team_goal.shape)
    home_team_ova = matches['home_ova']
    away_team_ova = matches['away_ova']
    if isAway:
        goals = float(np.dot(matches.away_team_goal.values * time_discount_vector(num_of_matches), home_team_ova.values))
    else:
        # print(away_team_ova.values)
        # print(matches.home_team_goal.values)
        goals = float(np.dot(matches.home_team_goal.values * time_discount_vector(num_of_matches), away_team_ova.values))


    # print('1234 {}'.format(goals))
    if num_of_matches == 0:
        average_goals = np.nan
    else:
        average_goals = goals / num_of_matches

    # print('5678 {}'.format(average_goals))
    return average_goals


def get_goals_lost(matches, isAway=False):
    """ Get the average goals lost from a set of matches. """
    # the set of matches must share the same home_team or away_team

    num_of_matches = matches.shape[0]
    home_team_ova = matches['home_ova']
    away_team_ova = matches['away_ova']
    if isAway:
        goals = float(np.dot(matches.home_team_goal.values * time_discount_vector(num_of_matches), 50 - home_team_ova.values))
    else:
        goals = float(np.dot(matches.away_team_goal.values * time_discount_vector(num_of_matches), 50 - away_team_ova.values))

    if num_of_matches == 0:
        average_goals = np.nan
    else:
        average_goals = goals / num_of_matches

    return average_goals


def get_average_league_score(matches, isAway=False):
    """ Get the average league score per match from a set of matches. """
    # the set of matches must share the same home_team or away_team

    num_of_matches = matches.shape[0]
    home_team_ova = matches['home_ova']
    away_team_ova = matches['away_ova']
    score = np.zeros(num_of_matches)
    if isAway:
        for i in range(0,num_of_matches):
            if matches.away_team_goal.iloc[i] > matches.home_team_goal.iloc[i]:
                score[i] = 3.0
            elif matches.away_team_goal.iloc[i] == matches.home_team_goal.iloc[i]:
                score[i] = 1.0
            # wins = float(matches[matches.away_team_goal > matches.home_team_goal].shape[0])
        score_times_ova = score * home_team_ova

    else:
    #     wins = float(matches[matches.away_team_goal < matches.home_team_goal].shape[0])
        for i in range(0,num_of_matches):
            if matches.home_team_goal.iloc[i] > matches.away_team_goal.iloc[i]:
                score[i] = 3.0
            elif matches.home_team_goal.iloc[i] == matches.away_team_goal.iloc[i]:
                score[i] = 1.0
        score_times_ova = score * away_team_ova

    # draws = float(matches[matches.away_team_goal == matches.home_team_goal].shape[0])

    if num_of_matches == 0:
        total_score = np.nan
    else:
        # total_score = (wins * 3.0 + draws * 1.0) / num_of_matches
        total_score = np.dot(score_times_ova, time_discount_vector(num_of_matches))
    return total_score


def get_shoton(matches, isAway=False):
    if matches.shape[0] == 0:
        return np.nan
    if isAway:
        team = matches.away_team_api_id.iloc[0]
    else:
        team = matches.home_team_api_id.iloc[0]
    shoton_matches = matches.shoton.values.tolist()
    cnt = 0
    num_of_matches = 0
    i = 0
    shot_on = np.zeros(matches.shape[0])
    home_team_ova = matches['home_ova']
    away_team_ova = matches['away_ova']
    for data in shoton_matches:
        DOMTree = xml.dom.minidom.parseString(str(data))
        collection = DOMTree.documentElement
        values = collection.getElementsByTagName('value')
        if len(values) > 0:
            # this is valid match with stat available
            num_of_matches += 1
            for value in values:
                team_get = value.getElementsByTagName('team')
                if len(team_get) > 0 and team_get[0].childNodes[0].data == str(team):
                    cnt += 1
        shot_on[i] = cnt
        cnt = 0
        i += 1

    if isAway:
        shoton_weighted = np.dot(shot_on * time_discount_vector(matches.shape[0]), home_team_ova)
    else:
        shoton_weighted = np.dot(shot_on * time_discount_vector(matches.shape[0]), away_team_ova)

    return float(shoton_weighted) / num_of_matches


def get_shotoff(matches, isAway=False):
    if matches.shape[0] == 0:
        return np.nan
    if isAway:
        team = matches.away_team_api_id.iloc[0]
    else:
        team = matches.home_team_api_id.iloc[0]
    shotoff_matches = matches.shotoff.values.tolist()
    cnt = 0
    num_of_matches = 0
    i = 0
    shot_off = np.zeros(matches.shape[0])
    home_team_ova = matches['home_ova']
    away_team_ova = matches['away_ova']
    for data in shotoff_matches:
        DOMTree = xml.dom.minidom.parseString(str(data))
        collection = DOMTree.documentElement
        values = collection.getElementsByTagName('value')
        if len(values) > 0:
            # this is valid match with stat available
            num_of_matches += 1
            for value in values:
                team_get = value.getElementsByTagName('team')
                if len(team_get) > 0 and team_get[0].childNodes[0].data == str(team):
                    cnt += 1
        shot_off[i] = cnt
        cnt = 0
        i += 1
    if isAway:
        shotoff_weighted = np.dot(shot_off * time_discount_vector(matches.shape[0]), home_team_ova)
    else:
        shotoff_weighted = np.dot(shot_off * time_discount_vector(matches.shape[0]), away_team_ova)

    return float(shotoff_weighted) / num_of_matches


def get_team_ova(matches, team, team_ova, home_ovas, away_ovas):
    num_of_matches = matches.shape[0]
    for i in range(0, num_of_matches):
        match = matches.iloc[i, :]
        # In "match" vector, search id via team_api_id
        home_team_id = match['home_team_api_id']
        away_team_id = match['away_team_api_id']

        # In "team" matrix, search team name via team id
        home_row = team[team['team_api_id'] == home_team_id].iloc[0, :]
        home_team_name = home_row['team_long_name']
        away_row = team[team['team_api_id'] == away_team_id].iloc[0, :]
        away_team_name = away_row['team_long_name']

        # In "match" vector, search year via season
        year = match['season']
        # In "team_ova" matrix, search rows with same year via year
        team_ova_years_rows = team_ova[team_ova['YEAR'] == year]
        # In "team_ova_years_rows" matrix, search row via team name
        home_team_ova_row = team_ova_years_rows[team_ova_years_rows['NAME'].str.contains(home_team_name)].iloc[0, :]
        away_team_ova_row = team_ova_years_rows[team_ova_years_rows['NAME'].str.contains(away_team_name)].iloc[0, :]
        #In "team_ova_row" vector, search team ova via OVA
        home_team_ova = home_team_ova_row['OVA']
        away_team_ova = away_team_ova_row['OVA']
        home_ovas[i] = home_team_ova - 50
        away_ovas[i] = away_team_ova - 50
    return (home_ovas, away_ovas)


def get_match_features(match, matches):
    """ Create match specific features for a given match. """

    # Define variables
    date = match.date
    season = match.season
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id

    matches_home_ashome = get_last_ashome_matches(matches, date, season, home_team, x=5)
    # print(matches_home_ashome, flush=True)
    matches_home_asaway = get_last_asaway_matches(matches, date, season, home_team, x=5)
    matches_away_ashome = get_last_ashome_matches(matches, date, season, away_team, x=5)
    matches_away_asaway = get_last_asaway_matches(matches, date, season, away_team, x=5)
    # matches_home = pd.concat([matches_home_ashome, matches_home_asaway])
    # matches_away = pd.concat([matches_away_ashome, matches_away_asaway])

    # Get last x matches of both teams against each other
    competing_matches_same = get_last_competing_matches(matches, date, home_team, away_team, x=2)
    competing_matches_diff = get_last_reverse_competing_matches(matches, date, home_team, away_team, x=2)

    # Define result data frame
    result = pd.Series()
    result['match_api_id'] = match.match_api_id
    result['league_id'] = match.league_id

    home_recent_goal_ashome = get_goals(matches_home_ashome)
    away_recent_goal_asaway = get_goals(matches_away_asaway, isAway=True)
    home_recent_lost_ashome = get_goals_lost(matches_home_ashome)
    away_recent_lost_asaway = get_goals_lost(matches_away_asaway, isAway=True)
    result['home_goal_index'] = home_recent_goal_ashome + away_recent_lost_asaway
    result['away_goal_index'] = away_recent_goal_asaway + home_recent_lost_ashome

    home_recent_goal_asaway = get_goals(matches_home_asaway, isAway=True)
    away_recent_goal_ashome = get_goals(matches_away_ashome)
    home_recent_lost_asaway = get_goals_lost(matches_home_asaway, isAway=True)
    away_recent_lost_ashome = get_goals_lost(matches_away_ashome)
    result['home_goal_index_2'] = home_recent_goal_asaway + away_recent_lost_ashome
    result['away_goal_index_2'] = away_recent_goal_ashome + home_recent_lost_asaway

    result['team_ova_diff'] = match.home_ova - match.away_ova

    # result['recent_score_diff'] \
    #     = get_average_league_score(matches_home_ashome) - get_average_league_score(matches_away_asaway, isAway=True)
    # result['recent_score_diff_2'] \
    #     = get_average_league_score(matches_home_asaway, isAway=True) - get_average_league_score(matches_away_ashome)
    result['home_recent_score'] \
        = get_average_league_score(matches_home_ashome) + get_average_league_score(matches_home_asaway, isAway=True)
    result['away_recent_score'] \
        = get_average_league_score(matches_away_ashome) + get_average_league_score(matches_away_asaway, isAway=True)

    result['competing_same_goal_diff'] = get_goals(competing_matches_same) - get_goals(competing_matches_same, isAway=True)
    result['competing_diff_goal_diff'] = get_goals(competing_matches_diff, isAway=True) - get_goals(competing_matches_diff)

    result['recent_shoton_diff_1'] = get_shoton(matches_home_ashome) - get_shoton(matches_away_asaway, isAway=True)
    result['recent_shoton_diff_2'] = get_shoton(matches_home_asaway, isAway=True) - get_shoton(matches_away_ashome)
    result['recent_shotoff_diff_1'] = get_shotoff(matches_home_ashome) - get_shotoff(matches_away_asaway, isAway=True)
    result['recent_shotoff_diff_2'] = get_shotoff(matches_home_asaway, isAway=True) - get_shotoff(matches_away_ashome)

    # print(result)
    return result


def create_feables(matches, fifa_stats, bookkeepers, verbose=True):
    """ Create and aggregate features and labels for all matches. """

    if verbose:
        print("Generating match features...")
    start = time()

    # Get match features for all matches (apply to each row)
    match_stats = matches.apply(lambda match: get_match_features(match, matches), axis=1)

    # Create dummies for league ID feature
    # deleting this as i am only looking at EPL
    # dummies = pd.get_dummies(match_stats['league_id']).rename(columns=lambda x: 'League_' + str(x))
    # match_stats = pd.concat([match_stats, dummies], axis=1)
    match_stats.drop(['league_id'], inplace=True, axis=1)

    end = time()
    if verbose:
        print("Match features generated in {:.1f} minutes".format((end - start) / 60))

    if verbose:
        print("Generating match labels...")
    start = time()

    # Create match labels
    labels = matches.apply(get_match_label, axis=1)
    end = time()
    if verbose:
        print("Match labels generated in {:.1f} minutes".format((end - start) / 60))

    # if verbose == True:
    #     print("Generating bookkeeper data...")
    # start = time()
    # Get bookkeeper quotas for all matches
    # bk_data = get_bookkeeper_data(matches, bookkeepers, horizontal=True)
    # bk_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    # end = time()
    # if verbose == True:
    #     print("Bookkeeper data generated in {:.1f} minutes".format((end - start) / 60))

    # Merges features and labels into one frame
    features = pd.merge(match_stats, fifa_stats, on='match_api_id', how='left')
    # features = pd.merge(features, bk_data, on='match_api_id', how='left')
    # features = match_stats
    feables = pd.merge(features, labels, on='match_api_id', how='left')

    # Drop NA values
    feables.dropna(inplace=True)

    # Return preprocessed data
    return feables


def train_classifier(clf, X_t, y_t, params, jobs):
    """ Fits a classifier to the training data. """

    # Start the clock, train the classifier, then stop the clock
    start = time()
    # estimators = [('clf', clf)]
    # pipeline = Pipeline(estimators)
    grid_obj = model_selection.GridSearchCV(estimator=clf, param_grid=params, cv=5, n_jobs=jobs)
    best = grid_obj.fit(X_t, y_t)
    # best_pipe = grid_obj.best_estimator_
    end = time()
    # Print the results
    print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))
    print(grid_obj.best_params_)

    # Return best pipe
    return best.best_estimator_


def predict_labels(clf, features, target):
    """ Makes predictions using a fit classifier based on scorer. """

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    # Print and return results
    print("Made predictions in {:.4f} seconds".format(end - start))
    return accuracy_score(target, y_pred)


def train_calibrate_predict(clf, X_t, y_t, X_v, y_v, params, jobs):
    """ Train and predict using a classifier based on scorer. """

    # Indicate the classifier and the training set size
    print("Training a {} with None...".format(clf.__class__.__name__))

    # Train the classifier
    clf = train_classifier(clf, X_t, y_t, params, jobs)

    # # Calibrate classifier
    # print("Calibrating probabilities of classifier...")
    # start = time()
    # clf = CalibratedClassifierCV(best_pipe.named_ste ps['clf'], cv='prefit', method='isotonic')
    # clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
    # end = time()
    # print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))

    # Print the results of prediction for both training and testing
    train_score = predict_labels(clf, X_t, y_t)
    test_score = predict_labels(clf, X_v, y_v)
    print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__, train_score))
    print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, test_score))

    # Return classifier, and score for train and test set
    return clf, train_score, test_score


def find_best_classifier(classifiers, X_t, y_t, X_v, y_v, params, jobs):
    """ Tune all classifier and dimensionality reduction combiantions to find best classifier. """

    # Initialize result storage
    clfs_return = []
    train_scores = []
    test_scores = []

    # Loop through classifiers
    for classifier in classifiers:
        # Grid search, calibrate, and test the classifier
        classifier, train_score, test_score = train_calibrate_predict(
            classifier, X_t, y_t, X_v, y_v, params[classifier], jobs)

        # Append the result to storage
        clfs_return.append(classifier)
        train_scores.append(train_score)
        test_scores.append(test_score)

    # Return storage
    return clfs_return, train_scores, test_scores


def plot_training_results(clfs, train_scores, test_scores):
    """ Plot results of classifier training. """

    # Set graph format
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1, rc={"lines.linewidth": 1})
    ax = plt.subplot(111)
    w = 0.5
    x = np.arange(len(train_scores))
    ax.set_yticks(x + w)
    ax.legend((train_scores[0], test_scores[0]), ("Train Scores", "Test Scores"))
    names = []

    # Loop throuugh classifiers
    for i in range(0, len(clfs)):
        # Define temporary variables
        clf = clfs[i]
        clf_name = clf.__class__.__name__

        # Create and store name
        name = "{}".format(clf_name)
        names.append(name)

    # Plot all names in horizontal bar plot
    ax.set_yticklabels((names))
    plt.xlim(0.5, 0.57)
    plt.barh(x, test_scores, color='b', alpha=0.6)
    plt.title("Test Data Accuracy Scores")
    fig = plt.figure(1)

    plt.show()


def explore_data(features, inputs, path):
    ''' Explore data by plotting KDE graphs. '''

    # Define figure subplots
    fig = plt.figure(1)
    # fig.subplots_adjust(bottom=-1, left=0.025, top=2, right=0.975)

    # Loop through features
    i = 1
    for col in features.columns:
        # Set subplot and plot format
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=0.5, rc={"lines.linewidth": 1})
        plt.subplot(3, 5, 0 + i)
        j = i - 1

        # Plot KDE for all labels
        sns.distplot(inputs[inputs['label'] == 'Win'].iloc[:, j], hist=False, label='Win')
        sns.distplot(inputs[inputs['label'] == 'Draw'].iloc[:, j], hist=False, label='Draw')
        sns.distplot(inputs[inputs['label'] == 'Defeat'].iloc[:, j], hist=False, label='Defeat')
        plt.legend();
        i = i + 1

    # Define plot format
    DefaultSize = fig.get_size_inches()
    fig.set_size_inches((DefaultSize[0] * 1.0, DefaultSize[1] * 1.0))

    plt.show()

    # Compute and print label weights
    labels = inputs.loc[:, 'label']
    class_weights = labels.value_counts() / len(labels)
    print(class_weights)

    # Store description of all features
    feature_details = features.describe().transpose()

    # Return feature details
    return feature_details


# print(time_discount_vector(6))
# a = np.zeros(6)
# a[[True, True, False, False, True, False]] = 1
# print(a, flush=True)

start = time()
## Fetching data
# Connecting to database
path = "./input/"  # Insert path here
database = path + 'database.sqlite'
conn = sqlite3.connect(database)

# Defining the number of jobs to be run in parallel during grid search
n_jobs = 1  # Insert number of parallel jobs here

# Fetching required data tables
player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)

# Fetching required csv data
team_OVA = pd.read_csv("./input/team_OVA.csv")

# Reduce match data to fulfill run time requirements
rows = ["league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
        "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
        "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
        "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
        "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
        "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]
match_data.dropna(subset=rows, inplace=True)
match_data = match_data.loc[lambda df: df.league_id == 1729, :]
# match_data = match_data.tail(100)
# print(match_data)

## Generating features, exploring the data, and preparing data for model training
# Generating or retrieving already existant FIFA data
fifa_data = get_fifa_data(match_data, player_stats_data, data_exists=False)

# Appending team overall rating
h_ovas = np.full([match_data.shape[0]], np.nan)
a_ovas = np.full([match_data.shape[0]], np.nan)
get_team_ova(match_data, team_data, team_OVA, h_ovas, a_ovas)
match_data['home_ova'] = h_ovas
match_data['away_ova'] = a_ovas
# print(match_data)

# Creating features and labels based on data provided
bk_cols = ['B365', 'BW', 'IW', 'LB', 'PS', 'WH', 'SJ', 'VC', 'GB', 'BS']
bk_cols_selected = ['B365', 'BW']
feables = create_feables(match_data, fifa_data, bk_cols_selected)
inputs = feables.drop('match_api_id', axis=1)

# Exploring the data and creating visualizations
labels = inputs.loc[:, 'label']
features = inputs.drop('label', axis=1)
# print(features.head(5))
print(inputs.head(5))
feature_details = explore_data(features.iloc[:, :12], inputs, path)

# standardized
scalar = preprocessing.StandardScaler()
features = scalar.fit_transform(features)
# print(features)
# print(features.shape)

# Splitting the data into Train, Calibrate, and Test data sets
# X_train, X_test, y_train, y_test = train_test_split(features.values,
#                                                     labels.values, test_size=0.3,
#                                                     stratify=labels.values, random_state=987)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                    stratify=labels.values, random_state=987)

# ros = RandomOverSampler()
# X_resampled, y_resampled = ros.fit_sample(X_train, y_train)

# Creating cross validation data splits
# seem not necessary, just set cv=5 when training is fine
# cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
# cv_sets.get_n_splits(X_train, y_train)

## Initializing all models and parameters
# Initializing classifiers
RF_clf = RandomForestClassifier(oob_score=True, max_depth=5, min_samples_leaf=2)
AB_clf = AdaBoostClassifier()
GNB_clf = GaussianNB()
KNN_clf = KNeighborsClassifier()
LOG_clf = linear_model.LogisticRegression()
SVC_clf = SVC()
clfs = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf, SVC_clf]

# Specficying parameters for grid search

parameters_RF = {'max_features': ['auto'], 'n_estimators': [800, 1000, 1500, 2000, 2500]}
parameters_AB = {'learning_rate': np.linspace(0.5, 2, 5), 'n_estimators': [5, 10, 30, 50, 100, 200]}
parameters_GNB = {}
parameters_KNN = {'n_neighbors': [5, 10, 15, 20, 35, 50]}
parameters_LOG = {'C': np.logspace(1, 4, 8), 'penalty': ['l1', 'l2']}
parameters_SVC = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}

parameters = {clfs[0]: parameters_RF,
              clfs[1]: parameters_AB,
              clfs[2]: parameters_GNB,
              clfs[3]: parameters_KNN,
              clfs[4]: parameters_LOG,
              clfs[5]: parameters_SVC}

## Training a baseline model and finding the best model composition using grid search
# Train a simple GBC classifier as baseline model
clf_bl = LOG_clf
clf_bl.fit(X_train, y_train)
print("Score of {} for training set: {:.4f}.".format(clf_bl.__class__.__name__,
                                                     accuracy_score(y_train, clf_bl.predict(X_train))))
print("Score of {} for test set: {:.4f}.".format(clf_bl.__class__.__name__,
                                                 accuracy_score(y_test, clf_bl.predict(X_test))))

# Training all classifiers and comparing them
clfs, train_scores, test_scores = find_best_classifier(clfs, X_train, y_train, X_test, y_test, parameters, n_jobs)

# Plotting train and test scores
plot_training_results(clfs, np.array(train_scores), np.array(test_scores))


def plot_confusion_matrix(y_test, X_test, clf, normalize=False):
    ''' Plot confusion matrix for given classifier and data. '''

    # Define label names and get confusion matrix values
    labels = ["Win", "Draw", "Defeat"]
    cm = confusion_matrix(y_test, clf.predict(X_test), labels)

    # Check if matrix should be normalized
    if normalize == True:
        # Normalize
        cm = cm.astype('float') / cm.sum()

    # Configure figure
    sns.set_style("whitegrid", {"axes.grid": False})
    fig = plt.figure(1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    title = "Confusion matrix of a {} with None".format(clf.__class__.__name__)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()

    # Print classification report
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


for clf in clfs:
    plot_confusion_matrix(y_test, X_test, clf, normalize = True)
