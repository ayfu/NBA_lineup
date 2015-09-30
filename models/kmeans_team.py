'''

__file__

    kmeans_team.py

__description__

    This file provides methods to perform kMeans analysis on nba_team_data
    dataframes ('nba_opponent_allstats_2015' in the nba_stats.db)

'''

import sys, os
from collections import defaultdict
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.cluster import KMeans

nb_cols = ['PLUS_MINUS_opt', 'OFF_RATING_opt', 'DEF_RATING_opt', 'PACE_opt']

params_km = {'n_clusters': 3,
             'max_iter': 10000,
             'n_init': 10,
             'init': 'k-means++',
             'precompute_distances': 'auto',
             'tol': 0.0001,
             'n_jobs': 1,
             'verbose': 0}



def km_cluster(df, params_km, n = 3, rounds = 10,
               columns = ['PLUS_MINUS_opt', 'PACE_opt']):
    '''
    Takes in dataframe (df), (n) number of clusters to explore,
    (rounds) number of rounds, and columns

    Returns a dataframe of the average and the standard deviation
    of value_counts()
    '''
    params_km['n_clusters'] = n
    dist = np.zeros(n)
    p = pd.DataFrame({'pred_0': dist})
    for x in range(rounds):
        est = KMeans(**params_km)
        X = df.as_matrix(columns).astype(float)
        est.fit(X)
        pred_km = est.predict(X)
        p['pred_'+str(x)] = np.array(pd.Series(pred_km).value_counts())
    results = pd.DataFrame({'mean': p.apply(np.mean, axis = 1),
                            'std': p.apply(np.std, axis = 1)})
    return results

def add_cluster(df, columns, params_km):
    '''
    Takes in: dataframe (df), columns to analyze, kMeans parameters (params_km)

    Returns: Dataframe with a new column for the cluster (to help categorize)
    '''
    est = KMeans(**params_km)
    X = df.as_matrix(columns).astype(float)
    est.fit(X)
    pred_km = est.predict(X)
    df['cluster'] = pred_km
    return df
