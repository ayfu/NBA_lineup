'''

__file__

    kmeans_build.py

__description__

    This file is used to preprocess dataframe by grouping opponents by kMeans.
    Averages bref variables based on kMeans groups.

    Returns a new dataframe based on kMeans analysis for bref features for the
    months before March and April

'''

import sys
import os
from collections import defaultdict
import datetime as dt
import sqlite3

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

params_km = {'n_clusters': 30,
             'max_iter': 10000,
             'n_init': 10,
             'init': 'k-means++',
             'precompute_distances': 'auto',
             'tol': 0.0001,
             'n_jobs': 1,
             'verbose': 0}
#columns = ['PLUS_MINUS_opt', 'OFF_RATING_opt', 'DEF_RATING_opt', 'PACE_opt']

columns = ['W_opt', 'L_opt', 'W_PCT_opt', 'PCT_FGA_2PT_opt', 'PCT_FGA_3PT_opt',
           'PCT_PTS_2PT_opt', 'PCT_PTS_2PT_MR_opt', 'PCT_PTS_3PT_opt',
           'PCT_PTS_FB_opt', 'PCT_PTS_FT_opt', 'PCT_PTS_OFF_TOV_opt',
           'PCT_PTS_PAINT_opt', 'PCT_AST_2PM_opt', 'PCT_UAST_2PM_opt',
           'PCT_AST_3PM_opt', 'PCT_UAST_3PM_opt', 'PCT_AST_FGM_opt',
           'PCT_UAST_FGM_opt', 'EFG_PCT_opt', 'FTA_RATE_opt', 'TM_TOV_PCT_opt',
           'OREB_PCT_opt', 'OPP_EFG_PCT_opt', 'OPP_FTA_RATE_opt',
           'OPP_TOV_PCT_opt', 'OPP_OREB_PCT_opt', 'PTS_OFF_TOV_opt',
           'PTS_2ND_CHANCE_opt', 'PTS_FB_opt', 'PTS_PAINT_opt',
           'OPP_PTS_OFF_TOV_opt', 'OPP_PTS_2ND_CHANCE_opt', 'OPP_PTS_FB_opt',
           'OPP_PTS_PAINT_opt', 'FGM_opt', 'FGA_opt', 'FG_PCT_opt', 'FG3M_opt',
           'FG3A_opt', 'FG3_PCT_opt', 'FTM_opt', 'FTA_opt', 'FT_PCT_opt',
           'OREB_opt', 'DREB_opt', 'REB_opt', 'AST_opt', 'TOV_opt', 'STL_opt',
           'BLK_opt', 'BLKA_opt', 'PF_opt', 'PFD_opt', 'PTS_opt',
           'PLUS_MINUS_opt', 'OFF_RATING_opt', 'DEF_RATING_opt',
           'NET_RATING_opt', 'AST_PCT_opt', 'AST_TO_opt', 'AST_RATIO_opt',
           'DREB_PCT_opt', 'REB_PCT_opt', 'TS_PCT_opt', 'PACE_opt', 'PIE_opt',
           'OPP_FGM_opt', 'OPP_FGA_opt', 'OPP_FG_PCT_opt', 'OPP_FG3M_opt',
           'OPP_FG3A_opt', 'OPP_FG3_PCT_opt', 'OPP_FTM_opt', 'OPP_FTA_opt',
           'OPP_FT_PCT_opt', 'OPP_OREB_opt', 'OPP_DREB_opt', 'OPP_REB_opt',
           'OPP_AST_opt', 'OPP_TOV_opt', 'OPP_STL_opt', 'OPP_BLK_opt',
           'OPP_BLKA_opt', 'OPP_PF_opt', 'OPP_PFD_opt', 'OPP_PTS_opt']

class dbConnect():
    '''
    Class to help with context management for 'with' statements.

    Connections are always committed with general with statements.
    This will ensure that every time I use a 'with' statement, the database
    connection will close automatically (__exit__ statement).

    http://www.webmasterwords.com/python-with-statement

    temp = dbConnect("sql/nba_stats.db")
    with temp:
        df.to_sql('nbadotcom', temp.con, flavor = 'sqlite')
    '''
    def __init__(self, fileName):
        self.fileName = fileName
        self.con = sqlite3.connect(self.fileName)
        self.cur = self.con.cursor()
    def __enter__(self):
        self.con = sqlite3.connect(self.fileName)
        self.cur = self.con.cursor()
    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.con.close()

def make_df(params_km = params_km, columns = columns):
    '''
    Takes in kMeans parameters and columns to perform kMeans

    Returns a merged dataframe with a new 'cluster' column (from kMeans)
    '''
    db = '../sql/nba_stats.db'
    temp = dbConnect(db)
    with temp:
        sql = 'SELECT * FROM nba_opponent_allstats_2015'
        nba_df = pd.read_sql_query(sql, temp.con)
        nba_df = nba_df.drop('index', axis = 1)

    est = KMeans(**params_km)
    X = nba_df.as_matrix(columns).astype(float)
    est.fit(X)
    pred_km = est.predict(X)
    nba_df['cluster'] = pred_km

    bigdf = pd.read_csv('../csv_data/nba_15season_all_150928.csv')
    #bigdf = pd.read_csv('../csv_data/testing.csv')
    d = nba_df[['opponent','cluster']].copy()
    bigdf = pd.merge(bigdf, d, how = 'left', on = 'opponent')
    # Add redundant column (using minutes_pm column to subset later)
    # the 'minutes' column will be converted to avg minutes based on cluster
    bigdf['minutes_pm'] = bigdf['minutes']
    columns = list(bigdf.columns)
    columns.remove('points')
    columns.append('points')
    bigdf = bigdf[columns]

    return bigdf

def cluster_features(bigdf):
    '''
    Takes in dataframe from make_df()

    Returns a new dataframe that averages bref stats over each cluster for the
    months not in March or April. (subsetting done later)
    '''
    brefcol = ['minutes', 'num_poss', 'opp_poss', 'pace_bref',
                'fg_pm', 'fga_pm', 'fg_percent', 'TP_pm',
                'TPA_pm', 'TP_percent', 'eFG', 'FT_pm',
                'FTA_pm', 'FT_percent', 'cluster']

    df = bigdf.copy()
    df_std = bigdf.copy()
    df_max = bigdf.copy()
    df_min = bigdf.copy()

    for lineup in df['lineup'].unique():
        t_df = df[(df['lineup'] == lineup) &
                  np.logical_not(df['month'].isin([3,4]))]
        for x in t_df['cluster'].unique():
            temp_df = pd.DataFrame(t_df.loc[t_df['cluster'] == x,
                                   brefcol[:-1]].apply(np.mean, axis = 0))
            temp_df = temp_df.T
            temp_df = temp_df.as_matrix(brefcol[:-1])
            df.loc[(df['lineup'] == lineup) & (df['cluster'] == x),
                    brefcol[:-1]] = temp_df

            # Making stdev columns
            temp_std = pd.DataFrame(t_df.loc[t_df['cluster'] == x,
                                   brefcol[:-1]].apply(np.std, axis = 0))
            temp_std = temp_std.T
            temp_std = temp_std.as_matrix(brefcol[:-1])
            df_std.loc[(df_std['lineup'] == lineup) & (df_std['cluster'] == x),
                    brefcol[:-1]] = temp_std

            # Making max columns
            temp_max = pd.DataFrame(t_df.loc[t_df['cluster'] == x,
                                   brefcol[:-1]].apply(np.max, axis = 0))
            temp_max = temp_max.T
            temp_max = temp_max.as_matrix(brefcol[:-1])
            df_max.loc[(df_max['lineup'] == lineup) & (df_max['cluster'] == x),
                    brefcol[:-1]] = temp_max

            # Making min columns
            temp_min = pd.DataFrame(t_df.loc[t_df['cluster'] == x,
                                   brefcol[:-1]].apply(np.min, axis = 0))
            temp_min = temp_min.T
            temp_min = temp_min.as_matrix(brefcol[:-1])
            df_min.loc[(df_min['lineup'] == lineup) & (df_min['cluster'] == x),
                    brefcol[:-1]] = temp_min

    df = df.drop('cluster', axis = 1)

    df_std = df_std[brefcol[:-1]]
    df_std.columns = [col + '_std' for col in df_std.columns]

    df_max = df_max[brefcol[:-1]]
    df_max.columns = [col + '_max' for col in df_max.columns]

    df_min = df_min[brefcol[:-1]]
    df_min.columns = [col + '_min' for col in df_min.columns]

    print 'df shape:', df.shape
    print 'df_std shape:', df_std.shape
    print 'df_max shape:', df_max.shape
    print 'df_min shape:', df_min.shape

    df = df.reset_index()
    df = pd.merge(df, df_std.reset_index(), how = 'left', on = 'index')
    df = pd.merge(df, df_max.reset_index(), how = 'left', on = 'index')
    df = pd.merge(df, df_min.reset_index(), how = 'left', on = 'index')
    df = df.drop('index', axis = 1)

    columns = list(df.columns)
    columns.remove('points')
    columns.append('points')
    df = df[columns]
    return df

def send_to_csv(name = 'nba_15season_kMeans_150929'):
    '''
    Function to make csv from the clustering analysis

    takes in: file name

    makes csv file in 'csv_data/' folder
    '''
    global params_km, columns
    name = name + '.csv'
    file_name = os.path.join('..', 'csv_data', name)
    bigdf = make_df(params_km, columns)
    df = cluster_features(bigdf)
    df.to_csv(file_name, index = False)
    print 'Dataframe shape:', df.shape
    print 'Made csv file:', file_name


if __name__ == '__main__':
    print 'Start KMeans build...'
    send_to_csv(name = 'nba_15season_kMeans30all_1510')
    print 'done'
