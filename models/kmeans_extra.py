'''

__file__

    kmeans_extra.py

__description__

    This file is used to preprocess dataframe by grouping opponents by kMeans.
    Averages bref variables based on kMeans groups. This file does another
    clustering on top of kmeans_build.py

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

params_km = {'n_clusters': 5,
             'max_iter': 10000,
             'n_init': 10,
             'init': 'k-means++',
             'precompute_distances': 'auto',
             'tol': 0.0001,
             'n_jobs': 1,
             'verbose': 0}

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
    """
    Used many times
    """
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


def cluster_stats(n = 3):
    '''
    Takes in number of clusters and automatically works on previously clustered
    dataset (this function is for extra clustering based on fewer clusters)

    Returns a new extra clustered dataframe that averages bref stats
    over each cluster for the months not in March or April.
    (subsetting done later)
    '''
    global params_km, columns
    params_km['n_clusters'] = n
    print params_km.items()
    brefcol = ['minutes', 'num_poss', 'opp_poss', 'pace_bref',
            'fg_pm', 'fga_pm', 'fg_percent', 'TP_pm',
            'TPA_pm', 'TP_percent', 'eFG', 'FT_pm',
            'FTA_pm', 'FT_percent', 'cluster']

    db = '../sql/nba_stats.db'
    #db = 'sql/nba_stats.db'
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

    d = nba_df[['opponent','cluster']].copy()
    df = pd.read_csv('../csv_data/nba_15season_all_150928.csv')
    #df = pd.read_csv('../csv_data/testing.csv')
    df = pd.merge(df, d, how = 'left', on = 'opponent')
    df_std = df.copy()
    df_max = df.copy()
    df_min = df.copy()

    for lineup in df['lineup'].unique():
        t_df = df[(df['lineup'] == lineup) &
                  np.logical_not(df['month'].isin([3,4]))]
        for x in t_df['cluster'].unique():
            temp_df = pd.DataFrame(t_df.loc[t_df['cluster'] == x,
                                   brefcol[:-1]].apply(np.mean, axis = 0))
            temp_df = temp_df.T
            temp_df = temp_df.as_matrix(brefcol[:-1])
            df.loc[(df['lineup']==lineup) & (df['cluster']==x),
                        brefcol[:-1]] = temp_df
            """
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
            """
    #df = df.drop('cluster', axis = 1)

    # Just get a subset of the columns for bref
    df = df[brefcol[:-1]]
    df.columns = [col + '_' + str(n) + 'mean' for col in df.columns]

    """
    df_std = df_std[brefcol[:-1]]
    df_std.columns = [col + '_' + str(n) + 'std' for col in df_std.columns]

    df_max = df_max[brefcol[:-1]]
    df_max.columns = [col + '_' + str(n) + 'max' for col in df_max.columns]

    df_min = df_min[brefcol[:-1]]
    df_min.columns = [col + '_' + str(n) + 'min' for col in df_min.columns]

    print 'df_mean shape:', df.shape
    print 'df_std shape:', df_std.shape
    print 'df_max shape:', df_max.shape
    print 'df_min shape:', df_min.shape
    """
    print 'df_mean shape:', df.shape
    df_ = pd.read_csv('../csv_data/nba_15season_kMeans29_150930.csv')
    #df_ = pd.read_csv('../csv_data/nba_15season_kMeans30all_1510.csv')
    df_ = df_.reset_index()
    df_ = pd.merge(df_, df.reset_index(), how = 'left', on = 'index')
    """
    df_ = pd.merge(df_, df_std.reset_index(), how = 'left', on = 'index')
    df_ = pd.merge(df_, df_max.reset_index(), how = 'left', on = 'index')
    df_ = pd.merge(df_, df_min.reset_index(), how = 'left', on = 'index')
    """
    df_ = df_.drop('index', axis = 1)

    """
    # examine lineups that play a team multiple times
    pdf = df_.copy()
    sublist = []
    for lineup in df['lineup'].unique():
        d = df.loc[df['lineup'] == lineup, 'opponent'].value_counts()
        subset = list(d[d>1].index)
        sublist += list(pdf[(pdf['lineup'] == lineup) &
                            (pdf['opponent'].isin(subset))].index)
    df_ = df_.loc[sublist,:]
    """

    columns = list(df_.columns)
    columns.remove('points')
    columns.append('points')
    df_ = df_[columns]
    return df_


def send_to_csv(name = 'nba_15season_kMeans30all3_150930', n = 3):
    '''
    Function to make csv from the clustering analysis

    takes in: file name

    makes csv file in 'csv_data/' folder
    '''
    global params_km, columns
    name = name + '.csv'
    file_name = os.path.join('..', 'csv_data', name)
    df = cluster_stats(n = n)
    df.to_csv(file_name, index = False)
    print 'Dataframe shape:', df.shape
    print 'Made csv file:', file_name


if __name__ == '__main__':
    print 'Start extra KMeans build...'
    send_to_csv(name = 'nba_15season_kMeans30all10_1510', n = 10)
    print 'done'
