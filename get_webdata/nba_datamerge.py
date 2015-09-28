'''

__file__

    nba_datamerge.py

__description__

    This file takes all nba and bball-ref tables from nba_stats.db and merges
    them together. It renames columns to resolve conflicts and adds year,
    month, dayofmonth, and day columns. It then merges and stores the large
    table in nba_stats.db and as a csv in csv_data/ folder.

    The dataframe has not been encoded yet (encoding.py)
'''

import pandas as pd
import numpy as np
import sqlite3
import sys, os
import datetime as dt
from collections import defaultdict

from nba_lineups import *
from bballref import *
sys.path.append(os.path.abspath("../sql/"))
from get_tables import *


all_teams = ['PHI','MIL','CHI','CLE','BOS','LAC','MEM','ATL','MIA','CHA',
             'UTA','SAC','NYK','LAL','ORL','DAL','NJN','DEN','IND','NOH',
             'DET','TOR','HOU','SAS','PHO','OKC','MIN','POR','GSW','WAS']

nba_bestcols = ['lineup', 'TEAM_ABBREVIATION', 'GP', 'MIN',
            'PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT', 'PCT_PTS_2PT_MR',
            'PCT_PTS_3PT', 'PCT_PTS_FB', 'PCT_PTS_FT', 'PCT_PTS_OFF_TOV',
            'PCT_PTS_PAINT', 'PCT_AST_2PM', 'PCT_UAST_2PM', 'PCT_AST_3PM',
            'PCT_UAST_3PM', 'PCT_AST_FGM', 'PCT_UAST_FGM', 'EFG_PCT',
            'FTA_RATE', 'TM_TOV_PCT', 'OREB_PCT', 'OPP_EFG_PCT', 'OPP_FTA_RATE',
            'OPP_TOV_PCT', 'OPP_OREB_PCT', 'PTS_OFF_TOV', 'PTS_2ND_CHANCE',
            'PTS_FB', 'PTS_PAINT', 'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE',
            'OPP_PTS_FB', 'OPP_PTS_PAINT', 'FGM', 'FGA', 'FG_PCT', 'FG3M',
            'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
            'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS',
            'PLUS_MINUS', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'AST_PCT',
            'AST_TO', 'AST_RATIO', 'DREB_PCT', 'REB_PCT', 'TS_PCT', 'PACE',
            'PIE', 'OPP_FGM', 'OPP_FGA', 'OPP_FG_PCT', 'OPP_FG3M', 'OPP_FG3A',
            'OPP_FG3_PCT', 'OPP_FTM', 'OPP_FTA', 'OPP_FT_PCT', 'OPP_OREB',
            'OPP_DREB', 'OPP_REB', 'OPP_AST', 'OPP_TOV', 'OPP_STL', 'OPP_BLK',
            'OPP_BLKA', 'OPP_PF', 'OPP_PFD', 'OPP_PTS']

db = "../sql/nba_stats.db"
df_dict_bref = {}
df_dict_nba = {}


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

def get_tables(db):
    '''
    Takes in: database file (db)
    Returns: list of tables in database
    '''
    #len(re.findall("[\w]+.db", db)) > 0
    # Build a list of all *.db files

    temp = dbConnect(db)
    with temp:
        temp.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = temp.cur.fetchall()
        tables = pd.Series([str(tables[x][0]) for x in range(len(tables))])
        return tables

def main_tables(somestats = True):
    global db, df_dict_bref, df_dict_nba

    tables = get_tables(db)
    #tables = pd.Series([str(tables[x][0]) for x in range(len(tables))])
    tables_nba = tables[tables.str.contains('nba')]
    tables_bballref = tables[tables.str.contains('bballref')]

    temp = dbConnect(db)
    with temp:
        # Get tables for bball-ref store in a dict
        for table in tables_bballref:
            name = table.split('_')[1]
            sql = "SELECT * FROM " + table
            df_temp = pd.read_sql_query(sql, temp.con)
            df_dict_bref[name] = df_temp
        # Get tables from stats.nba.com store in a dict
        for table in tables_nba:
            name = table.split('_')[1]
            sql = "SELECT * FROM " + table
            df_temp = pd.read_sql_query(sql, temp.con)
            df_dict_nba[name] = df_temp

        # Get tables from stats.nba.com for team data
        somestat = "SELECT * FROM nba_opponent_somestats_2015"
        allstat = "SELECT * FROM nba_opponent_allstats_2015"
        some_nba = pd.read_sql_query(somestat, temp.con)
        all_nba = pd.read_sql_query(allstat, temp.con)

    some_nba = some_nba.drop('index', axis = 1)
    all_nba = all_nba.drop('index', axis = 1)
    df_bref = pd.concat(df_dict_bref.values(), axis = 0)
    df_nba = pd.concat(df_dict_nba.values(), axis = 0)

    df_bref = df_bref.drop(['index', 'id'], axis = 1)
    df_nba = df_nba.drop('index', axis = 1)

    bref_cols = ['num_poss', 'opp_poss', 'fg', 'fga', 'TP',
                 'TPA', 'FT', 'FTA', 'points']
    for col in bref_cols:
        df_bref[col] = df_bref[col]/df_bref['minutes'].astype(float)*48.0

    # Change NJN to BRK (New Jersey Nets to Brooklyn Nets)
    df_bref['team'] = df_bref['team'].astype(str)
    df_bref['opponent'] = df_bref['opponent'].astype(str)
    df_bref.loc[df_bref['team'] == 'BRK', 'team'] = 'BKN'
    df_bref.loc[df_bref['opponent'] == 'BRK', 'opponent'] = 'BKN'
    df_bref.loc[df_bref['team'] == 'CHO', 'team'] = 'CHA'
    df_bref.loc[df_bref['opponent'] == 'CHO', 'opponent'] = 'CHA'
    # Rename columns to note conflict with NBA.com
    # *_pm meaning plusminus
    df_bref = df_bref.rename(columns = {'fg': 'fg_pm',
                                        'fga': 'fga_pm',
                                        'TP': 'TP_pm',
                                        'TPA': 'TPA_pm',
                                        'FT': 'FT_pm',
                                        'FTA': 'FTA_pm',
                                        'pace': 'pace_bref'})

    nba_cols = ['PTS_OFF_TOV', 'PTS_2ND_CHANCE', 'PTS_FB', 'PTS_PAINT',
                'OPP_PTS_OFF_TOV', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_FB',
                'OPP_PTS_PAINT',
                'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
                'REB', 'AST', 'TOV', 'STL', 'BLK',
                'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS',
                'OPP_FGM', 'OPP_FGA', 'OPP_FG3M', 'OPP_FG3A', 'OPP_FTM',
                'OPP_FTA', 'OPP_OREB', 'OPP_DREB', 'OPP_REB', 'OPP_AST',
                'OPP_TOV', 'OPP_STL', 'OPP_BLK', 'OPP_BLKA', 'OPP_PF',
                'OPP_PFD', 'OPP_PTS']
    for col in nba_cols:
        df_nba[col] = df_nba[col]/df_nba['MIN'].astype(float)*48.0

    #df = pd.merge(df_bref, df_nba, how = 'left', on = 'lineup')

    # Subset out lineups with multiple entries (64 entries/lineup)
    multi_entry = df_nba['lineup'].value_counts()
    multi_entry = list(multi_entry[multi_entry > 1].index)
    df_nba_1 = df_nba[np.logical_not(df_nba['lineup'].isin(multi_entry))]

    # Include the first entry of each multi-lineup entry
    df_list = []
    for entry in multi_entry:
        #temp = df_nba[df_nba['lineup'].isin(multi_entry)]
        temp = df_nba[df_nba['lineup'] == entry]
        temp = temp.reset_index().drop('index', axis = 1)
        temp = temp.iloc[0]
        df_list += [temp]
    df_nba_2 = pd.DataFrame(df_list)
    df_nba_2 = df_nba_2.reset_index().drop('index', axis = 1)

    # Make full lineup again shape should be (7446, 90)
    df_nba = pd.concat([df_nba_1, df_nba_2], axis = 0)
    #df_nba = df_nba.drop(['W', 'L', 'W_PCT'], axis = 1) #[nba_bestcols]
    df_nba = df_nba[nba_bestcols]
    df = pd.merge(df_bref, df_nba, how = 'left', on = 'lineup')

    # Incorporating opponent data
    if somestats:
        df = pd.merge(df, some_nba, how = 'left', on = 'opponent')
    else:
        df = pd.merge(df, all_nba, how = 'left', on = 'opponent')

    # Formatting dataframe
    df['date'] = pd.to_datetime(df.date, format = '%Y-%m-%d %H:%M:%S')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    df['day'] = [date.days for date in df['date'] - dt.date(2014,10,1)]
    columns = list(df.columns)
    columns.remove('points')
    columns.append('points')
    df = df[columns]

    return df

def send_to_csv(name = 'nba_15season_some_150928', somestats = True):
    '''
    Function to make csv from full merge of stats.nba.com and bballref

    takes in: file name

    makes csv file in 'csv_data/' folder
    '''
    name = name + '.csv'
    file_name = os.path.join('..', 'csv_data', name)
    nba_df = main_tables(somestats = somestats)
    nba_df.to_csv(file_name, index = False)
    print 'Dataframe shape:', nba_df.shape
    print 'Made csv file:', file_name

def send_to_sql(name = 'nba_15season_150920'):
    '''
    Function to make sql table from full merge of stats.nba.com and bballref

    takes in: file name

    makes csv file in 'sql/' folder
    '''
    file_name = os.path.join('sql', name)
    nba_df = main_tables(somestats = True)
    temp = dbConnect("../sql/nba_stats.db")
    with temp:
        nba_df.to_sql(name, temp.con, flavor = 'sqlite')
    print 'Made table in sql/nba_stats.db:', name

def read_sql(name = 'nba_15season_150920'):
    '''
    Get full nba_15season_150920 table

    Returns a dataframe
    '''
    temp = dbConnect(db)
    with temp:
        sql = "SELECT * FROM " + name
        df = pd.read_sql_query(sql, temp.con)
        df = df.drop('index', axis = 1)
    return df


if __name__ == '__main__':
    send_to_csv(name = 'nba_15season_all_150928', somestats = False)
    print 'done'
