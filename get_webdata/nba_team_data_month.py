'''

__file__

    nba_team_data_month.py

__description__

    This file uses gathers monthly team data from stats.nba.com and sends it to
    sql/nba_stats.db

'''



import sys, os
from collections import defaultdict
import pandas as pd
import numpy as np
import datetime as dt
import requests
from bs4 import BeautifulSoup
import re
import json
import sqlite3
sys.path.append(os.path.abspath("../sql/"))
from get_tables import get_tables


def team_months(team = 'GSW'):

    #team_id = {team_name[x]: all_teams[x] for x in range(len(team_name))}

    types = ['Base', 'Advanced', 'Four+Factors', 'Misc', 'Scoring', 'Opponent']
    stype = types[0] #change this value for other stat types
    seasons = [#'1996-97', '1997-98', '1998-99', '1999-00',
               '2000-01', '2001-02', '2002-03', '2003-04', '2004-05',
               '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
               '2010-11', '2011-12', '2012-13', '2013-14', '2014-15']
    playoffs = ['Regular+Season', 'Playoffs']

    teams = {'ATL': str(1610612737), 'BOS': str(1610612738), 'BRK': str(1610612751),
         'CHA': str(1610612766), 'CHI': str(1610612741), 'CLE': str(1610612739),
         'DAL': str(1610612742), 'DEN': str(1610612743), 'DET': str(1610612765),
         'GSW': str(1610612744), 'HOU': str(1610612745), 'IND': str(1610612754),
         'LAC': str(1610612746), 'LAL': str(1610612747), 'MEM': str(1610612763),
         'MIA': str(1610612748), 'MIL': str(1610612749), 'MIN': str(1610612750),
         'NOH': str(1610612740), 'NYK': str(1610612752), 'OKC': str(1610612760),
         'ORL': str(1610612753), 'PHI': str(1610612755), 'PHO': str(1610612756),
         'POR': str(1610612757), 'SAC': str(1610612758), 'SAS': str(1610612759),
         'TOR': str(1610612761), 'UTA': str(1610612762), 'WAS': str(1610612764)}

    url = ['http://stats.nba.com/stats/teamdashboardbygeneralsplits?DateFrom=',
           'DateTo=',
           'GameSegment=',
           'LastNGames=0',
           'LeagueID=00',
           'Location=',
           'MeasureType='+stype,
           'Month=0',
           'OpponentTeamID=0',
           'Outcome=',
           'PORound=0',
           'PaceAdjust=N',
           'PerMode=PerGame',
           'Period=0',
           'PlusMinus=N',
           'Rank=N',
           'Season=2014-15',
           'SeasonSegment=',
           'SeasonType=Regular+Season',
           'ShotClockRange=',
           'TeamID=' + teams[team],
           'VsConference=',
           'VsDivision=']


    """
    url = ['http://stats.nba.com/stats/teaminfocommon?LeagueID=00',
           'SeasonType=Regular+Season',
           'TeamID=1610612744',
           'season=2014-15']
    """

    url = ('&').join(url)

    r = requests.get(url)
    dat = json.loads(r.text)
    header = dat['resultSets'][3]['headers']
    data = dat['resultSets'][3]['rowSet']
    header = [x + '_m' for x in header]
    df = pd.DataFrame(data, columns = header)
    df['month_m'] = pd.Series([10, 11, 12, 1, 2, 3, 4])
    df = df.drop(['GROUP_SET_m', 'GROUP_VALUE_m', 'SEASON_MONTH_NAME_m',
                  'CFPARAMS_m', 'CFID_m'], axis = 1)
    df = df[df['month_m'].isin([10, 11, 12, 1, 2,])]
    df = df[df.columns].apply(np.mean, axis = 0)
    df = pd.DataFrame(df).T
    df['team_m'] = [team]*df.shape[0]
    df = df.drop('month_m', axis = 1)
    return df

def allteam_months():
    t_keys = ['MIL', 'GSW', 'MIN', 'MIA', 'ATL', 'BOS', 'DET',
              'NYK', 'DEN', 'DAL', 'POR', 'ORL', 'TOR', 'CLE',
              'SAS', 'CHA', 'UTA', 'CHI', 'HOU', 'PHO', 'WAS',
              'LAL', 'PHI', 'NOH', 'MEM', 'LAC', 'SAC', 'OKC',
              'BRK', 'IND']
    df_dict = {}
    for team in t_keys:
        df_dict[team] = team_months(team = team)

    df = pd.concat(df_dict.values(), axis = 0)
    df = df.reset_index().drop('index', axis = 1)
    return df

class dbConnect():
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

def month_sql():
    db = "../sql/nba_stats.db"
    temp = dbConnect(db)
    tables = get_tables(db)
    with temp:
        tablename = ['nba', 'opponent', 'month', '2015']
        tablename = ('_').join(tablename)
        if tablename in list(tables):
            print "Table already exists"
        else:
            df = allteam_months()
            df.to_sql(tablename, temp.con, flavor = 'sqlite')
            print
            print 'From stats.nba.com'
            print 'Finished team month stats table for %s' % ('2015')
            print tablename
            print

if __name__ == '__main__':
    print 'Building Dataframe...'
    month_sql()
    print 'Done'
