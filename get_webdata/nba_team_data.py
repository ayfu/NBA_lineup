'''

__file__

    nba_team_data.py

__description__

    This file uses gathers team data from stats.nba.com and sends it to
    sql/nba_stats.db

'''

import sys, os
from collections import defaultdict
import re
import json
import sqlite3
import datetime as dt

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.abspath("../sql/"))
from get_tables import get_tables

season = '2014-15'

def team_stats(stattype = 'Base', season = '2014-15',
               playoff = 'Regular+Season'):

    all_teams = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL',
             'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL',
             'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKC',
             'ORL', 'PHI', 'PHO', 'POR', 'SAC', 'SAS', 'TOR',
             'UTA', 'WAS']

    team_name = [u'Atlanta Hawks', u'Boston Celtics', u'Brooklyn Nets',
           u'Charlotte Hornets', u'Chicago Bulls', u'Cleveland Cavaliers',
           u'Dallas Mavericks', u'Denver Nuggets', u'Detroit Pistons',
           u'Golden State Warriors', u'Houston Rockets', u'Indiana Pacers',
           u'Los Angeles Clippers', u'Los Angeles Lakers',
           u'Memphis Grizzlies', u'Miami Heat', u'Milwaukee Bucks',
           u'Minnesota Timberwolves', u'New Orleans Pelicans',
           u'New York Knicks', u'Oklahoma City Thunder', u'Orlando Magic',
           u'Philadelphia 76ers', u'Phoenix Suns', u'Portland Trail Blazers',
           u'Sacramento Kings', u'San Antonio Spurs', u'Toronto Raptors',
           u'Utah Jazz', u'Washington Wizards']

    team_id = {team_name[x]: all_teams[x] for x in range(len(team_name))}

    types = ['Base', 'Advanced', 'Four+Factors', 'Misc', 'Scoring', 'Opponent']
    seasons = [#'1996-97', '1997-98', '1998-99', '1999-00',
               '2000-01', '2001-02', '2002-03', '2003-04', '2004-05',
               '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
               '2010-11', '2011-12', '2012-13', '2013-14', '2014-15']
    playoffs = ['Regular+Season', 'Playoffs']

    url = ['http://stats.nba.com/stats/leaguedashteamstats?',
           'Conference=',
           'DateFrom=',
           'DateTo=',
           'Division=',
           'GameScope=',
           'GameSegment=',
           'LastNGames=0',
           'LeagueID=00',
           'Location=',
           'MeasureType=' + stattype, ##
           'Month=0',
           'OpponentTeamID=0',
           'Outcome=',
           'PORound=0',
           'PaceAdjust=N',
           'PerMode=Per48', #per48 minutes
           'Period=0',
           'PlayerExperience=',
           'PlayerPosition=',
           'PlusMinus=N',
           'Rank=N&Season=' + season,
           'SeasonSegment=',
           'SeasonType=' + playoff,
           'ShotClockRange=',
           'StarterBench=',
           'TeamID=0',
           'VsConference=',
           'VsDivision=']

    url = ('&').join(url)

    r = requests.get(url)
    dat = json.loads(r.text)

    header = dat['resultSets'][0]['headers']
    data = dat['resultSets'][0]['rowSet']
    header = [x + '_opt' for x in header]
    df = pd.DataFrame(data, columns = header)
    df = df.rename(columns = {'TEAM_NAME_opt':'opponent'})
    # Convert team name to the abbreviation
    df['opponent'] = [team_id[x] for x in df['opponent']]
    return df


def team_merge(season = '2014-15', playoff = 'Regular+Season',
               somestats = True):
    '''
    Takes in a NBA season designated by the year (ie. '2014-15'),
    playoff ('Regular+Season', 'Playoffs')

    Returns a dataframe of all NBA lineups for every stattype on
    stats.nba.com for lineups (merges all 6 dataframes together)
    '''

    if somestats:
        types = ['Base', 'Advanced']
    else:
        types = ['Base', 'Advanced', 'Four+Factors', 'Misc',
                 'Scoring', 'Opponent']

    df_dict = {}
    for stattype in types:
        df_dict[stattype] = team_stats(stattype = stattype, season = season,
                                       playoff = playoff)

    df = df_dict[df_dict.keys()[0]]
    df = df.drop(['TEAM_ID_opt', 'GP_opt', 'MIN_opt',
                  'CFID_opt', 'CFPARAMS_opt'], axis = 1)
    for x in df_dict.keys()[1:]:
        df_temp = df_dict[x]
        # Drop repetitive columns
        df_temp = df_temp.drop(['TEAM_ID_opt', 'GP_opt', 'MIN_opt',
                                'CFID_opt', 'CFPARAMS_opt', 'W_opt',
                                'L_opt', 'W_PCT_opt'], axis = 1)
        df = pd.merge(df, df_temp, how = 'left', on = 'opponent')

    # Remove '_x' from columns with it
    # This is from merging dataframes with the same columns
    # NBA.com has overlapping columns
    tempcol = df.columns[df.columns.str.contains('(_x)$')].tolist()
    col_dict = {}
    for x in range(len(tempcol)):
        col_dict[tempcol[x]] = tempcol[x][:-2]
    df = df.rename(columns = col_dict)

    # Subset out all columns with '_y' because they are duplicates
    # From merging process. We will keep '_x' (they are the same)
    newcol = np.logical_not(df.columns.str.contains('(_y)$'))
    newcol = df.columns[newcol]
    df = df[newcol]

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

def send_to_sql():
    global season
    db = "../sql/nba_stats.db"
    temp = dbConnect(db)
    tables = get_tables(db)
    with temp:
        for stats in ['somestats', 'allstats']:
            tablename = ['nba', 'opponent', stats, '20' + season[-2:]]
            tablename = ('_').join(tablename)
            if tablename in list(tables):
                print "Table already exists"
                continue
            else:
                if stats == 'somestats':
                    df = team_merge(season = '2014-15',
                                    playoff = 'Regular+Season',
                                    somestats = True)
                else:
                    df = team_merge(season = '2014-15',
                                    playoff = 'Regular+Season',
                                    somestats = False)
                df.to_sql(tablename, temp.con, flavor = 'sqlite')
                print
                print 'From stats.nba.com'
                print 'Finished team stats table for %s' % ('20' + season[-2:])
                print tablename
                print

if __name__ == '__main__':
    send_to_sql()
