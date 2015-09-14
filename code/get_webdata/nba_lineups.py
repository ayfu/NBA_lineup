'''

__file__

    nba_lineups.py

__description__

    This  file provides a way to get data from stats.nba.com for the
    statistics on various lineups throughout the league. It uses nba_id.py
    to format the dataframe

'''

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import json

from nba_id import *

def lineups(stattype = 'Base', season = '2014-15', playoff = 'Regular+Season'):
    '''
    Takes in a NBA season designated by the year (ie. '2014-15'),
    Stat type ('Base','Advanced','Four+Factors','Misc','Scoring',
    'Opponent')
    playoff ('Regular+Season', 'Playoffs')

    Returns a dataframe of all NBA lineups and their summary stats
    '''

    types = ['Base', 'Advanced', 'Four+Factors', 'Misc', 'Scoring', 'Opponent']
    seasons = [#'1996-97', '1997-98', '1998-99', '1999-00',
               '2000-01', '2001-02', '2002-03', '2003-04', '2004-05',
               '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
               '2010-11', '2011-12', '2012-13', '2013-14', '2014-15']
    playoffs = ['Regular+Season', 'Playoffs']

    if type(season) != str:
        raise TypeError('Must input a string')
    if season not in seasons:
        raise ValueError("Error, please enter in a season between '2000-01' and '2014-15'")
    if type(stattype) != str:
        raise TypeError('Must input a string')
    if stattype not in types:
        raise ValueError("Error, please enter in a valid stattype"+\
                         "'Base', 'Advanced', 'Four+Factors',"+\
                         "'Misc', 'Scoring', 'Opponent'")
    if type(playoff) != str:
        raise TypeError('Must input a string')
    if playoff not in playoffs:
        raise ValueError("Error, please enter 'Regular+Season' or 'Playoffs'")

    url = ['http://stats.nba.com/stats/leaguedashlineups?DateFrom=',
           'DateTo=',
           'GameID=',
           'GameSegment=',
           'GroupQuantity=5',
           'LastNGames=0',
           'LeagueID=00',
           'Location=',
           'MeasureType='+stattype,
           'Month=0',
           'OpponentTeamID=0',
           'Outcome=',
           'PaceAdjust=N',
           'PerMode=PerGame',
           'Period=0',
           'PlusMinus=N',
           'Rank=N',
           'Season='+season,
           'SeasonSegment=',
           'SeasonType='+playoff,
           'VsConference=',
           'VsDivision=']
    url = '&'.join(url)

    r = requests.get(url)
    dat = json.loads(r.text)

    header = dat['resultSets'][0]['headers']
    data = dat['resultSets'][0]['rowSet']
    df = pd.DataFrame(data, columns = header)
    if stattype == 'Base':
        df = df.sort('PLUS_MINUS', ascending = False)
        df = df.reset_index().drop('index', axis = 1)
    df2 = pID(season) #create a player ID data frame to match ID with ab_name

    # Create new 'lineup' column with the names of players
    # Formatted the same as in bball-ref.com data from bballref.py
    df['lineup'] = [np.nan]*df.shape[0]
    for i in range(df.shape[0]):
        temp = map(int, df.loc[i, 'GROUP_ID'].split(' - '))
        reformat = [df2.loc[df2['id'] == x, 'ab_name'].values[0] for x in temp]
        df.loc[i, 'lineup'] = (' | ').join(reformat)
    """
    cols = ['lineup', 'GROUP_SET', 'GROUP_ID', 'GROUP_NAME', 'TEAM_ID',
            'TEAM_ABBREVIATION', 'GP', 'W', 'L', 'W_PCT', 'MIN', 'FGM',
            'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
            'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK',
            'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']
    """
    df = df.drop(['GROUP_SET', 'GROUP_ID', 'GROUP_NAME', 'TEAM_ID'],
                 axis = 1)

    # Move lineups column to the left of the table
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df

def mergedf(season = '2014-15', playoff = 'Regular+Season'):
    '''
    Takes in a NBA season designated by the year (ie. '2014-15'),
    playoff ('Regular+Season', 'Playoffs')

    Returns a dataframe of all NBA lineups for every stattype on
    stats.nba.com for lineups (merges all 6 dataframes together)
    '''

    types = ['Base', 'Advanced', 'Four+Factors', 'Misc', 'Scoring', 'Opponent']

    df_dict = {}
    for stattype in types:
        df_dict[stattype] = lineups(stattype = stattype, season = season, playoff = playoff)
    # Start with first dataframe (doesn't matter which one)
    # merge on it with the rest of the dataframes in the dictionary
    df = df_dict[df_dict.keys()[0]]
    for x in df_dict.keys()[1:]:
        df_temp = df_dict[x]
        # Drop repetitive columns
        df_temp = df_temp.drop(['TEAM_ABBREVIATION', 'GP', 'W', 'L',
                                'W_PCT', 'MIN'], axis = 1)
        df = pd.merge(df, df_temp, how = 'left', on = 'lineup')

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
