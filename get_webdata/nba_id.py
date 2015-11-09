'''

__file__

    nba_id.py

__description__

    provides a function that creates a dataframe of all current NBA players
    User can define

'''

import json
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np


def pID(season = '2014-15'):
    '''
    Takes in a NBA season designated by the year (ie. '2014-15')

    Returns a dataframe of a players NBA ID, name, and abbreviated name
    '''
    seasons = ['2000-01', '2001-02', '2002-03', '2003-04', '2004-05',
               '2005-06', '2006-07', '2007-08', '2008-09', '2009-10',
               '2010-11', '2011-12', '2012-13', '2013-14', '2014-15']
    if type(season) != str:
        raise TypeError('Must input a string')
    if season not in seasons:
        raise ValueError("Error, please enter in a season between" +\
                         " '2000-01' and '2014-15'")

    playerurl = 'http://stats.nba.com/league/player/#!/'
    # URL to grab all NBA players, not filtered.
    purl = 'http://stats.nba.com/stats/leaguedashplayerstats?College=' +\
           '&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=' +\
           '&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0'+\
            '&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0'+\
            '&Outcome=&PORound=0&PaceAdjust=N&PerMode=PerGame&Period=0'+\
            '&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N'+\
            '&Season='+ season +'&SeasonSegment=&SeasonType=Regular+Season&'+\
            'ShotClockRange=&StarterBench=&TeamID=0&VsConference=' +\
            '&VsDivision=&Weight='

    r = requests.get(purl)
    dataplayer = json.loads(r.text)

    # Create a data frame of player ID, player name,
    # and abbreviated name on basketball-reference.com
    playerID = {}
    for x in dataplayer['resultSets'][0]['rowSet']:
        playerID[x[0]] = x[1]

    # Create new dataframe
    df = pd.DataFrame({'id': playerID.keys(), 'name': playerID.values()})
    df['ab_name'] = np.zeros(df.shape[0])

    for i in range(df.shape[0]):
        # Nene is the only strange player with one name on NBA.com
        if df.loc[i, 'name'] == 'Nene':
            df.loc[i, 'ab_name'] = 'N. Hilario'
        else:
            # Take each name, split it to First and last name
            # Jeremy Lin -> J. Lin
            first = df.loc[i, 'name'].split(' ')[0][0]
            last = df.loc[i, 'name'].split(' ')[1]
            n = first + '. ' + last
            df.loc[i, 'ab_name'] = n

    df = df.sort_values('id')
    df = df.reset_index().drop('index', axis = 1)
    return df
