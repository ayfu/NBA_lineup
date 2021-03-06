'''

__file__

    bballref.py

__description__

    This  file provides a way to scrape basketball-reference.com for the
    statistics on various lineups throughout the league. The user can specify
    what kind of lineup type, which team, year, playoffs (Y/N/Both = ''),
    opposing team, and month.

'''

import sys
import unicodedata
import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np



def get_url(lineup = '5-man', output = 'total', year = '2015',
            playoffs = 'N', team = 'GSW', opp = '', month = ''):
    '''
    takes in lineup, output, year, playoffs, team, opposing team, and month

    returns the URL of basketball-reference.com with a user defined lineup.

    can specify lineup, output, year, playoffs, team, opponent, month
    '''
    u = 'http://www.basketball-reference.com/play-index/plus/lineup_finder.cgi'
    error = 'Incorrect input for'
    all_teams = ['PHI','MIL','CHI','CLE','BOS','LAC','MEM','ATL','MIA','CHA',
                 'UTA','SAC','NYK','LAL','ORL','DAL','NJN','DEN','IND','NOH',
                 'DET','TOR','HOU','SAS','PHO','OKC','MIN','POR','GSW','WAS',
                '']
    all_years = ['2015','2014','2013','2012','2011','2010','2009',
                 '2008','2007','2006','2005','2004','2003','2002','2001']

    if lineup not in ['5-man','4-man','3-man','2-man']:
        return error + 'lineup'
    if output not in ['total','per_min','per_poss']:
        return error + 'output'
    if year not in all_years:
        return error + 'year'
    if playoffs not in ['N','Y','']:
        return error + 'playoffs'
    if team not in all_teams:
        return error + 'team'
    if opp not in all_teams:
        return error + 'opp'
    if month not in ['10','11','12','1','2','3','4','5','6','']:
        return error + 'month'

    params = ['?request=1',
              'player_id=',
              'match=game',
              'lineup_type='+lineup,
              'output='+output,
              'year_id='+year,
              'is_playoffs='+playoffs,
              'team_id='+team,
              'opp_id='+opp,
              'game_num_min=0',
              'game_num_max=99',
              'game_month='+month,
              'game_location=',
              'game_result=',
              'c1stat=',
              'c1comp=ge',
              'c1val=',
              'c2stat=',
              'c2comp=ge',
              'c2val=',
              'c3stat=',
              'c3comp=ge',
              'c3val=',
              'c4stat=',
              'c4comp=ge',
              'c4val=',
              'order_by=diff_pts']

    return u+'&'.join(params)


def get_pagedata(url):
    '''
    takes in a bball-ref url for lineup stats

    returns a dataframe from the data in the basketball-reference.com URL

    This should be specific to the lineup_finder.
    '''
    df = pd.DataFrame({'id': [],
                   'lineup':[],
                   'date':[],
                   'team':[],
                   'away_game':[],
                   'opponent':[],
                   'result':[],
                   'minutes':[],
                   'num_poss':[],
                   'opp_poss':[],
                   'pace':[],
                   'fg':[],
                   'fga':[],
                   'fg_percent':[],
                   'TP':[],
                   'TPA':[],
                   'TP_percent':[],
                   'eFG':[],
                   'FT':[],
                   'FTA':[],
                   'FT_percent':[],
                   'points':[]})
    cols = ['id','lineup','date','team','away_game','opponent', 'result',
            'minutes','num_poss', 'opp_poss','pace','fg','fga', 'fg_percent',
            'TP', 'TPA','TP_percent','eFG','FT','FTA', 'FT_percent','points']
    df = df[cols]

    html = requests.get(url)
    soup = BeautifulSoup(html.content, "lxml")
    # Find the 'tbody' section, the body of the main table. Only one per page
    dat = soup.find('tbody')
    # Create a list of all the elements in the Rank column (labeled 1,2,3,...)
    # Skips over the table headers of Rank, Lineup, Date, ...
    try:
        dat = dat.findAll('td', {'align':{"right"}}, csk = re.compile('[0-9]+'))
        for i in range(len(dat)):
            # Convert unicode to ascii format to concatenate it into a dataframe
            temp = unicodedata.normalize(
                    'NFKD', dat[i].parent.get_text()).encode('ascii','ignore')
            # Create a list of each element in a row
            # before it was just one long string
            temp = temp.split('\n')[1:-1]
            df_temp = pd.DataFrame(np.array([temp]),columns = cols)
            df = pd.concat([df, df_temp],axis = 0)
        return df
    except AttributeError:
        return None

def get_nextlink(url):
    '''
    takes in a bball-ref url for lineup stats

    returns the URL of the link for 'Next page' on the webpage.
    If there's no 'Next page' link, then it returns None.
    '''
    html = requests.get(url)
    soup = BeautifulSoup(html.content, "lxml")
    urlbase = 'http://www.basketball-reference.com'
    # Find all the links that are either previous page or next page
    ind = soup.findAll(
            'a', href = re.compile('^(/play-index/plus/lineup_finder.cgi)'+\
                                   '[A-Za-z0-9\.&_;+:?]+'))
    newurl = None
    for x in ind:
        if x.get_text() == 'Next page':
            newurl = urlbase + x.attrs['href']
        else:
            continue
    return newurl

def get_alldata(url):
    '''
    takes in a bball-ref url for lineup stats

    returns a dataframe of all the data associated with the URL query for
    bball-reference lineup.

    data is spread across multiple pages
    '''
    # Store each dataframe associated with each page as elements in a dictionary
    # Get a dictionary of dataframes
    # Concatenate all the dataframes together as the last step
    df_dict = {}
    i = 1
    df_dict['page'+str(i)] = get_pagedata(url)
    while get_nextlink(url) != None:
        i += 1
        name = 'page' + str(i)
        url = get_nextlink(url)
        df = get_pagedata(url)
        if df is None:
            break
        else:
            df_dict[name] = df
    df = pd.concat(df_dict.values(), axis = 0)
    return df

def convert_numbers(df):
    '''
    takes in a dataframe (from get_alldata(url))

    returns a dataframe with the columns in numcols converted to floats
    '''
    numcols = ['minutes','num_poss','opp_poss','pace','fg','fga','fg_percent',
               'TP','TPA','TP_percent','eFG','FT','FTA','FT_percent',
               'points']

    # convert id column to int and then sort dataframe based on id
    df['id']=df['id'].astype(int)
    df = df.sort_values('id')
    df = df.reset_index().drop('index',axis = 1)
    df['date'] = pd.to_datetime(df.date,format= '%Y-%m-%d')

    #away_game, Away = 1, Home = 0
    df.loc[df['away_game'] == '','away_game'] = np.nan
    df.loc[pd.notnull(df['away_game']),'away_game'] = 1
    df.loc[pd.isnull(df['away_game']),'away_game'] = 0

    # convert columns with strings for numbers to floats
    # An empty element is converted to 99999 first and then converted to NaN
    # 99999 is chosen because it is an unlikely number.
    for col in numcols:
        if len(df.loc[df[col] == '',col]) > 0:
            if 99999 not in df[col].unique():
                df.loc[df[col] == '',col] = 99999
                df[col] = df[col].astype(float)
                df.loc[df[col] == 99999, col] = np.nan
            else:
                print "Error found 99999 in column", x
        elif len(df.loc[df[col] == '',col]) == 0:
            df[col] = df[col].astype(float)
        else:
            print "Loop not supposed to get here for", x
    return df

"""
if __name__ == "__main__":
    team = sys.argv[1]
    if len(team)>0:
        df = get_alldata(get_url(team = team))
        df = convert_numbers(df)
    else:
        df = get_alldata(get_url())
        df = convert_numbers(df)

"""
