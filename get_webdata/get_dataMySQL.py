'''

__file__

    get_dataMySQL.py

__description__

    This file uses bballref.py, nba_lineups.py, nba_team_data,
    nba_team_data_month, and nba_id.py to scrape data and send to a MySQL database

'''

import sys
import os
import datetime as dt

import pandas as pd
import numpy as np
import MySQLdb

from bballref import *
from nba_lineups import *
from nba_team_data import *
from nba_team_data_month import *
from nba_id import *

# variables for stats.nba.com
season = '2014-15'
playoff = 'Regular+Season'

# variables for basketball-reference.com
all_teams = ['PHI','MIL','CHI','CLE','BOS','LAC','MEM','ATL','MIA','CHA',
             'UTA','SAC','NYK','LAL','ORL','DAL','NJN','DEN','IND','NOH',
             'DET','TOR','HOU','SAS','PHO','OKC','MIN','POR','GSW','WAS']
year = '2015'

class dbConnect():
    '''
    Class to help with context management for 'with' statements.

    http://www.webmasterwords.com/python-with-statement

    Example Below:
    c = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'nba_stats')
    with c:
        df.to_sql('nbadotcom', c.con, flavor = 'mysql', dtype = dtype)
    '''
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db
    def __enter__(self):
        self.con = MySQLdb.connect(host = self.host, user = self.user,
                                   passwd = self.passwd, db = self.db)
        self.cur = self.con.cursor()
    def __exit__(self, type, value, traceback):
        self.cur.close()
        self.con.close()

def all_tables(db = 'nba_stats'):
    """
    Takes in: database

    Returns: List of tables in database
    """
    c = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = db)
    with c:
        c.cur.execute('SHOW TABLES;')
        tables = c.cur.fetchall()
        tables = [str(tables[x][0]) for x in range(len(tables))]
        print tables


def main():
    """
    Scrape basketball-reference.com and nba.com for stats and populate MySQL
    """
    global season, playoff, all_teams, year

    c = dbConnect(host = 'localhost', user = 'root',
                  passwd = 'default', db = 'nba_stats')

    # Scrape basketball-reference.com and populate nba_stats database
    with c:
        for team in all_teams:
            url = get_url(team = team, year = year)
            df = get_alldata(url)
            df = convert_numbers(df)
            # Make sure it can fit dtypes
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['lineup', 'team', 'opponent',
                                     'date', 'result']:
                    dtype[df.columns[i]] = 'TEXT'
                elif df.columns[i] in ['away_game']:
                    dtype[df.columns[i]] = 'INTEGER'
                else:
                    dtype[df.columns[i]] = 'REAL'
            tablename = ['bballref', team, year]
            tablename = ('_').join(tablename)
            try:
                df.to_sql(name = tablename, con = c.con,
                          flavor = 'mysql', dtype = dtype)
                print
                print 'From basketball-reference.com:'
                print 'Finished buliding dataframe for %s, %s' % (team, year)
                print tablename
                print
            except ValueError:
                raise ValueError('Table ' + tablename + ' already exists.')

    # Scrape nba.com (lineup info) and populate nba_stats database
    with c:
        for team in teams:
            df = mergedf(team = teams[team], season = season,
                         playoff = playoff)
            tablename = ['nba', team, year]
            tablename = ('_').join(tablename)
            # Make sure it can fit dtypes
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['lineup', 'TEAM_ABBREVIATION']:
                    dtype[df.columns[i]] = 'TEXT'
                elif df.columns[i] in ['GP', 'W', 'L']:
                    dtype[df.columns[i]] = 'INTEGER'
                else:
                    dtype[df.columns[i]] = 'REAL'
            tablename = ['nba', team, year]
            tablename = ('_').join(tablename)
            try:
                df.to_sql(name = tablename, con = c.con,
                          flavor = 'mysql', dtype = dtype)
                print
                print 'From stats.nba.com:'
                print 'Finished building dataframe for %s, %s' % (team, year)
                print tablename
                print
            except ValueError:
                raise ValueError('Table ' + tablename + ' already exists.')

    # Scrape nba.com (team info by month) and populate nba_stats database
    with c:
        tablename = ['nba', 'opponent', 'month', '2015']
        tablename = ('_').join(tablename)
        df = allteam_months()
        # Make sure it can fit dtypes
        dtype = {}
        for i in range(len(df.columns)):
            if df.columns[i] in ['team_m']:
                dtype[df.columns[i]] = 'TEXT'
            else:
                dtype[df.columns[i]] = 'REAL'
        df.to_sql(name = tablename, con = c.con,
                  flavor = 'mysql', dtype = dtype)
        print
        print 'From stats.nba.com'
        print 'Finished team month stats table for %s' % ('2015')
        print tablename
        print

    # Scrape nba.com (team info extra stats) and populate nba_stats database
    with c:
        for stats in ['somestats', 'allstats']:
            tablename = ['nba', 'opponent', stats, '20' + season[-2:]]
            tablename = ('_').join(tablename)

            if stats == 'somestats':
                df = team_merge(season = '2014-15',
                                playoff = 'Regular+Season',
                                somestats = True)
            else:
                df = team_merge(season = '2014-15',
                                playoff = 'Regular+Season',
                                somestats = False)
                    # Make sure it can fit dtypes
            dtype = {}
            for i in range(len(df.columns)):
                if df.columns[i] in ['opponent']:
                    dtype[df.columns[i]] = 'TEXT'
                elif df.columns[i] in ['W_opt', 'L_opt']:
                    dtype[df.columns[i]] = 'INTEGER'
                else:
                    dtype[df.columns[i]] = 'REAL'
            df.to_sql(name = tablename, con = c.con,
                      flavor = 'mysql', dtype = dtype)
            print
            print 'From stats.nba.com'
            print 'Finished team stats table for %s' % ('20' + season[-2:])
            print tablename
            print

if __name__ == "__main__":
    main()
