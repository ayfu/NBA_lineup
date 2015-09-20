'''

__file__

    get_dataSQL.py

__description__

    This file uses bballref.py, nba_lineups.py, and nba_id.py to gather data
    and send to database

'''

import sys, os, glob
import pandas as pd
import numpy as np
import sqlite3
import re

#sys.path.append(os.path.abspath("./"))
from nba_lineups import *
from bballref import *

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

def all_db():
    '''
    Prints all available databases
    '''
    alldb = glob.glob(os.path.join('..', 'sql', '*.db'))
    for db in alldb:
        print 'Folder:', db.split('\\')[:-1]
        print 'File:', db.split('\\')[-1]
        print

def all_tables(db):
    '''
    Prints all available tables in db
    '''
    #len(re.findall("[\w]+.db", db)) > 0
    # Build a list of all *.db files
    alldb = glob.glob(os.path.join('..', 'sql', '*.db'))
    temp_db = alldb[:]
    for x in range(len(temp_db)):
        alldb[x] = alldb[x].split('\\')[-1]

    if db in alldb:
        sql_db = "sql/" + db
        temp = dbConnect(sql_db)
        with temp:
            temp.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            print temp.cur.fetchall()

def main():

    season = '2014-15'
    playoff = 'Regular+Season'

    alldb = glob.glob(os.path.join('..', 'sql', '*.db'))
    temp_db = alldb[:]
    for x in range(len(temp_db)):
        alldb[x] = alldb[x].split('\\')[-1]

    if 'nba_stats.db' not in alldb:
        print 'Creating new database: nba_stats.db'
    temp = dbConnect("../sql/nba_stats.db")
    with temp:
        # methods from nba_lineups and bballref
        for team in teams:
            df = mergedf(team = teams[team], season = season,
                         playoff = playoff)
            tablename = ['nba', team, season]
            tablename = ('_').join(tablename)
            df.to_sql(tablename, temp.con, flavor = 'sqlite')
            print 'Finished building dataframe for', team
            print

if __name__ == "__main__":
    main()
