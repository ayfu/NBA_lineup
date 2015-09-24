'''

__file__

    get_tables.py

__description__

    This file takes a sqlite database and returns the name of tables inside the
    database

'''

import sys, os, glob
import pandas as pd
import numpy as np
import sqlite3
import re

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


def all_tables(db):
    '''
    Takes in: database file (db)
    Returns: list of tables in database
    '''
    #len(re.findall("[\w]+.db", db)) > 0
    # Build a list of all *.db files
    alldb = glob.glob(os.path.join('*.db'))
    temp_db = alldb[:]
    for x in range(len(temp_db)):
        alldb[x] = alldb[x].split('\\')[-1]

    if db in alldb:
        sql_db = db
        temp = dbConnect(sql_db)
        with temp:
            temp.cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
            table_list = temp.cur.fetchall()
            return table_list


def main():
    db = 'nba_stats.db'
    table_list = all_tables(db)
    print
    print 'List of files:'
    print table_list
    print
    print 'Number of files:'
    print len(table_list)

if __name__ == '__main__':
    main()