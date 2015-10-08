
import sys, os
from collections import defaultdict
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split


class PruneLabelEncoder(LabelEncoder):
    def __init___(self):
        super(PruneLabelEncoder, self).__init__()
    def fit(self, series, cutoff=10):
        self.cutoff = cutoff
        # Generate the transformation classes and also the map for low output munging
        super(PruneLabelEncoder, self).fit(series)
        trans_series = super(PruneLabelEncoder, self).transform(series)
        self.val_count_map = defaultdict(int)
        for i in trans_series:
            self.val_count_map[i] += 1
        # identify the first key with low frequency and use it for all low freq vals
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                self.low_cnt_target = key
                break
    def transform(self, series):
        trans_series = super(PruneLabelEncoder, self).transform(series)
        # Transform all the low frequency keys into the low frequency target key
        for key, val in self.val_count_map.items():
            if val < self.cutoff:
                trans_series[trans_series==key] = self.low_cnt_target
        return trans_series

def encode(df, columns, TRANSFORM_CUTOFF):
    '''
    Takes in a dataframe, columns of interest, and a cutoff value for bucketing encoding values

    If the frequency of an encoded value is below the cutoff, it will bucket
    everything to the first value it encounters that is below the cutoff value
    '''
    temp = df.copy()

    # Checking if there are 2 or more unique values in each column
    for x in columns:
        if len(df[x].unique()) < 2:
            return 'Error: Fewer than 2 unique values in a column'

    for col in columns:
        if type(df[col].unique()[1]) == str:
            le = PruneLabelEncoder()
            le.fit(df[col],TRANSFORM_CUTOFF)
            df[col] = le.transform(df[col])

    return df

def encode_force(df, columns, TRANSFORM_CUTOFF):
    '''
    takes in a dataframe, list of columns, and TRANSFORM_CUTOFF

    same as encode but it doesn't do the str check
    '''
    temp = df.copy()

    # Checking if there are 2 or more unique values in each column
    for x in columns:
        if len(df[x].unique()) < 2:
            return 'Error: Fewer than 2 unique values in a column'

    for col in columns:
        le = PruneLabelEncoder()
        le.fit(df[col],TRANSFORM_CUTOFF)
        df[col] = le.transform(df[col])

    return df


def forest_encode(filename = '../csv_data/nba_15season_kMeans_150929.csv',
                  min_cutoff = 1, TRANSFORM_CUTOFF = 1, testsize = 0.3):
    """
    Encode dataframe and subset out useless rows from imperfect merge.

    Takes in: filename,
              min_cutoff = minimum minutes played,
              TRANSFORM_CUTOFF = min cutoff for binning of labelencoder
              testsize = fraction of dataframe used for size of test dataframe
    Returns: train, test, id_df for decision tree regression.
             id_df allows me to match labelencoded id with actual lineup
    """
    nba_df = subset_frame(filename = filename, min_cutoff = min_cutoff,
                  TRANSFORM_CUTOFF = TRANSFORM_CUTOFF)
    ###########################################################################
    # Encode categorical variables
    ###########################################################################
    #Label encode categorical variables lineup
    id_df = pd.DataFrame(nba_df['lineup'])
    lecolumns = ['lineup']
    nba_df = encode_force(nba_df, lecolumns, TRANSFORM_CUTOFF)
    id_df['id'] = nba_df['lineup']

    #combine team and opp first and then label encode
    team_vals = nba_df.team.values
    opp_vals = nba_df.opponent.values
    allteam_vals = np.concatenate((team_vals, opp_vals))
    ple_team = PruneLabelEncoder()
    ple_team.fit(allteam_vals, cutoff= TRANSFORM_CUTOFF)
    nba_df['team'] = ple_team.transform(nba_df.team.values)
    nba_df['opponent'] = ple_team.transform(nba_df.opponent.values)

    # After checking bball_ref, these null values should be set to 0
    """
    # Commented out for dropping columns in subset_frame()
    nullcols = ['fg_percent', 'TP_percent', 'eFG', 'FT_percent']
    for nullcol in nullcols:
        nba_df.loc[pd.isnull(nba_df[nullcol]), nullcol] = 0

    print 'Current shape of dataframe:', nba_df.shape
    """

    ###########################################################################
    # SPLIT DATAFRAME BY MONTH (PREDICT ON MONTHS 3 and 4 (march and april)
    ###########################################################################
    np.random.seed(1)
    nba_df = nba_df.reindex(np.random.permutation(nba_df.index))
    month_id = nba_df['month'].isin([3,4])

    train = nba_df[np.logical_not(month_id)]
    test = nba_df[month_id]

    train = train.reset_index().drop('index', axis = 1)
    test = test.reset_index().drop('index', axis = 1)
    """
    ###########################################################################
    # SPLIT DATAFRAME ON UNIQUE LINEUPS TO PREVENT DATA LEAKAGE
    ###########################################################################
    uniq_id = nba_df['lineup'].unique()
    np.random.seed(0) # make sure this split is repeatable
    uniq_id = uniq_id[np.random.permutation(len(uniq_id))]

    train_id, test_id = train_test_split(uniq_id, test_size = testsize,
                                         random_state = 1)

    trainID_boolmask = nba_df['lineup'].isin(train_id)
    #testID_boolmask = nba_df['lineup'].isin(test_id)
    testID_boolmask = np.logical_not(trainID_boolmask)

    train = nba_df[trainID_boolmask]
    test = nba_df[testID_boolmask]

    train = train.reset_index().drop('index', axis = 1)
    test = test.reset_index().drop('index', axis = 1)
    """
    ###########################################################################
    ###########################################################################
    print
    print 'Train dataframe shape:', train.shape
    print 'Test dataframe shape:', test.shape
    print 'Percentage of df for test', test.shape[0]/float(nba_df.shape[0])
    print '# of overlapping lineups:', sum(train['lineup'].isin(test['lineup']))
    print 'id_df to match labelencoding with lineup in string form:', id_df.shape
    #print (nba_df['lineup'].isin(test_id) == np.logical_not(trainID_boolmask)).all()

    #X_train, X_test, y_train, y_test = train_test_split(nba_df, nba_df['points'], test_size = 0.4, random_state = 1)
    #X_train, X_test = train_test_split(nba_df, test_size = 0.4, random_state = 1)
    #print '# of overlapping lineups:', sum(X_train['lineup'].isin(X_test['lineup']))

    #print X_train.shape, X_test.shape, y_train.shape, y_test.shape
    #print X_train.shape, X_test.shape

    return train, test, id_df


def lin_encode(filename = '../csv_data/nba_15season_kMeans_150929.csv',
               min_cutoff = 1, TRANSFORM_CUTOFF = 1, testsize = 0.3):
    """
    Encode dataframe and subset out useless rows from imperfect merge.
    Specific for lin_reg

    Takes in: filename,
              min_cutoff = minimum minutes played,
              TRANSFORM_CUTOFF = min cutoff for binning of labelencoder
              testsize = fraction of dataframe used for size of test dataframe
    Returns: train, test, id_df for linear regression.
             id_df allows me to match labelencoded id with actual lineup
    """
    nba_df = subset_frame(filename = filename, min_cutoff = min_cutoff,
              TRANSFORM_CUTOFF = TRANSFORM_CUTOFF)
    ###########################################################################
    # Encode categorical variables
    ###########################################################################
    #Label encode categorical variables lineup
    id_df = pd.DataFrame(nba_df['lineup'])
    lecolumns = ['lineup']
    nba_df = encode_force(nba_df, lecolumns, TRANSFORM_CUTOFF)
    id_df['id'] = nba_df['lineup']

    #combine team and opp first and then label encode
    team_vals = nba_df.team.values
    opp_vals = nba_df.opponent.values
    allteam_vals = np.concatenate((team_vals, opp_vals))
    ple_team = PruneLabelEncoder()
    ple_team.fit(allteam_vals, cutoff= TRANSFORM_CUTOFF)
    nba_df['team'] = ple_team.transform(nba_df.team.values)
    nba_df['opponent'] = ple_team.transform(nba_df.opponent.values)

    # After checking bball_ref, these null values should be set to 0
    """
    nullcols = ['fg_percent', 'TP_percent', 'eFG', 'FT_percent']
    for nullcol in nullcols:
        nba_df.loc[pd.isnull(nba_df[nullcol]), nullcol] = 0

    print 'Current shape of dataframe:', nba_df.shape
    """
    ##########################ONE HOT ENCODING#################################

    points = nba_df['points'].copy()
    nba_df = nba_df.drop('points', axis = 1)

    onehotcol = ['team', 'opponent']
    for col in onehotcol:
        onehottemp = nba_df[col].values
        lbl = OneHotEncoder()
        lbl.fit(np.resize(np.array(onehottemp), (len(onehottemp), 1)))

        onehottemp = lbl.transform(np.resize(np.array(onehottemp),
                                             (len(onehottemp),1))).toarray()
        for i in range(onehottemp.shape[1]):
            nba_df[col + '_' + str(i)] = onehottemp[:,i]
        nba_df = nba_df.drop([col], axis = 1)

    nba_df['points'] = points
    ###########################################################################
    # SPLIT DATAFRAME BY MONTH (PREDICT ON MONTHS 3 and 4 (march and april)
    ###########################################################################
    np.random.seed(1)
    nba_df = nba_df.reindex(np.random.permutation(nba_df.index))
    month_id = nba_df['month'].isin([3,4])

    train = nba_df[np.logical_not(month_id)]
    test = nba_df[month_id]

    train = train.reset_index().drop('index', axis = 1)
    test = test.reset_index().drop('index', axis = 1)
    ###########################################################################
    ###########################################################################
    print
    print 'Train dataframe shape:', train.shape
    print 'Test dataframe shape:', test.shape
    print 'Percentage of df for test', test.shape[0]/float(nba_df.shape[0])
    print '# of overlapping lineups:', sum(train['lineup'].isin(test['lineup']))
    print 'id_df to match labelencoding with lineup in string form:', id_df.shape

    return train, test, id_df

def subset_frame(filename = '../csv_data/nba_15season_kMeans_150929.csv',
                 min_cutoff = 1, TRANSFORM_CUTOFF = 1):
    """
    Encode dataframe and subset out useless rows from imperfect merge.

    Takes in: filename,
              min_cutoff = minimum minutes played,
              TRANSFORM_CUTOFF = min cutoff for binning of labelencoder
              testsize = fraction of dataframe used for size of test dataframe
    Returns: train, test, id_df for decision tree regression.
             id_df allows me to match labelencoded id with actual lineup
    """
    #filename = os.path.join('..', 'csv_data', filename)
    nba_df = pd.read_csv(filename, header = 0)
    print 'Full dataframe:', nba_df.shape

    # Filter out useless rows from imperfect merge
    nba_df = nba_df[pd.notnull(nba_df['MIN'])]
    #nba_df = nba_df[pd.notnull(nba_df['points'])]
    #nba_df = nba_df[np.logical_not(nba_df['points'].isin([np.inf, -np.inf]))]
    print 'Subset out NULL/non-matched rows from merging' +\
          ' nba with bref:', nba_df.shape
    nba_df = nba_df[nba_df['minutes_pm'] > min_cutoff]
    print 'Subset out lineups with fewer minutes' +\
          ' than min_cutoff', nba_df.shape
    # Drop useless columns
    nba_df = nba_df.drop(['result', 'TEAM_ABBREVIATION', 'date',
                          #'minutes', 'num_poss', 'opp_poss', 'pace_bref',
                          #'fg_pm', 'fga_pm', 'fg_percent', 'TP_pm',
                          #'TPA_pm', 'TP_percent', 'eFG', 'FT_pm',
                          #'FTA_pm', 'FT_percent',
                          'minutes_pm'],
                          axis = 1)

    # Using regex to find all bref type columns
    cols = ['minutes', 'num_poss', 'opp_poss', 'pace_bref', 'fg_pm',
        'fga_pm', 'fg_percent', 'TP_pm', 'TPA_pm', 'TP_percent',
        'eFG', 'FT_pm', 'FTA_pm', 'FT_percent']
    cols = '|'.join(cols)
    cols = nba_df.columns[nba_df.columns.str.contains(cols)]
    # Filter out poor sample size for a train test split
    for col in cols:
        nba_df = nba_df[pd.notnull(nba_df[col])]
        nba_df = nba_df[np.logical_not(nba_df[col].isin([np.inf, -np.inf]))]

    # Define new feature avg_pm which gives the plus/minus on average before
    # This feature is also used in building kMeans as well since it's very
    # predictive
    nba_df['avg_pm'] = 2*(nba_df['fg_pm']-nba_df['TP_pm']) + nba_df['FT_pm'] +  3*nba_df['TP_pm']

    columns = list(nba_df.columns)
    columns.remove('points')
    columns.append('points')
    nba_df = nba_df[columns]

    print 'Current shape of dataframe:', nba_df.shape
    return nba_df

if __name__ == '__main__':
    train, test, id_df = forest_encode(filename = 'nba_15season_kMeans_150929')
    #train, test, id_df = lin_encode()
    print train.shape, test.shape, id_df.shape
