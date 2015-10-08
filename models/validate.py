'''

__file__

    validate.py

__description__

    This file gives example code that I used in building validation models in an
    iPython notebook. I frequently adjust the parameters and features used
    in the model. Test models on half the test set in a hold out method.

'''

import sys, os
from collections import defaultdict
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error

from nba_encoding import *
from kmeans_team import *
sys.path.append(os.path.abspath("../sql/"))
from get_tables import *

'''
Parameters for RandomForestRegressor
'''
params_rf = {'n_estimators': 200,
          'criterion': "mse",
          'max_features': "auto",
          'max_depth': None,
          'min_samples_split': 2,
          'min_samples_leaf': 2,
          'min_weight_fraction_leaf': 0,
          'oob_score': True,
          #'max_leaf_notes': None
          'verbose': 1
          }

params_gbr = {'loss': 'ls',
              'learning_rate': 0.02,
              'n_estimators': 100,
              'max_depth': 5,
              'min_samples_split': 5,
              'min_samples_leaf': 3,
              'subsample': 0.7
             }

params_lin = {'fit_intercept': True,
          'normalize': False,
          'copy_X': True,
          'n_jobs': 1
          }

params_lasso = {'alpha': 0.2,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}

params_ridge = {'alpha': 0.4,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}


class rfModel():
    '''
    Takes in: parameters for Random Forest

    Returns: predictions
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF
    def build_rfmodel(self):
        #x = 'nba_15season_all_150928.csv'
        self.train, self.test, self.id_df = forest_encode(filename = self.name,
                                                     min_cutoff = self.min_cutoff,
                                                     TRANSFORM_CUTOFF = self.t_cutoff)
        #Create validation set
        np.random.seed(2)
        self.test = self.test.reindex(np.random.permutation(self.test.index))
        self.test = self.test.iloc[:self.test.shape[0]/2,:]
        self.test = self.test.reset_index().drop('index', axis = 1)
        """
        # TESTING GROUNDS
        bestcol = ['PLUS_MINUS', 'NET_RATING', 'dayofmonth', 'day', 'MIN', 'GP',
           'lineup', 'month', 'PIE', 'team', 'PACE', 'away_game',
           'OPP_PTS_FB', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_OFF_TOV', 'OPP_AST',
           'PCT_PTS_2PT_MR', 'OPP_FG3M', 'FT_PCT', 'OPP_DREB', 'STL', 'OPP_FG3A',
           'OPP_FT_PCT', 'PTS_2ND_CHANCE', 'OPP_PTS_PAINT', 'FGA', 'OPP_FGA',
           'OREB', 'OPP_STL', 'DREB']

        """

        """
        bestcol = ['fg_pm', 'FT_pm', 'eFG', 'TP_pm', 'TP_percent', 'minutes',
                   'FTA_pm', 'PLUS_MINUS', 'dayofmonth', 'day', 'fg_percent',
                   'NET_RATING', 'fga_pm', 'TPA_pm', 'GP', 'FT_pm_3mean', 'PTS',
                   'away_game', 'TP_pm_3mean', 'opp_poss', 'PIE', 'TP_percent_3mean',
                   'num_poss', 'month', 'FTM', 'OPP_FGM', 'FT_percent',
                   'minutes_3mean', 'OPP_AST', 'PCT_PTS_2PT', 'OPP_PTS', 'lineup',
                   'PTS_OFF_TOV', 'PCT_PTS_2PT_MR', 'MIN', 'team', 'fg_pm_3mean',
                   'FT_percent_3mean', 'pace_bref', 'FGA_opt', 'OPP_FG3_PCT', 'OPP_FGA',
                   'FT_PCT', 'OPP_FG3A', 'OFF_RATING', 'FG3_PCT', 'PCT_PTS_PAINT', 'FTA_RATE',
                   'DEF_RATING', 'OPP_FG3M', 'points']

        bestcol = ['fga_pm', 'TP_pm', 'FT_pm',
                   #'TPA_pm', 'FTA_pm'
                  ]
        """
        bestcol = np.logical_not(self.train.columns.str.contains('_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]
        #bestcol = bestcol + ['points']
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]

        print 'After filtering: train shape , test shape:', self.train.shape, self.test.shape
        ###
        X = self.train.as_matrix(self.train.columns[:-1]).astype(float)
        y = self.train.as_matrix(['points'])[:, 0].astype(float)
        X_test = self.test.as_matrix(self.test.columns[:-1]).astype(float)
        self.y_test = self.test.as_matrix(['points'])[:, 0].astype(float)

        rf = RandomForestRegressor(**self.params)
        rf.fit(X, y)
        self.y_pred = rf.predict(X_test)

        print 'OOB score:', rf.oob_score_
        error = mean_squared_error(self.y_pred, self.y_test)
        print 'Mean squared error:', error

        # Getting attributes from RandomForestRegressor()
        feat = rf.feature_importances_
        self.feat_imp = pd.DataFrame({'feature': self.train.columns[:-1],
                                      'importance': feat})
        self.feat_imp = self.feat_imp.sort('importance', ascending = False)
        self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
        self.oob_prediction_ = rf.oob_prediction_
        self.estimators_ = rf.estimators_

    def features(self, color = 'purple'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.feat_imp.index[:15], self.feat_imp['importance'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.feat_imp.head(15))))
        ax.set_xticklabels(self.feat_imp['feature'].head(15), rotation=90, fontsize=15)
        ax.set_ylabel('Importance', fontsize = 15)

    def plot_result(self, color = 'green'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.scatter(self.y_test, self.y_pred, color = color,
                   label = 'Data', s = 100, alpha = 0.1)
        #ax.plot(x,pred_y, label = 'Fit', lw = 5)
        ax.set_xlabel('Actual +/- (points/48 min)',fontsize = 20)
        ax.set_ylabel('Predicted +/- (points/48 min)', fontsize = 20)
        ax.set_title('Results of Model', fontsize = 25)
        ax.set_xlim(-150,150)
        ax.set_ylim(-150,150)
        ax.legend(loc=2, fontsize = 20)
        ax.tick_params(labelsize =20)


class gbModel():
    '''
    Takes in: parameters for Random Forest

    Returns: predictions
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF
    def build_gbmodel(self):
        #x = 'nba_15season_all_150928.csv'
        """
        self.train, self.test, self.id_df = forest_encode(filename = self.name,
                                                     min_cutoff = self.min_cutoff,
                                                     TRANSFORM_CUTOFF = self.t_cutoff)
        """
        self.train, self.test, self.id_df = lin_encode(filename = self.name,
                                                     min_cutoff = self.min_cutoff,
                                                     TRANSFORM_CUTOFF = self.t_cutoff)
        #Create validation set
        np.random.seed(2)
        self.test = self.test.reindex(np.random.permutation(self.test.index))
        self.test = self.test.iloc[:self.test.shape[0]/2,:]
        self.test = self.test.reset_index().drop('index', axis = 1)

        """
        # TESTING GROUNDS
        bestcol = ['PLUS_MINUS', 'NET_RATING', 'dayofmonth', 'day', 'MIN', 'GP',
           'lineup', 'month', 'PIE', 'team', 'PACE', 'away_game',
           'OPP_PTS_FB', 'OPP_PTS_2ND_CHANCE', 'OPP_PTS_OFF_TOV', 'OPP_AST',
           'PCT_PTS_2PT_MR', 'OPP_FG3M', 'FT_PCT', 'OPP_DREB', 'STL', 'OPP_FG3A',
           'OPP_FT_PCT', 'PTS_2ND_CHANCE', 'OPP_PTS_PAINT', 'FGA', 'OPP_FGA',
           'OREB', 'OPP_STL', 'DREB']

        """

        """
        bestcol = ['fg_pm', 'FT_pm', 'eFG', 'TP_pm', 'TP_percent', 'minutes',
                   'FTA_pm', 'PLUS_MINUS', 'dayofmonth', 'day', 'fg_percent',
                   'NET_RATING', 'fga_pm', 'TPA_pm', 'GP', 'FT_pm_3mean', 'PTS',
                   'away_game', 'TP_pm_3mean', 'opp_poss', 'PIE', 'TP_percent_3mean',
                   'num_poss', 'month', 'FTM', 'OPP_FGM', 'FT_percent',
                   'minutes_3mean', 'OPP_AST', 'PCT_PTS_2PT', 'OPP_PTS', 'lineup',
                   'PTS_OFF_TOV', 'PCT_PTS_2PT_MR', 'MIN', 'team', 'fg_pm_3mean',
                   'FT_percent_3mean', 'pace_bref', 'FGA_opt', 'OPP_FG3_PCT', 'OPP_FGA',
                   'FT_PCT', 'OPP_FG3A', 'OFF_RATING', 'FG3_PCT', 'PCT_PTS_PAINT', 'FTA_RATE',
                   'DEF_RATING', 'OPP_FG3M', 'points']
        """

        bestcol = np.logical_not(self.train.columns.str.contains('_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]

        print 'After filtering: train shape , test shape:', self.train.shape, self.test.shape
        ###
        X = self.train.as_matrix(self.train.columns[:-1]).astype(float)
        y = self.train.as_matrix(['points'])[:, 0].astype(float)
        X_test = self.test.as_matrix(self.test.columns[:-1]).astype(float)
        self.y_test = self.test.as_matrix(['points'])[:, 0].astype(float)

        rf = GradientBoostingRegressor(**self.params)
        rf.fit(X, y)
        self.y_pred = rf.predict(X_test)

        error = mean_squared_error(self.y_pred, self.y_test)
        print 'Mean squared error:', error

        # Getting attributes from RandomForestRegressor()
        feat = rf.feature_importances_
        self.feat_imp = pd.DataFrame({'feature': self.train.columns[:-1],
                                      'importance': feat})
        self.feat_imp = self.feat_imp.sort('importance', ascending = False)
        self.feat_imp = self.feat_imp.reset_index().drop('index', axis = 1)
        self.estimators_ = rf.estimators_

    def features(self, color = 'purple'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.feat_imp.index[:15], self.feat_imp['importance'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.feat_imp.head(15))))
        ax.set_xticklabels(self.feat_imp['feature'].head(15), rotation=90, fontsize=15)
        ax.set_ylabel('Importance', fontsize = 15)

    def plot_result(self, color = 'green'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.scatter(self.y_test, self.y_pred, color = color,
                   label = 'Data', s = 100, alpha = 0.1)
        #ax.plot(x,pred_y, label = 'Fit', lw = 5)
        ax.set_xlabel('Actual +/- (points/48 min)',fontsize = 20)
        ax.set_ylabel('Predicted +/- (points/48 min)', fontsize = 20)
        ax.set_title('Results of Model', fontsize = 25)
        ax.set_xlim(-150,150)
        ax.set_ylim(-150,150)
        ax.legend(loc=2, fontsize = 20)
        ax.tick_params(labelsize =20)

class linModel():
    '''
    Takes in: parameters for linearRegression

    Returns: predictions
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1, lintype = 'Linear'):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF
        self.lintype = lintype
    def build_linmodel(self):
        x = 'nba_15season_all_150928.csv'
        self.train, self.test, self.id_df = lin_encode(filename = self.name,
                                                     min_cutoff = self.min_cutoff,
                                                     TRANSFORM_CUTOFF = self.t_cutoff)
        #Create validation set
        np.random.seed(2)
        self.test = self.test.reindex(np.random.permutation(self.test.index))
        self.test = self.test.iloc[:self.test.shape[0]/2,:]
        self.test = self.test.reset_index().drop('index', axis = 1)

        """
        # TESTING GROUNDS
        bestcol = ['PCT_FGA_2PT', 'PCT_FGA_3PT', 'PCT_PTS_2PT_MR', 'PCT_PTS_PAINT', 'PCT_PTS_2PT',
                   'OPP_OREB_PCT', 'DREB_PCT', 'PCT_PTS_3PT', 'PCT_PTS_FT', 'PCT_AST_FGM', 'EFG_PCT',
                   'FG_PCT', 'AST_PCT', 'TM_TOV_PCT', 'year', 'OPP_FG_PCT', 'OPP_EFG_PCT', 'TS_PCT',
                   'PCT_PTS_OFF_TOV', 'PCT_UAST_FGM', 'OREB_PCT', 'FTA_RATE', 'OPP_TOV_PCT',
                   'PCT_UAST_2PM', 'month', 'DEF_RATING', 'NET_RATING', 'OFF_RATING', 'BLK',
                   'OPP_BLKA', 'PFD', 'OPP_PF', 'PCT_AST_3PM', 'OPP_BLK', 'BLKA', 'team_26',
                   'OPP_FG3_PCT', 'PCT_UAST_3PM', 'team_14', 'team_8', 'team_0', 'FG3_PCT',
                   'PIE', 'away_game', 'team_22', 'team_2', 'team_23', 'team_28', 'team_1',
                   'team_13', 'team_12', 'REB_PCT', 'PCT_PTS_FB', 'team_18', 'team_20', 'team_3',
                   'PTS', 'team_29', 'OPP_FT_PCT', 'team_5', 'OPP_FTA_RATE', 'FG3M', 'OPP_REB',
                   'OPP_DREB', 'team_6', 'OPP_FGM', 'FGM', 'team_11', 'opponent_15', 'team_19',
                   'team_21', 'team_17', 'team_27']


        bestcol = ['fga_pm', 'TP_pm', 'FT_pm',
                   #'TPA_pm', 'FTA_pm'
                  ]

        bestcol = ['team_11', 'team_21', 'team_6', 'team_12', 'team_14', 'team_18', 'team_22',
                   'team_13', 'team_29', 'team_23', 'team_10', 'team_4', 'team_28', 'team_3',
                   'eFG', 'PCT_PTS_2PT_MR', 'PIE', 'OPP_FG3_PCT', 'AST_RATIO', 'OPP_FGM',
                   'FT_percent', 'opponent_13', 'AST_TO', 'OREB_opt', 'BLK', 'OPP_AST_opt',
                   'OPP_PTS_FB_opt', 'NET_RATING_opt', 'OFF_RATING', 'OPP_FTM', 'OPP_REB',
                   'PTS', 'REB', 'FT_PCT', 'OPP_FG3M', 'OPP_BLK_opt', 'TOV', 'PLUS_MINUS',
                   'MIN', 'OPP_PTS_OFF_TOV_opt', 'opponent_5', 'OPP_FTA_opt', 'BLKA_opt',
                   'FGA_opt', 'OPP_PFD', 'OPP_FG3A', 'opponent_27', 'TP_percent', 'OPP_PTS',
                   'FGA', 'opponent_26', 'OPP_BLK', 'opponent_23', 'opp_poss', 'OPP_PTS_PAINT',
                   'fga_pm', 'PTS_PAINT', 'OPP_PTS_2ND_CHANCE', 'FTM_opt', 'OPP_PTS_PAINT_opt',
                   'avg_pm', 'FT_pm', 'W_opt', 'FTA', 'PTS_FB_opt', 'OPP_PF', 'lineup', 'L_opt',
                   'OPP_STL', 'FTA_pm', 'TPA_pm', 'FG3A', 'TP_pm', 'num_poss',
                   'day', 'GP', 'dayofmonth', 'OPP_AST', 'OPP_PTS_OFF_TOV', 'STL',
                   'PTS_OFF_TOV', 'PTS_FB', 'OPP_PTS_2ND_CHANCE_opt', 'PTS_2ND_CHANCE',
                   'OPP_FTA', 'pace_bref', 'month', 'away_game', 'OPP_FGA_opt',
                   'OPP_PTS_FB', 'PTS_opt', 'AST_opt', 'PF', 'OPP_REB_opt', 'BLK_opt',
                   'PTS_OFF_TOV_opt', 'minutes', 'PCT_UAST_2PM', 'OPP_TOV', 'FG3M',
                   'OPP_FGA', 'PTS_PAINT_opt', 'OREB', 'FTM', 'OPP_BLKA_opt', 'DREB',
                   'FTA_opt', 'PACE', 'OPP_DREB', 'DREB_opt', 'OPP_OREB', 'OPP_BLKA',
                   'FG3M_opt', 'FGM', 'NET_RATING', 'opponent_24', 'AST', 'DEF_RATING',
                   'PCT_UAST_3PM', 'PCT_PTS_FB', 'REB_PCT', 'OPP_FT_PCT', 'OPP_FTA_RATE',
                   'DREB_PCT', 'team_5', 'team_27', 'team_1', 'team_24', 'team_20', 'team_25',
                   'team_7', 'team_19', 'team_8', 'team_2', 'team_0', 'team_15', 'team_17',
                   'team_9', 'team_16']


        bestcol = bestcol + ['points']
        """
        bestcol = np.logical_not(self.train.columns.str.contains('_10mean|_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]
        print self.train.shape, self.test.shape
        ###
        print 'train shape and test shape', self.train.shape, self.test.shape
        X = self.train.as_matrix(self.train.columns[:-1]).astype(float)
        y = self.train.as_matrix(['points'])[:, 0].astype(float)
        X_test = self.test.as_matrix(self.test.columns[:-1]).astype(float)
        self.y_test = self.test.as_matrix(['points'])[:, 0].astype(float)

        # Choose which type of linear regression to test
        if self.lintype == 'Linear':
            self.lr = LinearRegression(**self.params)
        elif self.lintype == 'Ridge':
            self.lr = Ridge(**self.params)
        elif self.lintype == 'Lasso':
            self.lr = Lasso(**self.params)
        else:
            return "Error: Choose lin. reg. type: 'Linear', 'Ridge', 'Lasso'"
        self.lr.fit(X, y)
        self.y_pred = self.lr.predict(X_test)

        error = mean_squared_error(self.y_pred, self.y_test)
        print 'Mean squared error:', error

        # Getting attributes from RandomForestRegressor()
        coef = self.lr.coef_
        self.coef_imp = pd.DataFrame({'feature': self.train.columns[:-1],
                                      'coefficient': coef})
        self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
        self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
        self.intercept = self.lr.intercept_

    def features(self, color = 'purple'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.coef_imp.index[:15], self.coef_imp['coefficient'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.coef_imp.head(15))))
        ax.set_xticklabels(self.coef_imp['feature'].head(15), rotation=90, fontsize=15)
        ax.set_ylabel('Coefficient', fontsize = 15)

    def plot_result(self, color = 'green'):
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.scatter(self.y_test, self.y_pred, color = color,
                   label = 'Data', s = 100, alpha = 0.1)
        #ax.plot(x,pred_y, label = 'Fit', lw = 5)
        ax.set_xlabel('Actual +/- (points/48 min)',fontsize = 20)
        ax.set_ylabel('Predicted +/- (points/48 min)', fontsize = 20)
        ax.set_title('Results of Model', fontsize = 25)
        ax.set_xlim(-150,150)
        ax.set_ylim(-150,150)
        ax.legend(loc=2, fontsize = 20)
        ax.tick_params(labelsize =20)
