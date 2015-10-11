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
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        bestcol = np.logical_not(self.train.columns.str.contains('_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
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
        error = mean_squared_error(self.y_pred, self.y_test)**0.5
        print 'RMSE:', error

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
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        bestcol = np.logical_not(self.train.columns.str.contains('_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
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

        error = mean_squared_error(self.y_pred, self.y_test)**0.5
        print 'RMSE:', error

        # Getting attributes from GradientBoostingRegressor()
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
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        bestcol = np.logical_not(self.train.columns.str.contains('_std|_max|_min|_5std|_5max|_5min|Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
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

        error = mean_squared_error(self.y_pred, self.y_test)**0.5
        print 'RMSE:', error

        # Getting attributes from LinearRegression()
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
