'''

__file__

    models.py

__description__

    This file gives example code that I used in building my models in an
    iPython notebook. I frequently adjust the parameters and features used
    in the model.

'''

import sys
import os
from collections import defaultdict
import datetime as dt

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import make_scorer, mean_squared_error

from nba_encoding import *
from kmeans_team import *
sys.path.append(os.path.abspath("../sql/"))
from get_tables import *

'''
Parameters for different Models
'''
# Random Forest
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
# Gradient Boosting
params_gbr = {'loss': 'ls',
              'learning_rate': 0.02,
              'n_estimators': 100,
              'max_depth': 5,
              'min_samples_split': 5,
              'min_samples_leaf': 3,
              'subsample': 0.7
             }
# Linear Regression
params_lin = {'fit_intercept': True,
          'normalize': False,
          'copy_X': True,
          'n_jobs': 1
          }
# Lasso Regression
params_lasso = {'alpha': 0.2,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}
# Ridge Regression
params_ridge = {'alpha': 0.4,
                'fit_intercept': True,
                'normalize': False,
                'copy_X': True}


class rfModel():
    '''
    Takes in: parameters for Random Forest

    Returns: predictions and an evaluation of the model
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF

    def build_rfmodel(self):
        """
        Takes in nothing (already set in Class Variable)

        Does Random Forest

        Makes predictions (self.y_pred) and calculates feature importances
        (self.feat_imp), oob prediction (self.oob_prediction_), and estimators
        (self.estimators_)

        Prints RMSE score on predicting the results of the last two months
        """
        self.train, self.test, self.id_df = forest_encode(
                                               filename = self.name,
                                               min_cutoff = self.min_cutoff,
                                               TRANSFORM_CUTOFF = self.t_cutoff)
        """
        # TESTING GROUNDS
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        # I do not always use these next two lines. I change them frequently
        bestcol = np.logical_not(self.train.columns.str.contains(
                                                       '_std|_max|_min|' +\
                                                       '_5std|_5max|_5min|'+\
                                                       'Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]

        print 'After filtering: train shape ,' +\
              ' test shape:', self.train.shape, self.test.shape

        # Convert to matrix for scikit learn and make a prediction
        X = self.train.as_matrix(self.train.columns[:-1]).astype(float)
        y = self.train.as_matrix(['points'])[:, 0].astype(float)
        X_test = self.test.as_matrix(self.test.columns[:-1]).astype(float)
        self.y_test = self.test.as_matrix(['points'])[:, 0].astype(float)

        rf = RandomForestRegressor(**self.params)
        rf.fit(X, y)
        self.y_pred = rf.predict(X_test)

        # Evaluate performance of the RF prediction
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
        """
        Takes in color preference

        Returns a bar plot of the 15 most important features. A dataframe of
        the feature importances can be found at self.feat_imp
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.feat_imp.index[:15], self.feat_imp['importance'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.feat_imp.head(15))))
        ax.set_xticklabels(self.feat_imp['feature'].head(15),
                           rotation=90, fontsize=15)
        ax.set_ylabel('Importance', fontsize = 15)

    def plot_result(self, color = 'green'):
        """
        Takes in color preference

        Returns a scatter plot of the predicted values vs. the actual values.
        This helps with visualizing the performance of the model.
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        # Plot fill
        x1 = np.arange(0,180)
        y1 = np.zeros(180)+180
        x2 = np.arange(0,-180, -1)
        y2 = np.zeros(180)-180
        ax.plot(x1, y1)
        plt.fill_between(x1, y1, 0, color=(0.01,0.40,0.1), alpha = 0.25)
        plt.fill_between(x2, y2, 0, color=(0.01,0.40,0.1), alpha = 0.25)

        # Plot Results
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
    Takes in: parameters for Gradient Boosting

    Returns: predictions and an evaluation performance of the model
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF
    def build_gbmodel(self):
        """
        Takes in nothing (already set in Class Variable)

        Does Stochastic Gradient Boosting Regression

        Makes predictions (self.y_pred) and calculates feature importances
        (self.feat_imp) and estimators (self.estimators_)

        Prints RMSE score on predicting the results of the last two months
        """
        # Can try two different encoding schemes: forest_encode or lin_encode
        # lin_encode one-hot encodes the categorical variables
        # usually not necessary for decision trees
        """
        self.train, self.test, self.id_df = forest_encode(
                                               filename = self.name,
                                               min_cutoff = self.min_cutoff,
                                               TRANSFORM_CUTOFF = self.t_cutoff)
        """
        self.train, self.test, self.id_df = lin_encode(
                                               filename = self.name,
                                               min_cutoff = self.min_cutoff,
                                               TRANSFORM_CUTOFF = self.t_cutoff)

        """
        # TESTING GROUNDS
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        # I do not always use these next two lines. I change them frequently
        bestcol = np.logical_not(self.train.columns.str.contains(
                                                                '_std|_max|'+\
                                                                '_min|_5std|'+\
                                                                '_5max|_5min|'+\
                                                                'Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]

        print 'After filtering: train shape ,'+\
              ' test shape:', self.train.shape, self.test.shape
        ###
        # Convert to matrix for scikit-learn
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
        """
        Takes in color preference

        Returns a bar plot of the 15 most important features. A dataframe of
        the feature importances can be found at self.feat_imp
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.feat_imp.index[:15], self.feat_imp['importance'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.feat_imp.head(15))))
        ax.set_xticklabels(self.feat_imp['feature'].head(15),
                           rotation=90, fontsize=15)
        ax.set_ylabel('Importance', fontsize = 15)

    def plot_result(self, color = 'green'):
        """
        Takes in color preference

        Returns a scatter plot of the predicted values vs. the actual values.
        This helps with visualizing the performance of the model.
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        # Plot fill
        x1 = np.arange(0,180)
        y1 = np.zeros(180)+180
        x2 = np.arange(0,-180, -1)
        y2 = np.zeros(180)-180
        ax.plot(x1, y1)
        plt.fill_between(x1, y1, 0, color=(0.01,0.40,0.1), alpha = 0.25)
        plt.fill_between(x2, y2, 0, color=(0.01,0.40,0.1), alpha = 0.25)

        # Plot results
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

    Returns: predictions and an evaluation of the performance of the model
    '''
    def __init__(self, params, name, min_cutoff = 1,
                  TRANSFORM_CUTOFF = 1, lintype = 'Linear'):
        self.params = params
        self.name = name
        self.min_cutoff = min_cutoff
        self.t_cutoff = TRANSFORM_CUTOFF
        self.lintype = lintype
    def build_linmodel(self):
        """
        Takes in nothing (already set in Class Variable)
        Choosen between unregularized, Lasso, or Ridge

        Does Linear Regression

        Makes predictions (self.y_pred) and calculates coefficients
        (self.coef_imp) and intercepts (self.intercept)

        Prints RMSE score on predicting the results of the last two months
        """
        self.train, self.test, self.id_df = lin_encode(
                                               filename = self.name,
                                               min_cutoff = self.min_cutoff,
                                               TRANSFORM_CUTOFF = self.t_cutoff)
        """
        # TESTING GROUNDS
        bestcol = [...put list of columns here for testing features...]
        bestcol = bestcol + ['points']
        """
        # Use these next two lines if you want to filter out these aggregates
        # I do not always use these next two lines. I change them frequently
        bestcol = np.logical_not(self.train.columns.str.contains(
                                                                '_std|_max|'+\
                                                                '_min|_5std|'+\
                                                                '_5max|_5min|'+\
                                                                'Unnamed'))
        bestcol = self.train.columns[bestcol]

        # Subset of features you want to train on
        self.train = self.train[bestcol]
        self.test = self.test[bestcol]

        print self.train.shape, self.test.shape
        ###
        print 'train shape and test shape', self.train.shape, self.test.shape
        #Convert to matrix for scikit-learn
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

        # Getting coefficients from Linear Regression()
        coef = self.lr.coef_
        self.coef_imp = pd.DataFrame({'feature': self.train.columns[:-1],
                                      'coefficient': coef})
        self.coef_imp = self.coef_imp.sort('coefficient', ascending = False)
        self.coef_imp = self.coef_imp.reset_index().drop('index', axis = 1)
        self.intercept = self.lr.intercept_

    def features(self, color = 'purple'):
        """
        Takes in color preference

        Returns a bar plot of the 15 largest coefficients. A dataframe of
        the coefficients can be found at self.coef_imp
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        ax.bar(self.coef_imp.index[:15], self.coef_imp['coefficient'].head(15),
               align = 'center', color = color, alpha = 1)
        ax.set_xticks(np.arange(0,len(self.coef_imp.head(15))))
        ax.set_xticklabels(self.coef_imp['feature'].head(15),
                           rotation=90, fontsize=15)
        ax.set_ylabel('Coefficient', fontsize = 15)

    def plot_result(self, color = 'green'):
        """
        Takes in color preference

        Returns a scatter plot of the predicted values vs. the actual values.
        This helps with visualizing the performance of the model.
        """
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize = (10,8))

        # Plot fill
        x1 = np.arange(0,180)
        y1 = np.zeros(180)+180
        x2 = np.arange(0,-180, -1)
        y2 = np.zeros(180)-180
        ax.plot(x1, y1)
        plt.fill_between(x1, y1, 0, color=(0.01,0.40,0.1), alpha = 0.25)
        plt.fill_between(x2, y2, 0, color=(0.01,0.40,0.1), alpha = 0.25)

        # Plot results
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
