# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
#General usage
import pandas as pd
import numpy as np
import time as tm
import os

#Particle Swarms Optimization
from pyswarms.single.global_best import GlobalBestPSO
#Measures
from sklearn import metrics
from sklearn.metrics import pairwise_distances

#Scalers and normalizers
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer, Normalizer

#Regressión algorithms
from sklearn.svm import SVR, NuSVR, LinearSVR

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

#Split
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score

#To plot swarms optimization
from pyswarms.utils.plotters.formatters import Mesher, Designer
from pyswarms.utils.plotters import plot_cost_history, plot_contour, plot_surface
from sklearn import decomposition
import matplotlib.pyplot as plt


# -

#########################################################################################
###################################  Logistic regresion ALG WITH ACCURACY   ####################
#########################################################################################
class logistic_regresion_pso:
    
    def __init__(self, df, labels, column_to_group):
        
        self.X = df
        self.Y = labels
        self.group = column_to_group        
        self.X[column_to_group.name] = column_to_group

            
    #Optimize function
    def optimize_logistic_regresion_accuracy(self,x):
        
        mean_score = 0.0
        cv = KFold(n_splits=5, random_state=123) 
        list_unique_to_group = self.group.unique()
        #optimize_algorithm with cross validation
        for train_index, test_index in cv.split(list_unique_to_group):
            # Obtenemos los valores de los indices
            train_filter = list_unique_to_group[train_index]
            test_filter = list_unique_to_group[test_index]
            #Filtramos por esos valores
            x_train = self.X[self.X[self.group.name].isin(train_filter)].copy()
            x_train.drop([self.group.name], axis=1, inplace=True)
            y_train = self.Y[x_train.index]
            x_test = self.X[self.X[self.group.name].isin(test_filter)].copy()
            x_test.drop([self.group.name], axis=1, inplace=True)
            y_test = self.Y[x_test.index]
            
            #Train Algorithm
            logistic_regresion =  LogisticRegression(penalty="l2", C=np.round(x[0],2), tol=10**-1)
            algor = logistic_regresion.fit(x_train,y_train)

            #Get prediction about test data
            print("r2:", algor.score(x_test,y_test))

            #Obtenemos el resultado de R2
            mean_score = mean_score + algor.score(x_test,y_test)
        res =  np.round(mean_score/5,4)
        return (1-res)

    #Backpropagation
    def f_logistic_regresion_accuracy(self,x):
#         Higher-level method to do forward_prop in the whole swarm.
#         Inputs
#         x: numpy.ndarray of shape (n_particles, dimensions). The swarm that will perform the search
#         Returns
#         numpy.ndarray of shape (n_particles, ) -  The computed loss for each particle
        n_particles = x.shape[0]
        j = [self.optimize_logistic_regresion_accuracy(x[i]) for i in range(n_particles)]
        return np.array(j)
    
    #############################################################################
    ######################      Logistic regresion     #################################
    #############################################################################
    def run(self):
        #### Logistic regresion Optimization #####
        #### Parameters PSO ######
        iterations_logistic_regresion = 2
        particles_logistic_regresion = 1
        # Range of values for C paramters in PSO Optimization
        # default = default=1e-9
        min_c = 0.01
        max_c = 10000
        ######################
        #Create bounds up and down for the parameters of optimize function
        max_bound = max_c * np.ones(1)
        min_bound = min_c * np.ones(1)
        limite = (min_bound, max_bound)
        
        # instatiate the optimizer(parameters for the swarms(weight to the communication of the swarms(best swarms, best local swarms and inertia)))
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # #Create swarm with n_particles, dimensions of dimensional space, parameters for the swarms and bounds
        optimizer = GlobalBestPSO(n_particles=particles_logistic_regresion, dimensions=1, options=options, bounds=limite)
        # #Run the optimization
        cost, pos = optimizer.optimize(self.f_logistic_regresion_accuracy, iters=iterations_logistic_regresion)
        result = [cost, pos[0]]
        return result


#########################################################################################
###################################  LinearSVR ALG WITH R^2   #######################
#########################################################################################
class linearsvr_pso:
    
    def __init__(self, df, labels, column_to_group):
        self.X = df
        self.Y = labels
        self.group = column_to_group        
        self.X[column_to_group.name] = column_to_group
    
        
    #Optimize function
    def optimize_linearsvr_r2(self,x):
        mean_score = 0.0
        cv = KFold(n_splits=5, random_state=123) 
        list_unique_to_group = self.group.unique()
        #optimize_algorithm with cross validation
        for train_index, test_index in cv.split(list_unique_to_group):
            # Obtenemos los valores de los indices
            train_filter = list_unique_to_group[train_index]
            test_filter = list_unique_to_group[test_index]
            #Filtramos por esos valores
            x_train = self.X[self.X[self.group.name].isin(train_filter)].copy()
            x_train.drop([self.group.name], axis=1, inplace=True)
            y_train = self.Y[x_train.index]
            x_test = self.X[self.X[self.group.name].isin(test_filter)].copy()
            x_test.drop([self.group.name], axis=1, inplace=True)
            y_test = self.Y[x_test.index]
            
            #Train Algorithm
            linearsvr =  LinearSVR(C=np.round(x[0],2), max_iter=20000, tol=10**-1)
            algor = linearsvr.fit(x_train,y_train)

            #Get prediction about test data
            print("Accuracy:", algor.score(x_test,y_test))

            #Obtenemos el resultado de R2
            mean_score = mean_score + algor.score(x_test,y_test)
        res =  np.round(mean_score/5,4)
        return (1-res)

    #Backpropagation
    def f_linearsvr_r2(self,x):
#         Higher-level method to do forward_prop in the whole swarm.
#         Inputs
#         x: numpy.ndarray of shape (n_particles, dimensions). The swarm that will perform the search
#         Returns
#         numpy.ndarray of shape (n_particles, ) -  The computed loss for each particle
        n_particles = x.shape[0]
        j = [self.optimize_linearsvr_r2(x[i]) for i in range(n_particles)]

        #j = [forward_prop(x[i]) for i in range(n_particles)]
        return np.array(j)
    #############################################################################
    ######################      LinearSVR     ###################################
    #############################################################################
    def run(self):
        #### LinearSVR Optimization #####
        #### Parameters PSO ######
        iterations_linearsvr = 2
        particles_linearsvr = 1
        # Range of values for nusvr paramters in PSO Optimization
        # eps = 0.001 -- 5
        min_c = 0.0001
        max_c = 10000
        ######################
        #Create bounds up and down for the parameters of optimize function
        max_bound = max_c * np.ones(1)
        min_bound = min_c * np.ones(1)
        limite = (min_bound, max_bound)
        
        # instatiate the optimizer(parameters for the swarms(weight to the communication of the swarms(best swarms, best local swarms and inertia)))
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        # #Create swarm with n_particles, dimensions of dimensional space, parameters for the swarms and bounds
        optimizer = GlobalBestPSO(n_particles=particles_linearsvr, dimensions=1, options=options, bounds=limite)
        # #Run the optimization
        cost, pos = optimizer.optimize(self.f_linearsvr_r2, iters=iterations_linearsvr)
        result = [cost, pos[0]]
        return result

#PATH
path = os.getcwd()
data_path = path + "/data/"

# +
#Clasificacion
df_clasificacion = pd.read_csv(data_path + "clasificacion_scaled.csv", delimiter=';')

#Get labels to predict
labels = df_clasificacion['labels']
column = df_clasificacion['subject']
#Delete them from raw data
columns_to_drop = ['subject', 'Unnamed: 0', 'labels']
df_clasificacion.drop(columns_to_drop, axis=1, inplace=True)

pso = logistic_regresion_pso(df_clasificacion,labels,column)
pso.run()


# +
#Regresion data placebo
df_placebo = pd.read_csv(data_path + "regresion_placebo_scaled.csv", delimiter=';')

#Get labels to predict
labels = df_placebo['labels']
column = df_placebo['subject']
#Delete them from raw data
columns_to_drop = ['subject', 'Unnamed: 0', 'labels']
df_placebo.drop(columns_to_drop, axis=1, inplace=True)

pso = linearsvr_pso(df_placebo,labels,column)
pso.run()

# +
#Regresion data experimental
df_experimental = pd.read_csv(data_path + "regresion_producto_scaled.csv", delimiter=';')

#Get labels to predict
labels = df_experimental['labels']
column = df_experimental['subject']
#Delete them from raw data
columns_to_drop = ['subject', 'Unnamed: 0', 'labels']
df_experimental.drop(columns_to_drop, axis=1, inplace=True)


print(df_experimental.columns)

pso = linearsvr_pso(df_experimental,labels,column)
pso.run()
# -



