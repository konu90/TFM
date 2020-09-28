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
import itertools
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import PowerTransformer
    
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import os
from sklearn.externals import joblib


# -

class cvregresor:
    # Parameters:
        # X:         DataSet
        # Y:         Labels 
    # It will use MinMaxScaler for data normalization
    def __init__(self, df, labels, categorical, column_to_group):
        #Original data scaled
        self.X=[]
        #splits of df to train and test algorithms
        self.train = []
        self.y_train = []
        self.test = []
        self.y_test = []
        #Labels to predict
        self.Y=labels
        #Categorical columns
        self.df_ohe = pd.DataFrame()
        #Column to group the split
        self.group = column_to_group
        #Name of scalers
        self.scalnames = []
        #Define scalers
        scalers = [("MinMaxScaler", MinMaxScaler()),
           ("StandardScaler", StandardScaler()),           
           ("MaxAbsScaler", MaxAbsScaler()),
           ("RobustScaler", RobustScaler()),
           ("Quant-Normal", QuantileTransformer(output_distribution="normal")),
           ("Quant-Uniform", QuantileTransformer(output_distribution="uniform")),
           ("PowerTransf-yeoJhonson", PowerTransformer(method='yeo-johnson'))
           ]
        #Define regression models
        self.models = [SVR(C=1000),
            LinearSVR(C=1000),
            NuSVR(C=1000),
            DecisionTreeRegressor(),
            MLPRegressor(tol=0.05,max_iter=1000 ),
            KNeighborsRegressor(),
            GaussianProcessRegressor(random_state=0),
            AdaBoostRegressor(random_state=0, n_estimators=100),
            BaggingRegressor(base_estimator=SVR(), n_estimators=100, random_state=0)
         ]
        #Define names of regression models
        self.names = ["SVR",
                  "LinearSVR",
                 "NuSVR",          
                 "DecisionTree",
                 "MLP",
                 "KVecinos",
                 "GaussianProcess",
                 "Adaboost",
                 "Bagging"
                ]
        
        #Variables categoricas
        print("Scaling categorical variables")
        for i in categorical.columns:
            one_hot = pd.get_dummies(categorical[i], drop_first=True)
            self.df_ohe[one_hot.columns] = one_hot
            
        #Aplicamos los escaladores y añadimos la columna SIN ESCALAR por la que agrupar al df escalado
        df_columns = (df.columns)
        for i in range(0,len(scalers)):
                print ("Scaling data with ", scalers[i][0])
                #Añadimos nombre del escalador usado
                self.scalnames.append(scalers[i][0])
                #Escalamos los datos y añadimos
                self.X.append([scalers[i][0], pd.DataFrame(scalers[i][1].fit_transform(df), columns=df_columns)])
                #Añadimos variables categoricas
                self.X[i][1][self.df_ohe.columns] = self.df_ohe
                #añadimos columna por la que agrupar
                self.X[i][1][self.group.name] = column_to_group
                print ("Finished scaling data with ", scalers[i][0])           
        print ("Datas scaled ready with ", len(self.X), " scalers")
    
    # Parameters
        # numbCross:  Number of cross-validations
        # algors:     List of algorithms. Every algorithm must be complete with all its parameters
        # names:      Names of every algorithm (for visualization purposes)
    def getCrossValidation(self, numCross):
        #Obtenemos los unicos de la columna por la que agrupoar
        list_unique_to_group = self.group.unique()
        cv = KFold(n_splits=numCross, random_state=123)  
        #Para cada df escalado...
        array_r2 = []
        array_mae= []
        for i in range(0,len(self.X)):
            r2=[]
            mae = []
            #Aplicamos un algoritmo
            for algor, name in zip(self.models, self.names):
                # El np.round es par redondear a dos decimales
                print ("Starting cross-val with ", self.X[i][0], " and ", name)
                #R2
                results_r2 = 0.0
                #MASE
                results_mae= 0.0
                #Obtenenemos los conjuntos de datos
                for train_index, test_index in cv.split(list_unique_to_group):
                    # Obtenemos los valores de los indices
                    train_filter = list_unique_to_group[train_index]
                    test_filter = list_unique_to_group[test_index]
                    #Filtramos por esos valores
                    x_train_data = self.X[i][1][self.X[i][1][self.group.name].isin(train_filter)].copy()
                    x_train_data.drop([self.group.name], axis=1, inplace=True)
                    y_train_data = self.Y[x_train_data.index]
                    x_test_data = self.X[i][1][self.X[i][1][self.group.name].isin(test_filter)].copy()
                    x_test_data.drop([self.group.name], axis=1, inplace=True)
                    y_test_data = self.Y[x_test_data.index]
        
                    #Entrenamos
                    algor.fit(x_train_data, y_train_data)
                    
                    #Get prediction about test data
                    predicted = algor.predict(x_test_data)
                    print("r2:", algor.score(x_test_data,y_test_data))
                    print("mase:",self.mase(y_test_data, predicted,1))

                    #Obtenemos el resultado de R2
                    results_r2 = results_r2 + algor.score(x_test_data,y_test_data)
                    #Get Mase
                    results_mae = results_mae + self.mae(y_test_data, predicted)

                #Obtenemos media r2
                mean_r2 = np.round(results_r2/numCross,4)
                print("Media:", mean_r2)
                r2.append(mean_r2)
                #Obtenemos media mase
                mean_mae = np.round(results_mae/numCross,4)
                print("Media mase:", mean_mae)
                mae.append(mean_mae)
                
            #Añadimos al array de resultados    
            array_r2.append(r2)
            array_mae.append(mae)
        #Creamos dataframe con los resultados
        df_r2 = pd.DataFrame(array_r2, index=self.scalnames,columns=self.names)
        df_mae = pd.DataFrame(array_mae,index=self.scalnames,columns=self.names)
        return df_r2,df_mae
    
    ##### SIMPLE ERROR real - predicted
    def _error(self, actual, predicted):
        return actual - predicted

    #######MEAN ABSOLUTE ERROR#####
    def mae(self, actual, predicted):
        return np.mean(np.abs(self._error(actual, predicted)))
    
    ######NAIVE FORECASTING########
    """ Naive forecasting method which just repeats previous samples """
    def _naive_forecasting(self, actual, seasonality):
        return actual[:-seasonality]
    
    def mase(self, actual, predicted, seasonality):
        actual = actual.to_numpy()
        return self.mae(actual, predicted) / self.mae(actual[seasonality:], self._naive_forecasting(actual, seasonality))


# +
print("##################################")
print("Regresión sin ingenieria de caracteristicas GRUPO PLACEBO, CV:80-20")
print("##################################")

#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"

#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)
#Filter by placebo
df = df[df['producto'] == 1].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

#Call class to apply Regressor
regresor = cvregresor(df,labels,categorical_data, column)
r2,mae = regresor.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "placebo_r2_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
mae.to_csv(res_path + "placebo_mae_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "placebo_r2_sin_ingenieria_caracteristicas.csv", sep=';')
resultado


resultado = pd.read_csv(res_path + "placebo_mae_sin_ingenieria_caracteristicas.csv", sep=';')
resultado



# +
print("##################################")
print("Regresión sin ingenieria de caracteristicas, GRUPO EXPERIMENTAL CV:80-20")
print("##################################")
#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)
#Filter by experimental
df = df[df['producto'] == 2].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

#Call class to apply Regressor
regresor = cvregresor(df,labels,categorical_data, column)
r2,mae = regresor.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "producto_r2_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
mae.to_csv(res_path + "producto_mae_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "producto_r2_sin_ingenieria_caracteristicas.csv", sep=';')
resultado

resultado = pd.read_csv(res_path + "producto_mae_sin_ingenieria_caracteristicas.csv", sep=';')
resultado

# +
print("##################################")
print("Regresión sin OUTLIERS GRUPO PLACEBO, CV:80-20")
print("##################################")

#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"

#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels outliers
outlier = pd.read_csv(data_path + "outliers_placebo.csv", sep=';')
#Añadimos etiqueta y filtramos
df['outlier'] = outlier['outlier']
df = df[df['outlier'] == 1]
#Reseteamos indices 
df.reset_index(drop=True, inplace=True)
#Eliminamos el marcador de outlier
column_to_drop = ['outlier']
df.drop(column_to_drop,axis=1, inplace=True)

#Filter by placebo
df = df[df['producto'] == 1].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

#Call class to apply Regressor
regresor = cvregresor(df,labels,categorical_data, column)
r2,mae = regresor.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "placebo_r2_sin_outliers.csv", sep=';', index=True)
# mae.to_csv(res_path + "placebo_mae_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "placebo_r2_sin_outliers.csv", sep=';')
resultado

# +
print("##################################")
print("Regresión sin outliers, GRUPO EXPERIMENTAL CV:80-20")
print("##################################")
#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels outliers
outlier = pd.read_csv(data_path + "outliers_experimental.csv", sep=';')
#Añadimos etiqueta y filtramos
df['outlier'] = outlier['outlier']
df = df[df['outlier'] == 1]
#Reseteamos indices 
df.reset_index(drop=True, inplace=True)
#Eliminamos el marcador de outlier
column_to_drop = ['outlier']
df.drop(column_to_drop,axis=1, inplace=True)

#Filter by experimental
df = df[df['producto'] == 2].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

#Call class to apply Regressor
regresor = cvregresor(df,labels,categorical_data, column)
r2,mae = regresor.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "producto_r2_sin_outliers.csv", sep=';', index=True)
# mae.to_csv(res_path + "producto_mae_sin_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "producto_r2_sin_outliers.csv", sep=';')
resultado

# +
print("##################################")
print("Regresión con ingenieria de caracteristicas PLACEBO, CV:80-20")
print("##################################")
#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"

#Read data
df = pd.read_csv(data_path + "regresion_data.csv", sep=';')

#Filter by placebo
df = df[df['producto'] == 1].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo','imc_cat','imc']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)

#Call class to apply Regressor
test = cvregresor(df,labels,categorical_data, column)
r2,mae = test.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "placebo_r2_con_ingenieria_caracteristicas.csv", sep=';', index=True)
mae.to_csv(res_path + "placebo_mae_con_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "placebo_r2_con_ingenieria_caracteristicas.csv", sep=';')
resultado

# +
print("##################################")
print("Regresión con ingenieria de caracteristicas EXPERIMENTAL, CV:80-20")
print("##################################")
#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"

#Read data
df = pd.read_csv(data_path + "regresion_data.csv", sep=';')

#Filter by placebo
df = df[df['producto'] == 2].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
column = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo','imc_cat','imc']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)
#Call class to apply Regressor
test = cvregresor(df,labels,categorical_data, column)
r2,mase = test.getCrossValidation(5)
print("Guardando ficheros...")
r2.to_csv(res_path + "producto_r2_con_ingenieria_caracteristicas.csv", sep=';', index=True)
mase.to_csv(res_path + "producto_mase_con_ingenieria_caracteristicas.csv", sep=';', index=True)
print("Finalizado")
# -

resultado = pd.read_csv(res_path + "producto_r2_con_ingenieria_caracteristicas.csv", sep=';')
resultado




