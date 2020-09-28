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

#Basic import to operate with data and folders in windows
import itertools
import os
import numpy as np
import pandas as pd
#Import for PCA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#Import for Random Forest
from sklearn.ensemble import RandomForestClassifier
#Import for Logistic Classifier
from sklearn.linear_model import LogisticRegression
#Import Metrics
from sklearn.metrics import accuracy_score, roc_auc_score
#Import for crossvalidation
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
#Import for Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Import for KNN
from sklearn.neighbors import KNeighborsClassifier
#IMport for Multilayer Perceptron Classifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
#Import for scaling
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer
#Import for SVM
from sklearn.svm import SVC
#Import for Decision tree
from sklearn.tree import DecisionTreeClassifier
#Import for adaboost
from sklearn.ensemble import AdaBoostClassifier
#Import for Bagging classifier
from sklearn.ensemble import BaggingClassifier
#Import for Gaussian process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


class cvclasificador:
    # Parameters:
        # X:         DataSet
        # Y:         Labels 
    def __init__(self, df, labels,categorical, column_to_group):
        #Array for original data scaled
        self.X=[]
        #Labels to predict
        self.Y = labels
        #Categorical columns
        self.df_ohe = pd.DataFrame()
        #Column to group the split
        self.group = column_to_group
        #splits of df to train and test algorithms
        self.train = []
        self.y_train = []
        self.test = []
        self.y_test = []
        
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
        
        #Define names of classification models
        self.names = ["SVM",
                     "LR",
                     "K-Neighbors",
                     "DecisionTree",
                     "NB",
                     "RandomForest",
                     "MLP",
                     "GP",
                     "AdaBoost",
                     "Bagging"]
        
        #Define classification models
        self.models = [
            SVC(gamma="auto", C=10000),
            LogisticRegression(penalty="l2"),
            KNeighborsClassifier(n_neighbors=5),
            DecisionTreeClassifier(),
            GaussianNB(),
            RandomForestClassifier(n_estimators=100),
            MLPClassifier(tol=0.05,max_iter=1000),
            GaussianProcessClassifier(),
            AdaBoostClassifier(n_estimators=100),
            BaggingClassifier(base_estimator=SVC(), n_estimators=10)
         ]

        #Variables categoricas
        print("Scaling categorical variables")
        for i in categorical.columns:
            one_hot = pd.get_dummies(categorical[i], drop_first=True)
            self.df_ohe[one_hot.columns] = one_hot

        #Variables continuas
        #Aplicamos los escaladores y añadimos la columna SIN ESCALAR por la que agrupar al df escalado
        df_columns = (df.columns)
        for i in range(0,len(scalers)):
                print ("Scaling numerical data with ", scalers[i][0])
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
        # self: Internal parameters
        # numbCross:  Number of cross-validations
    def getCrossValidation(self, numCross):
        #Define Crosvalidation with numCross folds
        cv = StratifiedKFold(n_splits=numCross, random_state=123)
        #Obtenemos los unicos de la columna por la que agrupoar
        list_unique_to_group = self.group.unique()
        #Obtenemos las etiquetas de los pacientes para hacer el split
        aux = pd.DataFrame()
        aux[self.group.name] = self.X[0][1][self.group.name].copy()
        aux['labels'] = self.Y
        #Drop duplicates to get one row with subject and labels
        aux = aux.drop_duplicates().reset_index(drop=True)
        #Almacenamos resultados de la clase
        array_accuracy=[]
        for i in range(0,len(self.X)):
            accuracy=[]
            #Aplicamos un algoritmo
            for algor, name in zip(self.models, self.names):
                # El np.round es par redondear a dos decimales
                print ("Starting cross-val with ", self.X[i][0], " and ", name)
                results = 0.0
                results_by_group = 0.0
                #Obtenenemos los conjuntos de datos
                for train_index, test_index in cv.split(aux[self.group.name], aux['labels']):
                    # Obtenemos los valores de los indices
                    train_filter = list_unique_to_group[train_index]
                    test_filter = list_unique_to_group[test_index]
                    #Filtramos por esos valores
                    x_train_data = self.X[i][1][self.X[i][1][self.group.name].isin(train_filter)].copy()
                    x_train_data.drop([self.group.name], axis=1, inplace=True)
                    y_train_data = self.Y[x_train_data.index]
                    #Test data
                    x_test_data = self.X[i][1][self.X[i][1][self.group.name].isin(test_filter)].copy()
                    x_test_data.drop([self.group.name], axis=1, inplace=True)
                    y_test_data = self.Y[x_test_data.index]
                    #Entrenamos 
                    algor.fit(x_train_data, y_train_data)
                    ####ACCURACY####
                    #Obtenemos acuraccy de la iteracion
                    results = results + algor.score(x_test_data,y_test_data)
                #Obtenemos media
                mean = np.round(results/numCross,4)
                print("Media:", mean)
                accuracy.append(mean)
            array_accuracy.append(accuracy)
        #Creamos dataframe con los resultados
        df_accuracy = pd.DataFrame(array_accuracy, index=self.scalnames,columns=self.names)
        return df_accuracy


# +
#Resultado base 
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
#Clean useless columns
column_to_drop = ['hour']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels to predict
labels = df['producto']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)
print("##################################")
print("Clasificacion todos los pacientes sin ingenieria de caracteristicas")
print("##################################")
#Call class to apply classifier
cls = cvclasificador(df,labels, categorical_data,column_group)
res = cls.getCrossValidation(5)
res.to_csv(res_path + "clasificacion_sin_ingenieria de caracteristicas.csv", sep=';', index=True)
# -
resultado = pd.read_csv(res_path + "clasificacion_sin_ingenieria de caracteristicas.csv", sep=';')
resultado

# +
#Resultado base sin variables sesgadas
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
#Clean useless columns
column_to_drop = ['hour', 'masa_grasa','sexo','imc']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels to predict
labels = df['producto']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit']].copy()
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject','visit']
df.drop(columns_to_drop, axis=1, inplace=True)
print("##################################")
print("Clasificacion resultado base sin variables sesgadas")
print("##################################")
#Call class to apply classifier
cls = cvclasificador(df,labels, categorical_data,column_group)
res = cls.getCrossValidation(5)
res.to_csv(res_path + "clasificacion_base_sin_sesgo.csv", sep=';', index=True)
# -

resultado = pd.read_csv(res_path + "clasificacion_base_sin_sesgo.csv", sep=';')
resultado

# +
#Resultado base sin outliers
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
#Clean useless columns
column_to_drop = ['hour', 'masa_grasa','sexo','imc']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels outliers
outlier = pd.read_csv(data_path + "outliers_clasificacion.csv", sep=';')
#Añadimos etiqueta y filtramos
df['outlier'] = outlier['outlier']
df = df[df['outlier'] == 1]
#Reseteamos indices 
df.reset_index(drop=True, inplace=True)
#Eliminamos el marcador de outlier
column_to_drop = ['outlier']
df.drop(column_to_drop,axis=1, inplace=True)

#Get labels to predict
labels = df['producto']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit']].copy()
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject','visit']
df.drop(columns_to_drop, axis=1, inplace=True)
print("##################################")
print("Clasificacion resultado base sin variables sesgadas")
print("##################################")
#Call class to apply classifier
cls = cvclasificador(df,labels, categorical_data,column_group)
res = cls.getCrossValidation(5)
res.to_csv(res_path + "clasificacion_base_sin_outlier.csv", sep=';', index=True)
# -

resultado = pd.read_csv(res_path + "clasificacion_base_sin_outlier.csv", sep=';')
resultado

# +
#Relative path
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "clasification_data.csv", sep=';')
print(df.columns)

#Get labels to predict
labels = df['producto']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit']].copy()
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject','visit','sexo','systolic_diff','imc_cat','imc','masa_grasa']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)
print(categorical_data.columns)
print("####################################################################")
print("Clasificacion todos los pacientes CON ingenieria de caracteristicas")
print("####################################################################")
# #Call class to apply classifier
cls = cvclasificador(df,labels, categorical_data,column_group)
res = cls.getCrossValidation(5)
res.to_csv(res_path + "clasificacion_ingenieria_caracteristicas.csv", sep=';', index=True)
# -

resultado = pd.read_csv(res_path + "clasificacion_ingenieria_caracteristicas.csv", sep=';')
resultado


