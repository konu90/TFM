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

import pandas as pd
import os
import numpy as np
import itertools
from sklearn.decomposition import PCA
from sklearn import decomposition
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


# +
#Clase que aplica unos escaladores a un conjunto de datos,
#y las plotea aplicando colores y simbolos segun la clase o label a la que pertenece

class scalers_plot:
    #Inicializción de la clase
    def __init__(self, df, categorical, labels):
        #Original data scaled
        self.X=[]
        #Categorical variables
        self.categorical = categorical
        #Labels to plot
        self.Y=labels
        #Name of scalers
        self.scalnames = []
        self.simbols = [".",",","o","v", "^", "<", ">", "1", "2", "3","4", "8", "s",
                        "p","P", "*","h","H","+","x","X","D","d"]
        self.colors = ['red', 'green', 'blue', 'purple','yellow']
        
        
        #Define scalers
        self.scalers = [("MinMaxScaler", MinMaxScaler()),
           ("StandardScaler", StandardScaler()),           
           ("MaxAbsScaler", MaxAbsScaler()),
           ("RobustScaler", RobustScaler()),
           ("Quant-Normal", QuantileTransformer(output_distribution="normal")),
           ("Quant-Uniform", QuantileTransformer(output_distribution="uniform")),
           ("PowerTransf-yeoJhonson", PowerTransformer(method='yeo-johnson'))
           ]
        
        #Aplicamos los escaladores
        df_columns = (df.columns)
        for i in range(0,len(self.scalers)):
                print ("Scaling data with ", self.scalers[i][0])
                self.scalnames.append(self.scalers[i][0])
                #Creamos un dataframe con los datos escalados
                aux = pd.DataFrame(self.scalers[i][1].fit_transform(df), columns = df_columns)
                #Añadimos las variables categoricas
#                 aux[categorical.columns] = categorical
                self.X.append([self.scalers[i][0], aux])
                print ("Finished scaling data with ", self.scalers[i][0])
        print ("Datas scaled ready with ", len(self.X), " scalers")
    
    #Metodo que aplica PCA y plotea en funcion de las componentes elegidas (maximo 3)
    def plot(self, components):
        
        # #PCA
        # Select number of components
        pca = decomposition.PCA(n_components=components)
        for i in range(0,len(self.X)):                  
            #Apply PCA on dataframe and the variable that have the number of component of PCA
            principalComponents = pca.fit_transform(self.X[i][1])
            #Save the result in a dataframe(principalDF)
            if(components == 2):
                principalDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
            else:
                principalDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2','PC3'])
            #Add labels
            principalDF['label'] = self.Y
            #Para ver el grado de variabilidad de las componentes elegidas
            print(pca.explained_variance_ratio_)
            #Plot PCA over 2 components
            if(components == 2):
                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=df.columns)
                print(loadings)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                k = 0
                for j in (self.Y.unique()):
                    xs = principalDF['PC1'][principalDF['label'] == j]
                    ys = principalDF['PC2'][principalDF['label'] == j]
                    ax.scatter(xs,ys, c=self.colors[k])
                    k = k+1
                plt.title("PCA con " + self.scalers[i][0])
                plt.legend(principalDF['label'].unique())
                plt.show()
            #Plot PCA over 3 components
            else:
                loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2','PC3'], index=df.columns)
                print(loadings)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                k = 0
                for j in (self.Y.unique()):
                    xs = principalDF['PC1'][principalDF['label'] == j]
                    ys = principalDF['PC2'][principalDF['label'] == j]
                    zs = principalDF['PC3'][principalDF['label'] == j]
                    ax.scatter(xs,ys,zs, c=self.colors[k])
                    k = k+1
                plt.title("PCA con " + self.scalers[i][0])
                plt.legend(principalDF['label'].unique())
                plt.legend
                plt.show()

# +
###########READ DATA##############

#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#read data
df = pd.read_csv(path + "data_with_features.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#get variables to plot
#sexo
labels_sexo = df['producto'].copy()
#producto
labels_producto = df['producto'].copy()
labels_producto = labels_producto.map({1:'plabeco',2:'producto'})
categorical_labels = [labels_sexo, labels_producto]

df = df.drop(['subject','sexo','imc_cat','producto','visit'],axis=1)

pca = scalers_plot(df, labels_sexo, labels_sexo)
pca.plot(3)
# #drop categorical variables
# df = df.drop(['subject','sexo'])
# #Columnas numericas
# numerical_columns = ['kcal', 'met', '% sedentary', '% light', '% moderate', '%vigorous', 'steps counts', 'systolic', 'diastolic', 'pam', 'pp', 'heart rate', 'edad','peso','imc','masa_grasa','masa_magra']
# #Columnas categoricas

# labels = df['producto']
# df = df.drop(['subject','sexo','producto'], axis=1)
# df
# -


