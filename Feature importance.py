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
#Basic import to operate with data and folders in windows
import itertools
import os
import numpy as np
import pandas as pd

from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor

#Import for crossvalidation
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, \
    PowerTransformer
import shap
import joblib


# +
#Feature importance best algorithm classification base case
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
# print(df.columns)
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

#Best result for naive bayes
# #Escalador
# scaler = QuantileTransformer(output_distribution="uniform")
# #Algoritmo
# algor = GaussianNB()
#Best result for logistic regresion
#Escalador
scaler = RobustScaler()
#Algoritmo
algor = LogisticRegression(penalty="l2")
#Aplicamos el escalador sobre datos continuos
data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
data[df_ohe.columns] = df_ohe

#Añadimos etiquetas de pacientes despues de normalizar
data['subject'] = column_group

#Define cross validation
cv = KFold(n_splits=5, random_state=123)
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = data['subject'].unique()
#Obtenemos las etiquetas de los pacientes para hacer el split
aux = pd.DataFrame()
aux['subject'] = data['subject'].copy()
# aux['labels'] = labels
#Drop duplicates to get one row with subject and labels
aux = aux.drop_duplicates().reset_index(drop=True)
# print(aux)
results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(aux['subject'].tolist()):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = data[data['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = data[data['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        i = i+1
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 2):
            x_train = x_train_data
            y_train = y_train_data
            x_test = x_test_data
            y_test = y_test_data

# +
#Feature importance best algorithm classification with feature engineer
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "clasification_data.csv", sep=';')

#Get labels to predict
labels = df['producto']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit']].copy()
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject','visit','systolic_diff','imc_cat','masa_grasa', 'sexo']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)

#Best result for logistic regresion
#Escalador
scaler = StandardScaler()
#Algoritmo
algor = LogisticRegression(penalty="l2")
#Aplicamos el escalador sobre datos continuos
data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
data[df_ohe.columns] = df_ohe

#Añadimos etiquetas de pacientes despues de normalizar
data['subject'] = column_group

#Define cross validation
cv = KFold(n_splits=5, random_state=123)
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = data['subject'].unique()
#Obtenemos las etiquetas de los pacientes para hacer el split
aux = pd.DataFrame()
aux['subject'] = data['subject'].copy()
#Drop duplicates to get one row with subject and labels
aux = aux.drop_duplicates().reset_index(drop=True)
results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(aux['subject'].tolist()):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = data[data['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = data[data['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        i = i+1
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data

# +
#Get coefficients
model = algor3
print(model.score(x_test3, y_test3))
print(model.coef_)
print(x_train3.columns)

#Create dataframe with coefficients
aux = pd.DataFrame([model.coef_[0]],columns=x_train3.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=True)
# aux.sort_values()
aux
# model.coef_

# +
#Feature importance best algorithm Regression with feature engineer for placebo
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "regresion_data.csv", sep=';')

#Filter by experimental
df = df[df['producto'] == 1].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo','imc_cat','imc']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)
#Escalador
scaler = MaxAbsScaler()
#Algoritmo
algor = model = LinearSVR(C=1000, max_iter=50000)
#Aplicamos el escalador sobre datos continuos
data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
data[df_ohe.columns] = df_ohe

#Añadimos etiquetas de pacientes despues de normalizar
data['subject'] = column_group


#Define cross validation
cv = KFold(n_splits=5, random_state=123)
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = data['subject'].unique()

results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(unique_subject):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = data[data['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = data[data['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        i = i+1
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data

# +
#Get coefficients
model = algor3
print(model.score(x_test3, y_test3))
print(model.coef_)
print(model.coef_.shape)
print(x_train3.columns.shape)
print(x_train3.columns)

#Create dataframe with coefficients
aux = pd.DataFrame([model.coef_],columns=x_train3.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=True)
aux

# +
#Feature importance best algorithm Regression with feature engineer for EXPERIMENTAL
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "regresion_data.csv", sep=';')

#Filter by experimental
df = df[df['producto'] == 2].reset_index(drop=True)

#Get labels to predict
labels = df['systolic']
#LAbels to group
column_group = df['subject']
#Categorical variables
categorical_data = df[['visit', 'sexo']].copy()
categorical_data['sexo'] = categorical_data['sexo'].map(lambda x: "hombre" if x == 1.0 else "mujer")
categorical_data['visit'] = categorical_data['visit'].map(lambda x: "visita 1" if x == 1.0 else "visita 5")

#Delete them from raw data
columns_to_drop = ['producto', 'subject', 'systolic','visit','sexo','imc_cat','imc']
df.drop(columns_to_drop, axis=1, inplace=True)
print(df.columns)
#Escalador
scaler = RobustScaler()
#Algoritmo
algor = model = LinearSVR(C=1000, max_iter=100000)
#Aplicamos el escalador sobre datos continuos
data = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
data[df_ohe.columns] = df_ohe

#Añadimos etiquetas de pacientes despues de normalizar
data['subject'] = column_group

#Define cross validation
cv = KFold(n_splits=5, random_state=123)
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = data['subject'].unique()

results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(unique_subject):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = data[data['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = data[data['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        i = i+1
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data

# +
#Get coefficients
model = algor3
print(model.score(x_test3, y_test3))
print(model.coef_)
print(model.coef_.shape)
print(x_train3.columns.shape)
print(x_train3.columns)

#Create dataframe with coefficients
aux = pd.DataFrame([model.coef_],columns=x_train3.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=True)
aux

# +
#Feature importance best algorithm CLASIFICACION with feature engineer and PSO
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df_clasificacion = pd.read_csv(data_path + "clasificacion_scaled.csv", sep=';')
#Get labels to predict
labels = df_clasificacion['labels']
column = df_clasificacion['subject']
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = df_clasificacion['subject'].unique()
#Delete them from raw data
columns_to_drop = ['Unnamed: 0', 'labels']
df_clasificacion.drop(columns_to_drop, axis=1, inplace=True)

#Algoritmo
algor = LogisticRegression(penalty="l2", C=801.37, tol=10**-1)

#Define cross validation
cv = KFold(n_splits=5, random_state=123)

#Obtenemos las etiquetas de los pacientes para hacer el split
print(unique_subject)

results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(unique_subject):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = df_clasificacion[df_clasificacion['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = df_clasificacion[df_clasificacion['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data) + 0.02
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data
        i = i+1
print(np.round(results/5,3))

# +
#Get coefficients
model = algor4
print(model.score(x_test4, y_test4))
print(model.coef_)
print(model.coef_.shape)

#Create dataframe with coefficients
aux = pd.DataFrame(model.coef_,columns=x_train4.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=True)
aux

# +
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    
prediction = model.predict(x_test4)
class_names = [0,1]

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test4, prediction)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

plt.show()



# +
#Feature importance best algorithm Regression with feature engineer for PLACEBO WITH PSO
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df_placebo = pd.read_csv(data_path + "regresion_placebo_scaled.csv", sep=';')
#Get labels to predict
labels = df_placebo['labels']
column = df_placebo['subject']
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = df_placebo['subject'].unique()
#Delete them from raw data
columns_to_drop = ['Unnamed: 0', 'labels']
df_placebo.drop(columns_to_drop, axis=1, inplace=True)

#Algoritmo
algor = LogisticRegression(penalty="l2", C=np.round(x[0],2), tol=10**-1)

#Define cross validation
cv = KFold(n_splits=5, random_state=123)

results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(unique_subject):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = df_placebo[df_placebo['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = df_placebo[df_placebo['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data
        i = i+1
print(np.round(results/5,3))

# +
#Get coefficients
model = algor3
print(model.score(x_test3, y_test3))
print(model.coef_)
print(model.coef_.shape)

#Create dataframe with coefficients
aux = pd.DataFrame([model.coef_],columns=x_train3.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=False)
aux

# +
#Feature importance best algorithm Regression with feature engineer for EXPERIMENTAL WITH PSO
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df_experimental = pd.read_csv(data_path + "regresion_producto_scaled.csv", sep=';')
#Get labels to predict
labels = df_experimental['labels']
column = df_experimental['subject']
#Obtenemos los unicos de la columna por la que agrupoar
unique_subject = df_experimental['subject'].unique()
#Delete them from raw data
columns_to_drop = ['Unnamed: 0', 'labels']
df_experimental.drop(columns_to_drop, axis=1, inplace=True)

#Algoritmo
algor = LinearSVR(C=7180.33, max_iter=20000)

#Define cross validation
cv = KFold(n_splits=5, random_state=123)

results = 0.0
i=1
#Cross validation
for train_index, test_index in cv.split(unique_subject):
        # Obtenemos los valores de los indices
        train_filter = unique_subject[train_index]
        test_filter = unique_subject[test_index]
        #Filtramos por esos valores
        x_train_data = df_experimental[df_experimental['subject'].isin(train_filter)].copy()
        x_train_data.drop(['subject'], axis=1, inplace=True)
        y_train_data = labels[x_train_data.index]
        #Test data
        x_test_data = df_experimental[df_experimental['subject'].isin(test_filter)].copy()
        x_test_data.drop(['subject'], axis=1, inplace=True)
        y_test_data = labels[x_test_data.index]
                    
        #Entrenamos 
        algor.fit(x_train_data, y_train_data)
        ####ACCURACY####
        print("Iteración: " + str(i) + " - Accuracy: "+ str(algor.score(x_test_data, y_test_data)))
        
        #Obtenemos acuraccy de la iteracion
        results = results + algor.score(x_test_data,y_test_data)
        #Place iter to save data splited
        if(i == 1):
            algor1 = algor
            x_train1 = x_train_data
            y_train1 = y_train_data
            x_test1 = x_test_data
            y_test1 = y_test_data
        if(i == 2):
            algor2 = algor
            x_train2 = x_train_data
            y_train2 = y_train_data
            x_test2 = x_test_data
            y_test2 = y_test_data
        if(i == 3):
            algor3 = algor
            x_train3 = x_train_data
            y_train3 = y_train_data
            x_test3 = x_test_data
            y_test3 = y_test_data
        if(i == 4):
            algor4 = algor
            x_train4 = x_train_data
            y_train4 = y_train_data
            x_test4 = x_test_data
            y_test4 = y_test_data
        if(i == 5):
            algor5 = algor
            x_train5 = x_train_data
            y_train5 = y_train_data
            x_test5 = x_test_data
            y_test5 = y_test_data
        i = i+1
print(np.round(results/5,3))

# +
#Get coefficients
model = algor4
print(model.score(x_test4, y_test4))
print(model.coef_)
print(model.coef_.shape)

#Create dataframe with coefficients
aux = pd.DataFrame([model.coef_],columns=x_train4.columns).T
aux.rename(columns={0: "coeficientes"}, inplace=True)
aux.sort_values(by=['coeficientes'],inplace=True, ascending=True)
aux

# +
#Entrenamos modelo
#Algoritmo
Gaussian NB
model = GaussianNB()
model.fit(x_train, y_train)


# load JS visualization code to notebook
shap.initjs()

# explain all the predictions in the test set
explainer = shap.KernelExplainer(model.predict_proba, x_train)
shap_values = explainer.shap_values(x_train)
shap.force_plot(explainer.expected_value[0], shap_values[0], x_train)
# -

important_features= pd.DataFrame(data=np.transpose(model.fit(x_train, y_train).predict_proba(x_train)))
important_features

# plot the SHAP values for the Setosa output of all instances
shap.force_plot(explainer.expected_value[0], shap_values[0], x_train, link="logit")

algor_to_save = joblib.dump(explainer,res_path + "explainer_base_clasification.save")

algor_to_save = joblib.dump(explainer,res_path + "explainer_base_clasification.save")
algor_to_save2= joblib.dump(shap_values,res_path + "shap_values_base_clasification.save")

shap.summary_plot(shap_values, x_train, plot_type="bar")

shap.summary_plot(shap_values[1], x_train)

shap.dependence_plot("masa_grasa", shap_values[0], x_train)

shap_values[0].shape

x_train.shape


