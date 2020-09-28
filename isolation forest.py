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
#Standard Librarys
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
#Library to work with dates
from datetime import datetime
#Library to operate(subtract and sum) with dates
from datetime import timedelta
from scipy.stats import spearmanr
import seaborn as sns

#Library for Isolation Forest
from sklearn.ensemble import IsolationForest

# +
########CLASIFICACION#######
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

#Aplicamos el escalador sobre datos continuos
data = df.copy()
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
data[df_ohe.columns] = df_ohe

data
#Apply Isolation forest
clf = IsolationForest(n_estimators=10000,max_samples=len(data))
#Train
clf.fit(data)
#Save labels
y_pred_train = pd.DataFrame()
y_pred_train['y'] = clf.predict(data)
#Get stadistics from isolation forest
print("Cantidad de tuplas:", len(data))
print("Cantidad de tuplas con el valor 1: ",len(y_pred_train[y_pred_train['y'] == 1]))
print("Cantidad de tuplas con el valor -1: ",len(y_pred_train[y_pred_train['y'] == -1]))
print("Porcentaje de outliers encontrados:", len(y_pred_train[y_pred_train['y'] == -1])/len(y_pred_train))

#how rows are outlies?
df['outliers'] = y_pred_train['y']
df['subject'] = column_group

aux = df[['subject','outliers']].copy()
aux = aux[aux['outliers'] == -1]

aux = aux.groupby(by="subject").count().sort_values(by='outliers', ascending = False)

#Save outliers labels
outlier = pd.DataFrame(y_pred_train['y'], columns='outliers')
outlier
# -

df[df['subject'] == 75.0]

df[df['subject'] == 9.0]

#Save outliers labels
outlier = pd.DataFrame()
outlier['outlier'] = y_pred_train['y']
outlier.to_csv(data_path + "outliers_clasificacion.csv", sep=';', index=False)

# +
####REGRESION GRUPO PLACEBO
print("ISOLATION FOREST PARA EL GRUPO PLACEBO")
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
# print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)

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
columns_to_drop = ['producto', 'subject','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

data = df.copy()
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
#Add columns
data[df_ohe.columns] = df_ohe

data

#Apply Isolation forest
clf = IsolationForest(n_estimators=10000,max_samples=len(data), contamination=0.05)
#Train
clf.fit(data)
#Save labels
y_pred_train = pd.DataFrame()
y_pred_train['y'] = clf.predict(data)
#Get stadistics from isolation forest
print("Cantidad de tuplas:", len(data))
print("Cantidad de tuplas con el valor 1: ",len(y_pred_train[y_pred_train['y'] == 1]))
print("Cantidad de tuplas con el valor -1: ",len(y_pred_train[y_pred_train['y'] == -1]))
print("Porcentaje de outliers encontrados:", len(y_pred_train[y_pred_train['y'] == -1])/len(y_pred_train))

#how rows are outlies?
df['outliers'] = y_pred_train['y']
df['subject'] = column_group

aux = df[['subject','outliers']].copy()
aux = aux[aux['outliers'] == -1]

aux = aux.groupby(by="subject").count().sort_values(by='outliers', ascending = False)

#Save outliers labels
outlier = pd.DataFrame()
outlier['outlier'] = y_pred_train['y']
outlier.to_csv(data_path + "outliers_placebo.csv", sep=';', index=False)
# -

aux

df[df['masa_grasa'] >= 50]

# +
####REGRESION GRUPO EXPERIMENTAL
print("ISOLATION FOREST PARA EL GRUPO EXPERIMENTAL")
#Resultado base
path = os.getcwd()
data_path = path + "/data/"
res_path = path + "/resultados/"
#Read data
df = pd.read_csv(data_path + "hta_revised_cleaned.csv", sep=';')
# print(df.columns)
#Clean useless columns
column_to_drop = ['diastolic', 'pam', 'pp', 'hour']
df.drop(column_to_drop,axis=1, inplace=True)

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
columns_to_drop = ['producto', 'subject','visit','sexo']
df.drop(columns_to_drop, axis=1, inplace=True)

data = df.copy()
#Categorical data
df_ohe = pd.DataFrame()
for i in categorical_data.columns:
            one_hot = pd.get_dummies(categorical_data[i], drop_first=True)
            df_ohe[one_hot.columns] = one_hot
#Add columns
data[df_ohe.columns] = df_ohe

data

#Apply Isolation forest
clf = IsolationForest(n_estimators=10000,max_samples=len(data),contamination=0.05)
#Train
clf.fit(data)
#Save labels
y_pred_train = pd.DataFrame()
y_pred_train['y'] = clf.predict(data)
#Get stadistics from isolation forest
print("Cantidad de tuplas:", len(data))
print("Cantidad de tuplas con el valor 1: ",len(y_pred_train[y_pred_train['y'] == 1]))
print("Cantidad de tuplas con el valor -1: ",len(y_pred_train[y_pred_train['y'] == -1]))
print("Porcentaje de outliers encontrados:", len(y_pred_train[y_pred_train['y'] == -1])/len(y_pred_train))

#how rows are outlies?
df['outliers'] = y_pred_train['y']
df['subject'] = column_group

aux = df[['subject','outliers']].copy()
aux = aux[aux['outliers'] == -1]

aux = aux.groupby(by="subject").count().sort_values(by='outliers', ascending = False)

#Save outliers labels
outlier = pd.DataFrame()
outlier['outlier'] = y_pred_train['y']
outlier.to_csv(data_path + "outliers_experimental.csv", sep=';', index=False)
# -

aux

df[['kcal', 'MET', '% Sedentary', '% Light', '% Moderate', '%Vigorous', '% in MVPA', 'Steps Counts', 'Systolic', 'Diastolic', 'PAM', 'PP', 'Heart Rate']].describe()

df.describe()

df.groupby('Product').count()
df.describe()


df.groupby('Visit').count()

df.groupby('Gender (1: Male, 2: Female)').count()

# #Buba y su DDA (Deep Data Analysis)
# #Get mild outlier and extreme outlier by columns
# #Get quartiles Q1, Q2 and Q3 values of all column for each cluster
df_quantiles = df.quantile([0.25, 0.5, 0.75])
df_quantiles

# #Calculate IQR of all column for each cluster
df_iqr = df_quantiles.loc[0.75] - df_quantiles.loc[0.25]
df_iqr

# #point in the interval [Q1−1.5 ∗ I QR, Q3+1.5 ∗ I QR] is mild outlier
# #Calculate mild outlier
df_mild_outlier_bot = (df_quantiles.loc[0.25] - df_iqr.loc[:] * 1.5)
df_mild_outlier_top = (df_quantiles.loc[0.75] + df_iqr.loc[:] * 1.5)
# #point outside of interval [Q1 − 3 ∗ I QR, Q3 + 3 ∗ I QR] is extreme outlier
# #Calculate extreme outlier
df_extreme_outlier_bot = (df_quantiles.loc[0.25] - df_iqr.loc[:] * 3)
df_extreme_outlier_top = (df_quantiles.loc[0.75] + df_iqr.loc[:] * 3)

df_mild_outlier_bot

df_mild_outlier_top

df_extreme_outlier_bot

df_extreme_outlier_top

df_mild_outlier_top

df_extreme_outlier_top

# +
# #Check the graphic again
# #Valores para centrar la recta en 0
y = [0]*len(df)
# #Grafica de sistolico
fig, ax = plt.subplots()
ax.scatter(df['Systolic'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Systolic', fontsize=13)
plt.title('Sistolico')
plt.show()
# #Grafica de sistolico por abajo
fig, ax = plt.subplots()
ax.scatter(df['Systolic'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Systolic', fontsize=13)
plt.title('Sistolico 70-90')
plt.xlim(70,90)
plt.show()

# print("Grafica de sistolico en el rango 70-90")
fig, ax = plt.subplots()
ax.scatter(df['Systolic'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Systolic', fontsize=13)
plt.title('Sistolico 160-200')
plt.xlim(160,200)
plt.show()

# #Histograma sin representación de KDE(kerner density estimate)
sb.distplot(df['Systolic'], bins=200, kde=False, color="g").set_title("Histograma de sistólico")
plt.show()
#Histograma con KDE
sb.distplot(df['Systolic'], bins=200,  color="g").set_title('Histograma de sistólico con kde')
plt.show()


# +
# #Check the graphic again
# #Valores para centrar la recta en 0
y = [0]*len(df)
# #Grafica de sistolico
fig, ax = plt.subplots()
ax.scatter(df['Heart Rate'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Heart Rate', fontsize=13)
plt.title('Heart Rate')
plt.show()
# #Grafica de sistolico por abajo
fig, ax = plt.subplots()
ax.scatter(df['Heart Rate'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Heart Rate', fontsize=13)
plt.title('Heart Rate 20-40')
plt.xlim(35,50)
plt.show()

# print("Grafica de sistolico en el rango 70-90")
fig, ax = plt.subplots()
ax.scatter(df['Heart Rate'],y, s=1)
# plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('Heart Rate', fontsize=13)
plt.title('Heart Rate 160-200')
plt.xlim(140,190)
plt.show()

# #Histograma sin representación de KDE(kerner density estimate)
sb.distplot(df['Heart Rate'], bins=200, kde=False, color="g").set_title("Histograma de Heart Rate")
plt.show()
#Histograma con KDE
sb.distplot(df['Heart Rate'], bins=200,  color="g").set_title('Histograma de Heart Rate con KDE')
plt.show()
# -

# #queda fijar el threshold y decir cuantas tuplas perderiamos
subject = df.groupby('Subject').mean()
subject = subject.dropna(axis=0)
subject


sb.distplot(subject['Weight (kg)'], bins=200, kde=False, color="g").set_title("Histograma de Weight (kg)")
plt.show()

subject[subject['Weight (kg)'] > 110]
len(df['Subject'].unique())

df.isnull().any(axis=1)

# +
null_columns=df.columns[df.isnull().any()]

df[null_columns].isnull().sum()
# -

df[df['Weight (kg)'].isnull()]

df['total_exercise'] = df['% Sedentary'] + df['% Light'] + df['% Moderate'] + df['%Vigorous']


df.columns

test = df[df['total_exercise'] <= 0.90]
columns = ['Subject', 'kcal', 'MET', 'total_exercise']
test[columns]

test['MET']

test2 = df[(df['kcal'] == 0) & (df['MET'] == 1) ]
test2

#Detectar tuplas anomalas
df = pd.read_excel(data_path + "HTA Data.xlsx")
df = df.drop(['Unnamed: 23', 'Unnamed: 24','Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28'], axis=1)
df['total_exercise'] = df['% Sedentary'] + df['% Light'] + df['% Moderate'] + df['%Vigorous']
df['Subject'] = df['Subject'].interpolate(method='pad')
test = df[df['total_exercise'] <= 0.9]
columns = ['Subject', 'kcal', 'MET', 'total_exercise']
test2 = test[columns]
test2.to_excel(data_path+ "feedback.xlsx", index=True)

# +

df =  pd.read_csv(data_path + "bioimpedanciaNAN.csv", sep=';')

#Nulos por columnas
null_columns=df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

#dropna
print("Numero de columnas antes de borrar nulos:", len(df))
df = df.dropna(axis=0).reset_index(drop=True)
print("Numero de columnas despues de borrar nulos:", len(df))

#Añadimos hora visita
df['Hour24'] = df['Hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

inicioTratamiento = pd.to_timedelta("8:00:00")
auxMadrugada = datetime.strptime("0:00:00", "%H:%M:%S").time()
auxInicioTratamiento = datetime.strptime("8:00:00", "%H:%M:%S").time()

# Este código añade la hora de tratamiento, según paciente y visita
for i in range(0,len(df)):
    #Aux para comparar horas
    hActual = datetime.strptime(str(df.iloc[i]['Hour']),"%H:%M").time()
#     print(hActual)
    #Aux para operar con horas
    hAux = pd.to_timedelta(str(hActual))
    
    if((hActual >= auxMadrugada) & (hActual < auxInicioTratamiento) ):
        hora = pd.to_timedelta([hAux + timedelta(hours=16)]).astype('timedelta64[h]')
    else:
        hora = pd.to_timedelta([hAux - inicioTratamiento]).astype('timedelta64[h]')
    df.at[i,'horaVisita'] = hora[0]
    
#Eliminamos useless columns
df.drop(['% in MVPA','Hour', 'Hour24'], axis=1, inplace=True)
print(len(df))
df = df[df['Subject'] != 10.0]
print(len(df))
#Eliminar tuplas anomalas segun monteloeder
print("Eliminamos tuplas anomalas que no llegan a al menos un 90% de ejercicio total")
df['total_exercise'] = df['% Sedentary'] + df['% Light'] + df['% Moderate'] + df['%Vigorous']
# df['Subject'] = df['Subject'].interpolate(method='pad')
df = df[df['total_exercise'] >= 0.9]
df.drop(['total_exercise'], axis=1, inplace=True)
print(len(df))
print(df['Subject'].unique())
#save data
df.to_csv(data_path + "clasificacionFullData.csv", index=False, sep=';')
print(df.columns)
# -

df

# +
# #Detectar tuplas anomalas
# df = pd.read_excel(data_path + "HTA DataRevisado.xlsx")
# df = df.drop(['Unnamed: 23', 'Unnamed: 24','Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28'], axis=1)
# df['total_exercise'] = df['% Sedentary'] + df['% Light'] + df['% Moderate'] + df['%Vigorous']
# df['Subject'] = df['Subject'].interpolate(method='pad')
# test = df[df['total_exercise'] <= 0.90]
# columns = ['Subject', 'kcal', 'MET', 'total_exercise']
# test2 = test[columns]
# test2
# # test2.to_excel(data_path+ "feedback.xlsx", index=True)

# +
df =  pd.read_csv(data_path + "bioimpedanciaNAN.csv", sep=';')

#Nulos por columnas
null_columns=df.columns[df.isnull().any()]
print(df[null_columns].isnull().sum())

#Añadimos hora visita
df['Hour24'] = df['Hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

inicioTratamiento = pd.to_timedelta("8:00:00")
auxMadrugada = datetime.strptime("0:00:00", "%H:%M:%S").time()
auxInicioTratamiento = datetime.strptime("8:00:00", "%H:%M:%S").time()

# Este código añade la hora de tratamiento, según paciente y visita
for i in range(0,len(df)):
    #Aux para comparar horas
    hActual = datetime.strptime(str(df.iloc[i]['Hour']),"%H:%M").time()
#     print(hActual)
    #Aux para operar con horas
    hAux = pd.to_timedelta(str(hActual))
    
    if((hActual >= auxMadrugada) & (hActual < auxInicioTratamiento) ):
        hora = pd.to_timedelta([hAux + timedelta(hours=16)]).astype('timedelta64[h]')
    else:
        hora = pd.to_timedelta([hAux - inicioTratamiento]).astype('timedelta64[h]')
    df.at[i,'horaVisita'] = hora[0]
    
#Eliminamos useless columns
df.drop(['% in MVPA','Hour', 'Hour24'], axis=1, inplace=True)
print(len(df))
# df = df[df['Subject'] != 10.0]
print(len(df))
#save data
df.to_csv(data_path + "obj_secundarios.csv", index=False, sep=';')
print(len(df['Subject'].unique()))
# -

df['Subject'].unique()
df['Subject']
len(df['Subject'].unique())

# +
#Prepare csv to Regresion problem
df = pd.read_csv(data_path + "obj_secundarios.csv", sep=';')
# print(df.columns)
column_to_drop = ['Diastolic', 'PAM', 'PP']
df.drop(column_to_drop,axis=1, inplace=True)
placebo = df[df['Product'] == 1]
placebo.drop(['Product'], axis=1, inplace=True)
print(len(placebo))
print(placebo.columns)
producto = df[df['Product'] == 2]
producto.drop(['Product'], axis=1, inplace=True)
print(len(producto))
print(placebo.columns)
print(producto.columns)
placebo.to_csv(data_path + "placebo.csv", index=False, sep=";")
producto.to_csv(data_path + "producto.csv", index=False, sep=";")

len(df['Subject'].unique())
# -

producto.columns

placebo


# +
#Dado el coefiente de spearman y el pvalue obtenido, devuelve el coeficiente si pvalue tiene un nivel de confianza del 95%
#En caso contrario devuelve 0
def confianzaSpearman(spearman, pvalue):
    significancia = 0.05
    if(pvalue < significancia):
        return spearman
    else:
        return 0

def matrizSpearmanConfianza(dataset):
    spearman, pvalue = spearmanr(dataset)
#     print("Dataset de matriz")
#     print(dataset)
#     print("shape")
#     print(dataset.shape)
    resultado = []
    fila = []
    for i in range(len(dataset.columns)):
        fila = []
#         print(i)
        for j in range(len(dataset.columns)):
#             print(j)
            fila.append(confianzaSpearman(spearman[i][j], pvalue[i][j]))
        resultado.append(fila)
    df = pd.DataFrame(resultado, columns=dataset.columns)
    df.set_index(df.columns, inplace=True)   
    return df


    

    

# +
#Metodo a la que dado un dataframe te calcula spearman y lo plotea
def plot_heat_map(df,name):
#     print(df.shape)
#     print(df.columns)
    #Construimos matriz
    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True
    aux = matrizSpearmanConfianza(df)
    plt.figure(figsize=(15,15))
    g = sns.heatmap(aux, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(name, fontsize =20)
#     g.set_tittle(name)
#     ax.set_title('lalala')
    g.plot()
    
#Para plotear los diferentes permutaciones de los heat maps(los 4 mapas de calor y seleccionar las columnas correspondientes)
def heats_maps(df):
    #HM 1
    columns_drop_hm_1 = ['kcal','Weight (kg)','BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'Subject']
    df_hm_1 = df.drop(columns_drop_hm_1, axis=1)
    plot_heat_map(df_hm_1,"H & A (sin kcal y Bioimpedancia) - Todos los pacientes")
    
    #HM 2
    #Delete pacients
    df_hm_2 = df[df['Subject'] != 12.0]
    df_hm_2 = df_hm_2[df_hm_2['Subject'] != 35.0]
    #Delete columns
    columns_drop_hm_2 = ['kcal', 'Subject']
    df_hm_2 = df_hm_2.drop(columns_drop_hm_2, axis=1)
    plot_heat_map(df_hm_2, "Bioimpedancia con el resto de variables (sin kcal) - Sin paciente 12 y 35")
    
    #HM 3
    #Delete pacients
    df_hm_3 = df[df['Subject'] != 10.0]
    df_hm_3 = df_hm_3[df_hm_3['Subject'] != 12.0]
    df_hm_3 = df_hm_3[df_hm_3['Subject'] != 35.0]
    #Delete columns
    columns_drop_hm_3 = ['Subject', 'Visit', 'Gender (1: Male, 2: Female)', 'MET',
       '% Sedentary', '% Light', '% Moderate', '%Vigorous',
       'Steps Counts', 'Systolic', 'Heart Rate', 'horaVisita']
    df_hm_3 = df_hm_3.drop(columns_drop_hm_3,axis=1)
    plot_heat_map_by_column(df_hm_3,"kcal", "Kcal con bioimpedancia - Sin pacientes 10,12,35")
    #HM 4
#     df_hm4 = 
    columns_drop_hm_4 = ['Weight (kg)','BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'Subject']
    df_hm_4 = df.drop(columns_drop_hm_4,axis=1)
    plot_heat_map_by_column(df_hm_4, "kcal", "Kcal con resto de variables - Sin paciente 10")


# +
#Metodo al que dado un dataframe te calcula spearman y lo plotea
def plot_heat_map_by_column(df, column_name,title):
    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True
    aux = matrizSpearmanConfianza(df)
#     print(aux[[column_name]].sort_values(by=[column_name],ascending=False))

    plt.figure(figsize=(15,15))
    g = sns.heatmap(aux[[column_name]].sort_values(by=[column_name],ascending=False),
            vmin=-1,
            vmax=1,
            cmap='coolwarm',
            annot=True);
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(title, fontsize = 20)
    g.plot()
    
#$ enfoques de Monteloeder dado un dataframe
def heats_maps_by_column(df, column_name):
#     print("All columns")
#     print(df.columns)
    #HM 1
    columns_drop_hm1 = ['kcal','Weight (kg)','BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'Subject', 'Gender (1: Male, 2: Female)']
    df_hm_1 = df.drop(columns_drop_hm1, axis=1)
    plot_heat_map_by_column(df_hm_1,column_name,"H & A (sin kcal y Bioimpedancia) - Todos los pacientes")
    
    #HM 2
    #Delete pacients
    df_hm_2 = df[df['Subject'] != 12.0]
    df_hm_2 = df_hm_2[df_hm_2['Subject'] != 35.0]
    #Delete columns
    columns_drop_hm_2 = ['Subject', 'Visit', 'Gender (1: Male, 2: Female)', 'MET',
       '% Sedentary', '% Light', '% Moderate', '%Vigorous', 'Steps Counts',
       'Systolic', 'Heart Rate', 'horaVisita', 'kcal']
    df_hm_2 = df_hm_2.drop(columns_drop_hm_2, axis=1)
    plot_heat_map_by_column(df_hm_2, column_name,"Bioimpedancia - Sin paciente 12 y 35")
    
    #HM 3
    #Delete pacients
    df_hm_3 = df[df['Subject'] != 10.0]
    #Delete columns
    columns_drop_hm_3 = ['Subject', 'Visit', 'Gender (1: Male, 2: Female)', 'Weight (kg)',
       'BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'MET',
       '% Sedentary', '% Light', '% Moderate', '%Vigorous',
       'Systolic', 'Heart Rate', 'horaVisita']
    df_hm_3 = df_hm_3.drop(columns_drop_hm_3,axis=1)
    plot_heat_map_by_column(df_hm_3, column_name,"Con kcal - Sin paciente 10")


# -

# Placebo - OS3 - 6.1.3	Correlacionar MVPA (actividad moderada-vigorosa) con frecuencia cardiaca  Diferenciar además por la variable demográfica sexo.
df = pd.read_csv(data_path + "placebo.csv", delimiter=';')
df['MVPA'] = df['% Moderate'] + df['%Vigorous']
heats_maps_by_column(df,"MVPA")


data_path

#Placebo - OS2 - Como se correlacionan los datos de actividad (acelerómetro) con sistólico (Holter). 
#Como se correlacionan estos datos por la variable demográfica sexo
df = pd.read_csv(data_path + "placebo.csv", delimiter=';')
heats_maps(df)
print(len(df))


df_hombre = df[df['Gender (1: Male, 2: Female)'] == 1]
heats_maps(df_hombre)
print(len(df_hombre))

df_mujer = df[df['Gender (1: Male, 2: Female)'] == 2]
heats_maps(df_mujer)
print(len(df_mujer))

# Placebo - OS3 - 6.1.3	Correlacionar MVPA (actividad moderada-vigorosa) con frecuencia cardiaca en 
#momentos de reposo.  Diferenciar además por la variable demográfica sexo.
df = pd.read_csv(data_path + "placebo.csv", delimiter=';')
df['MVPA'] = df['% Moderate'] + df['%Vigorous']
# df = df[df['% Sedentary'] >= 0.50]
print(len(df))
heats_maps_by_column(df,"MVPA")


df_hombre = df[df['Gender (1: Male, 2: Female)']==1]
print(len(df_hombre))
heats_maps_by_column(df_hombre,"MVPA")

df_mujer = df[df['Gender (1: Male, 2: Female)']==2]
print(len(df_mujer))
heats_maps_by_column(df_mujer,"MVPA")

# +
#Placebo OS 4 - Correlacionar cambios de presión arterial durante la noche (al dormir) comparado con el día. 
#Diferenciar además por la variable demográfica sexo.
df = pd.read_csv(data_path + "placebo.csv", delimiter=';')

def getMomento(horaVisita):
#     timedelta = pd.to_timedelta(hora)
    momento = ""
    if((horaVisita >= 0) and (horaVisita <= 12)):
        momento = "dia"
    else:
        momento = "noche"
        
    return momento

df['momento'] = df['horaVisita'].apply(lambda x: getMomento(x))
len(df)
# print(df['Subject'].unique())
# 
# -

df_dia = df[df['momento'] == "dia"]
df_dia.drop(['momento'], axis=1, inplace=True)
print(len(df_dia))
heats_maps(df_dia)

# +
df_dia_hombre = df_dia[df_dia['Gender (1: Male, 2: Female)'] == 1]
df_dia_hombre.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_dia_hombre))
heats_maps(df_dia_hombre)


# -

df_dia_mujer = df_dia[df_dia['Gender (1: Male, 2: Female)'] == 2]
df_dia_mujer.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_dia_mujer))
heats_maps(df_dia_mujer)

df_noche = df[df['momento'] == "noche"]
df_noche.drop(['momento'], axis=1, inplace=True)
print(len(df_noche))
heats_maps(df_noche)

df_noche_hombre = df_noche[df_noche['Gender (1: Male, 2: Female)'] == 1]
df_noche_hombre.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_noche_hombre))
heats_maps(df_noche_hombre)


# +
df_noche_mujer = df_noche[df_noche['Gender (1: Male, 2: Female)'] == 2]
df_noche_mujer.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_noche_mujer))

heats_maps(df_noche_mujer)
# -

producto


#Experimental OS3 - Como se correlacionan los datos de actividad (acelerómetro) con sistólico (Holter) para el
#grupo experimental. Como es esta correlación diferenciando por la variable demográfica sexo.
df = pd.read_csv(data_path + "producto.csv", delimiter=';')
print(len(df))
heats_maps(df)


df_hombre = df[df["Gender (1: Male, 2: Female)"] == 1]
df_hombre.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_hombre))
heats_maps(df_hombre)

df_mujer = df[df["Gender (1: Male, 2: Female)"] == 2]
df_mujer.drop(['Gender (1: Male, 2: Female)'], axis=1)
print(len(df_mujer))
heats_maps(df_mujer)

#Experimental OS 4 - 6.2.4	Correlacionar MVPA (actividad moderada-vigorosa) con sistólico en momentos de reposo. 
#¿Es diferente el comportamiento discriminando por la variable demográfica sexo?
df = pd.read_csv(data_path + "producto.csv", delimiter=';')
df['MVPA'] = df['% Moderate'] + df['%Vigorous']
print(len(df))
heats_maps_by_column(df,"MVPA")

df_hombre = df[df['Gender (1: Male, 2: Female)']==1]
print(len(df_hombre))
heats_maps_by_column(df_hombre,"MVPA")

df_mujer = df[df['Gender (1: Male, 2: Female)']==2]
print(len(df_mujer))
heats_maps_by_column(df_mujer,"MVPA")

# +
#Experimental OS 5 - Correlacionar cambios de presión arterial durante la noche (al dormir) comparado con el día. 
#Diferenciar además por la variable demográfica sexo.
df = pd.read_csv(data_path + "producto.csv", delimiter=';')

def getMomento(horaVisita):
#     timedelta = pd.to_timedelta(hora)
    momento = ""
    if((horaVisita >= 0) and (horaVisita <= 12)):
        momento = "dia"
    else:
        momento = "noche"
        
    return momento

df['momento'] = df['horaVisita'].apply(lambda x: getMomento(x))
len(df)
# df
# -

df_dia = df[df['momento'] == "dia"]
df_dia.drop(['momento'], axis=1, inplace=True)
print(len(df_dia))
heats_maps(df_dia)

# +
df_dia_hombre = df_dia[df_dia['Gender (1: Male, 2: Female)'] == 1]
print(len(df_dia_hombre))
# df_dia_hombre.drop(['Gender (1: Male, 2: Female)'], axis=1, inplace=True)
heats_maps(df_dia_hombre)


# -

df_dia_mujer = df_dia[df_dia['Gender (1: Male, 2: Female)'] == 2]
print(len(df_dia_mujer))
# df_dia_mujer.drop(['Gender (1: Male, 2: Female)'], axis=1)
heats_maps(df_dia_mujer)

df_noche = df[df['momento'] == "noche"]
df_noche.drop(['momento'], axis=1, inplace=True)
print(len(df_noche))
heats_maps(df_noche)

# +
df_noche_hombre = df_noche[df_noche['Gender (1: Male, 2: Female)'] == 1]
print(len(df_noche_hombre))
heats_maps(df_noche_hombre)


# -

df_noche_mujer = df_noche[df_noche['Gender (1: Male, 2: Female)'] == 2]
print(len(df_noche_mujer))
heats_maps(df_noche_mujer)

# +
#OS - Diferencia durante el dia y noche entre placebo y experimental
#Por simplicidad, se van a eliminar los pacientes 10, 12 y 35


#Metodo que añade la etiqueta del momento del dia
def getMomento(horaVisita):
#     timedelta = pd.to_timedelta(hora)
    momento = ""
    if((horaVisita >= 0) and (horaVisita <= 12)):
        momento = "dia"
    else:
        momento = "noche"
        
    return momento

#GRUPO PLACEBO
#Leemos datos
df_placebo = pd.read_csv(data_path + "placebo.csv", delimiter = ';')
#Eliminamos pacientes
df_placebo = df_placebo[df_placebo['Subject'] != 10.0]
df_placebo = df_placebo[df_placebo['Subject'] != 12.0]
df_placebo = df_placebo[df_placebo['Subject'] != 35.0]
df_placebo.drop(['Subject', 'Visit','Gender (1: Male, 2: Female)' ], axis=1, inplace=True)
#Añadimos etiqueta momento del dia
df_placebo['momento'] = df_placebo['horaVisita'].apply(lambda x: getMomento(x))
#Filtramos por dia
df_placebo_dia = df_placebo[df_placebo['momento']=='dia']
df_placebo_dia.drop(['momento'], axis=1, inplace=True)
#filtramos por noche
df_placebo_noche = df_placebo[df_placebo['momento']=="noche"]
df_placebo_noche.drop(['momento'], axis=1, inplace=True)


#GRUPO PRODUCTO
#Leemos datos
df_producto = pd.read_csv(data_path + "producto.csv", delimiter = ';')
#Eliminamos pacientes
df_producto = df_producto[df_producto['Subject'] != 10.0]
df_producto = df_producto[df_producto['Subject'] != 12.0]
df_producto = df_producto[df_producto['Subject'] != 35.0]
df_producto.drop(['Subject','Visit','Gender (1: Male, 2: Female)'], axis=1, inplace=True)
#Añadimos etiqueta mometno del dia
df_producto['momento'] = df_producto['horaVisita'].apply(lambda x: getMomento(x))
#Filtramos por dia
df_producto_dia = df_producto[df_producto['momento']=='dia']
df_producto_dia.drop(['momento'], axis=1, inplace=True)
#filtramos por noche
df_producto_noche = df_producto[df_producto['momento']=='noche']
df_producto_noche.drop(['momento'], axis=1, inplace=True)

# calculamos correlaciones
df_corr_placebo_dia = matrizSpearmanConfianza(df_placebo_dia)
df_corr_placebo_noche = matrizSpearmanConfianza(df_placebo_noche)
df_corr_producto_dia = matrizSpearmanConfianza(df_producto_dia)
df_corr_producto_noche = matrizSpearmanConfianza(df_producto_noche)

#Calculamos diferencia
df_corr_dia = df_corr_producto_dia - df_corr_placebo_dia
df_corr_noche = df_corr_producto_noche - df_corr_placebo_noche

#Heat_map de las correlaciones durante el dia
mask = np.zeros_like(df_corr_dia.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,15))
g = sns.heatmap(df_corr_dia, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Diferencia entre las correlaciones (experimental - placebo) durante el dia", fontsize =20)

g.plot()
#ehat_map de las correlaciones durante la noche
mask = np.zeros_like(df_corr_noche.corr())
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize=(15,15))
g = sns.heatmap(df_corr_noche, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Diferencia entre las correlaciones (experimental - placebo) durante la noche", fontsize =20)

g.plot()

# -

df_corr_dia

# +
#TEST

# ##Dado el coefiente de spearman y el pvalue obtenido, devuelve el coeficiente si pvalue tiene un nivel de confianza del 95%
# #En caso contrario devuelve 0
# def confianzaSpearman(spearman, pvalue):
#     significancia = 0.05
#     if(pvalue < significancia):
#         return spearman
#     else:
#         return 0

# def matrizSpearmanConfianza(dataset):
#     spearman, pvalue = spearmanr(dataset)
#     print("Dataset de matriz")
#     print(dataset)
#     print("shape")
#     print(dataset.shape)
#     print("Spearman")
#     print(spearman)
#     print("Pvalue")
#     print(pvalue)
#     resultado = []
#     fila = []
#     for i in range(len(dataset.columns)):
#         fila = []
#         print(i)
#         for j in range(len(dataset.columns)):
#             print(j)
# #             print(spearman[i][j])
#             print(pvalue[i][j])
#             fila.append(confianzaSpearman(spearman[i][j], pvalue[i][j]))
#         resultado.append(fila)
#     df = pd.DataFrame(resultado, columns=dataset.columns)
#     df.set_index(df.columns, inplace=True)   
#     return df

# #Metodo al que dado un dataframe te calcula spearman y lo plotea
# def plot_heat_map_by_column(df, column_name,title):
#     mask = np.zeros_like(df.corr())
#     mask[np.triu_indices_from(mask)] = True
#     aux = matrizSpearmanConfianza(df)
# #     print(aux[[column_name]].sort_values(by=[column_name],ascending=False))

#     plt.figure(figsize=(15,15))
#     g = sns.heatmap(aux[[column_name]].sort_values(by=[column_name],ascending=False),
#             vmin=-1,
#             vmax=1,
#             cmap='coolwarm',
#             annot=True);
#     bottom, top = g.get_ylim()
#     g.set_ylim(bottom + 0.5, top - 0.5)
#     plt.title(title, fontsize = 20)
#     g.plot()
    
# #$ enfoques de Monteloeder dado un dataframe
# def heats_maps_by_column(df, column_name):
# #     print("All columns")
# #     print(df.columns)
#     #HM 1
#     columns_drop_hm1 = ['kcal','Weight (kg)','BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'Subject']
#     df_hm_1 = df.drop(columns_drop_hm1, axis=1)
#     plot_heat_map_by_column(df_hm_1,column_name,"H & A (sin kcal y Bioimpedancia) - Todos los pacientes")
    
#     #HM 2
#     #Delete pacients
#     df_hm_2 = df[df['Subject'] != 12.0]
#     df_hm_2 = df_hm_2[df_hm_2['Subject'] != 35.0]
#     #Delete columns
#     columns_drop_hm_2 = ['Subject', 'Visit', 'Gender (1: Male, 2: Female)', 'MET',
#        '% Sedentary', '% Light', '% Moderate', '%Vigorous', 'Steps Counts',
#        'Systolic', 'Heart Rate', 'horaVisita', 'kcal']
#     df_hm_2 = df_hm_2.drop(columns_drop_hm_2, axis=1)
#     plot_heat_map_by_column(df_hm_2, column_name,"Bioimpedancia - Sin paciente 12 y 35")
    
#     #HM 3
#     #Delete pacients
#     df_hm_3 = df[df['Subject'] != 10.0]
#     #Delete columns
#     columns_drop_hm_3 = ['Subject', 'Visit', 'Gender (1: Male, 2: Female)', 'Weight (kg)',
#        'BMI ', 'Fat Mass (kg)', 'Lean Mass (kg)', 'Age', 'MET',
#        '% Sedentary', '% Light', '% Moderate', '%Vigorous',
#        'Systolic', 'Heart Rate', 'horaVisita']
#     df_hm_3 = df_hm_3.drop(columns_drop_hm_3,axis=1)
#     print(df_hm_3)
#     plot_heat_map_by_column(df_hm_3, column_name,"Con kcal - Sin paciente 10")

# # Placebo - OS3 - 6.1.3	Correlacionar MVPA (actividad moderada-vigorosa) con frecuencia cardiaca en 
# #momentos de reposo.  Diferenciar además por la variable demográfica sexo.
# df = pd.read_csv(data_path + "placebo.csv", delimiter=';')
# df['MVPA'] = df['% Moderate'] + df['%Vigorous']
# heats_maps_by_column(df,"MVPA")




# +
#Objetivo principal 2
df =  pd.read_csv(data_path + "clasificacionFullData.csv", sep=';')
print(len(df))
print(df['Subject'].unique())

# calculamos correlaciones
df_corr = matrizSpearmanConfianza(df)

#ehat_map de las correlaciones durante la noche
mask = np.zeros_like(df.corr())
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(15,15))
g = sns.heatmap(df_corr, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Correlacion entre los parametros de los dispositivos de H & A", fontsize =20)

g.plot()

plot_heat_map_by_column(df_corr,"Systolic", "Correlacion de sistolico con los parametros de los dispositivos de H & A")
print(len(df))
# -








