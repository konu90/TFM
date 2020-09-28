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
from scipy.stats import pearsonr
from scipy import stats
import seaborn as sns

# +
###########TOMA DE PRODUCTO Y REPRESENTACION TEMPORAL############
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)
df['hora_24'] = df['hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

#Variables de comparacion
inicioTratamiento = pd.to_timedelta("8:00:00")
auxMadrugada = datetime.strptime("0:00:00", "%H:%M:%S").time()
auxInicioTratamiento = datetime.strptime("8:00:00", "%H:%M:%S").time()

# Para cada tupla, comprobamos la hora y operamos
for i in range(0,len(df)):
    #Aux para comparar horas
    hActual = datetime.strptime(str(df.iloc[i]['hour']),"%H:%M").time()
    #Aux para operar con horas
    hAux = pd.to_timedelta(str(hActual))
    
    #Si la hora es entre las 00:00 y las 8:00 le sumamos 16 horas
    if((hActual >= auxMadrugada) & (hActual < auxInicioTratamiento) ):
        hora = pd.to_timedelta([hAux + timedelta(hours=16)]).astype('timedelta64[h]')
    #En caso contrario le restamos 8 horas a la hora actual
    else:
        hora = pd.to_timedelta([hAux - inicioTratamiento]).astype('timedelta64[h]')
    df.at[i,'horas_tratamiento'] = hora[0]

#Delete useless columns
df = df.drop(['hour','hora_24'], axis=1)
#Save
df.to_csv(path + "data_with_features.csv", sep =';', index=False)

df


# +
############Categorización de IMC segun la OMS############
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "data_with_features.csv", sep=';')
#En función del imc devuelve un valor categorico segun la oms
# 1 insuficiencia ponderal
# 2 Intervalo normal
# 3 preobesidad
# 4 obesidad clase I
# 5 obesidad clase II
# 6 obesidad clase III
def imc_oms(imc):
#     aux = float(imc)
    if imc < 18.5:
        return "insuficiencia ponderal"
    elif (imc >= 18.5) & (imc < 25):
        return "intervalo normal"
    elif (imc >= 25) & (imc < 30):
        return "preobesidad"
    elif (imc >= 30) & (imc < 35):
        return "obesidad clase I"
    elif (imc >= 35) & (imc < 40):
        return "obesidad clase II"
    else:
        return "obesidad clase III"
#Aplicamos la funcion imc_oms sobre la columna imc
df['imc_cat'] = df['imc'].apply(lambda x: imc_oms(x))
#Save
df.to_csv(path + "data_with_features.csv", sep =';', index=False)
df



# +
#######TRENDS CLASIFICACION#################
#Vamos a calcular el diferencial de sistolico con la lectura anterior

#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "data_with_features.csv", sep=';')

#Creamos columna para los diferenciales, por defecto sera 0
#Se asume que la primera medida la diferencia es 0, ya que no se disponen de datos anteriores
df['systolic_diff'] = 0

aux_sys = 0
#Recorremos el dataframe ordenado por visita,paciente
for i in range(1,len(df)):
    #Guardamos el numero de paciente de la tupla anterior para detectar si cambia y debemos reiniciar
    auxpac = df.iloc[i-1]['subject']
    #Si el paciente de la tupla anterior es distinto al de la actual(ha cambiado el paciente)
    #guardamos el primer valor systolic de ese paciente y calculamos diferencial
    if (auxpac != df.iloc[i]['subject']):
        #Get first value
        aux_sys = df.iloc[i]['systolic']
        #Calculamos diferencial
        df.at[i,'systolic_diff'] = df.iloc[i]['systolic'] - aux_sys
    else:
        aux_diff = df.iloc[i-1]['systolic']
        #Calculamos diferencial
        df.at[i,'systolic_diff'] = df.iloc[i]['systolic'] - aux_diff

df.to_csv(path + "clasification_with_features.csv", sep =';', index=False)
df

# +
#######TRENDS REGRESION#################
#Vamos a calcular el estado medio del paciente

#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "data_with_features.csv", sep=';')

subjects = df['subject'].unique()
print(subjects)
trend_df = pd.DataFrame()
#Recorremos el dataframe ordenado por visita,paciente
for i in subjects:
    #Filtramos por paciente
    aux_df = df[df['subject'] == i].copy()
    #Calculamos la media para ese paciente
    mean = aux_df['systolic'].mean()
    #Asignamos nueva columna para dicho paciente
    aux_df['systolic_mean'] = mean
    #Unimos al resto de pacientes
    trend_df = trend_df.append(aux_df)
    
trend_df.to_csv(path + "regresion_with_features.csv", sep =';', index=False)
trend_df

# +
###########Lags CLASIFICACION############
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "clasification_with_features.csv", sep=';')

df
#Get lists to operate
subjects = df['subject'].unique()
columns = ['kcal', 'met', '% sedentary', '% light', '% moderate', '%vigorous', 'steps counts', 'systolic', 'diastolic', 'pam', 'pp', 'heart rate']
visits = df['visit'].unique()
#For each subject
df_rolling = pd.DataFrame()
for i in subjects:
    #Filter by subject
    aux = df[df['subject'] == i].copy()
    #for each visit calculate
    df_vis = pd.DataFrame()
    for j in visits:
        #filter by visit
        aux_vis = aux[aux['visit'] == j].copy()
        #Shift by each column in list
        for k in columns:
            aux_vis[k + "_shift"] = aux_vis[k].shift(1)
        #Add data by visit
        df_vis = df_vis.append(aux_vis)
    #Add data by subject
    df_rolling = df_rolling.append(df_vis)

#drop na
df_rolling = df_rolling.dropna(axis=0)
#Save    
df_rolling.to_csv(path + "clasification_data.csv", sep =';', index=False)
df_rolling.columns

# +
#################LAG REGRESION#################
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "regresion_with_features.csv", sep=';')

#Drop variables cardiovasculares
df = df.drop(['diastolic','pam','pp'], axis=1)

#Get lists to operate
subjects = df['subject'].unique()
columns = ['kcal', 'met', '% sedentary', '% light', '% moderate', '%vigorous', 'steps counts']
visits = df['visit'].unique()
#For each subject
df_rolling = pd.DataFrame()
for i in subjects:
    #Filter by subject
    aux = df[df['subject'] == i].copy()
    #for each visit calculate
    df_vis = pd.DataFrame()
    for j in visits:
        #filter by visit
        aux_vis = aux[aux['visit'] == j].copy()
        #Shift by each column in list
        for k in columns:
            aux_vis[k + "_shift"] = aux_vis[k].shift(1)
        #Add data by visit
        df_vis = df_vis.append(aux_vis)
    #Add data by subject
    df_rolling = df_rolling.append(df_vis)
    
#drop na
df_rolling = df_rolling.dropna(axis=0)
#Save    
df_rolling.to_csv(path + "regresion_data.csv", sep =';', index=False)
df_rolling
print(df_rolling.columns)

# +
#CLEAN USELESS COLUMNS IN CLASSIFICATION DATA
###########Lags CLASIFICACION############
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "clasification_data.csv", sep=';')

df.columns

# -

#CLEAN USELESS COLUMNS IN REGRESION DATA
###########Lags CLASIFICACION############
#Path
path = os.getcwd() + "/data/"
#Read data
df = pd.read_csv(path + "regresion_data.csv", sep=';')
df.columns


