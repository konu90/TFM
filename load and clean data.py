# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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
import os
import pandas as pd
import re
import numpy as np

#Library to work with dates
from datetime import datetime
#Library to operate(subtract and sum) with dates
from datetime import timedelta


# +
######################CARGA DE DATOS DE ACELEROMETRÍA#####################

#Metodo para parsear el campo Subject
def parseSubject(subject):
    aux = str(subject)
    r = aux.replace('T','').replace('.','').lstrip('A').replace('A','').replace('B','')
    return r

#Parse letterSubject' of Holter
def parseLetterSubject(subject):
    aux = str(subject).upper()
    tam = len(aux)-1
    if (aux[tam] == 'A') or (aux[tam] == 'B') or (aux[tam] == 'C') or (aux[tam] == 'D') or (aux[tam] == 'E') or (aux[tam] == 'S'):
        aux = aux[:-1]
    return aux

#Metodo que carga un fichero y efectua algunas operaciones basicas sobre las variables y elimina variables no relevantes
def cleanDataEHealth(path):
    paciente = pd.read_excel(path, sheet_name="Hourly")
    ##########CLEAN COLUMN NAMES###################
    paciente.columns = paciente.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('.','')
    ##########DELETE USELESS COLUMNS###############
    pacienteClean = paciente[['subject', 
                              'filename',
                              'day_of_week_num', 
                              'date',
                              'hour',
                              'kcals',
                              'mets',
                              'total_activity_counts_of_freedson_1998_bouts_occurring_on_this_hour',
                              'total_time_of_freedson_1998_bouts_occurring_on_this_hour',
                              'number_of_sedentary_bouts_occurring_on_this_hour',
                              'total_time_of_sedentary_breaks_occuring_on_this_hour',
                              'sedentary',
                              'light',
                              'moderate',
                              'vigorous',
                              '%_in_sedentary',
                              '%_in_light',
                              '%_in_moderate',
                              '%_in_vigorous',
                              'total_mvpa',
                              '%_in_mvpa',
                              'steps_counts',
                              'steps_average_counts']].copy()
    ##### COLUMNA VISITA ######
    #Segun el Path desde donde se han cargado los ficheros, sabemos si es la visita 1 o visita 5
    if "PRE" in nameFile:
        visita = 1
    else:
        visita = 5
    pacienteClean['visita'] = visita
    
    ####### COLUMNA DIA DE LA PRUEBA (contador del dia de la prueba, dia 1, dia 2, dia 3...
    #Testeado que si el dia de la semana empieza en 5,6,7,1, unique me mantiene el orden en el que aparecen
    #Get array of unique elements
    arrayDias = pacienteClean['day_of_week_num'].unique()
    #con la lista de distinct, nos creamos una funcion que comprobara en que posicion del array se encuentra el dia
    # de manera que independientemente de cuando empieze, si la semana 5,6,7,1, el dia 5 estara en la 
    #primera posicion(dia de la prueba)
    #Method to return the first index of array with the element
    def getIndexFirstElementInArray(dia,arrayDias):
        r = np.where(dia == arrayDias)
        return r[0][0] + 1
    #Create a new column from the day of week and apply method to get index from unique array unique
    pacienteClean['dia_prueba'] = pacienteClean['day_of_week_num'].apply(lambda x: getIndexFirstElementInArray(x,arrayDias))
    ############Eliminamos registros anteriores o posteriores a la duracion del experimento######### 
    #Obtenemos la primera hora en la que el paciente usa el dispositivo y la ultima hora en la que lo ha llevado
    #Filter hours that no in wear time validation
    #Function to read date in a specific format
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')
    #parse dates in a specific format(dateparse function)
    wearTime = pd.read_excel(path, sheet_name="Wear Time Validation Details", parse_dates=['Wear Time Start','Wear Time End'], date_parser=dateparse)
    #Primera hora en la que el paciente comienza el experimento(se pone el dispositivo por primera vez)
    first = min(wearTime['Wear Time Start'])
    #Ultima hora en la que el paciente usa el dispositivo
    last = max(wearTime['Wear Time End'])
    #Timedelta aplica operacione sbre horas, le quitamos una hora para evitar la ultima toma parcial
    #Se anota una toma a las 12:00(correspondiente a la hora 12:00 a 13:00)
    #pero el paciente se quito el reloj a las 12:30 por lo que la muestra tomada es parcial y eliminada
    last = last - timedelta(hours=1)
    #Combinar las columnas date y hour en una unica columna, para filtrar filas anteriores y posteriores al wearTimeValidation
    #Concatenar columnas fecha y hora del dataframe y añadir nueva columna al dataframe
    string_date_rng = [str(x) for x in (pacienteClean['date'] + ' ' + pacienteClean['hour'])]
    #Convertimos a tipo datetime
    timestamp_date_rng = pd.to_datetime(string_date_rng, format='%d/%m/%Y %H:%M')
    #La añadimos al dataframe
    pacienteClean['fecha'] = timestamp_date_rng
    #Filtramos las fechas anteriores y posteriores a la hora de uso del dispositivo
    pacienteClean = pacienteClean[pacienteClean.fecha >= first]
    pacienteClean = pacienteClean[pacienteClean.fecha <= last]
    #Reseteamos Indices
    pacienteClean = pacienteClean.reset_index(drop=True)
    #Parseamos la columna Subject
    pacienteClean['subject'] = pacienteClean['subject'].apply(lambda x: parseSubject(x))
    pacienteClean['subject'] = pacienteClean['subject'].apply(lambda x: parseLetterSubject(x))
    pacienteClean['subject'] = pacienteClean['subject'].astype(int)
    #Seleccionamos las variables relevantes
    rPaciente = pacienteClean[['subject',
                          'visita',
                          'dia_prueba',
                          'date',
                          'hour',
                          'kcals',
                          'mets',
                          '%_in_sedentary',
                          '%_in_light',
                          '%_in_moderate',
                          '%_in_vigorous',
                          'steps_counts']].copy()
    return rPaciente  


# +
#Añadimos las carpetas donde se encuentran los archivos proporcionados
path = os.getcwd() + "/data/"
dirF1PRE = path + "acelerometria/1FASE/PRE/"
dirF1POST = path + "acelerometria/1FASE/POST/"
dirF2PRE = path + "acelerometria/2FASE/PRE/"
dirF2POST = path + "acelerometria/2FASE/POST/"

pathDir = []
pathDir.append(dirF1PRE)
pathDir.append(dirF1POST)
pathDir.append(dirF2PRE)
pathDir.append(dirF2POST)

allData = pd.DataFrame()
#Para cada directorio en pathDir
for dir in pathDir:
    #Obten lista de ficheros dentro
    entries = os.listdir(dir)
    #Para cada fichero
    for entry in entries:
        #Path del fichero
        nameFile = dir + entry
        print(nameFile)
        #procesamos y acumulamos los diferentes ficheros leidos
        allData = allData.append(cleanDataEHealth(nameFile))

#Save data
allData.to_csv(path + "acelerometria_loaded.csv", sep =';', index=False)

# -

allData


# +
#Load datos demograficos
path = os.getcwd() + "\\data\\"
pathDemografico = path + "\\acelerometria\\Datos Demograficos.xlsx"
#Leemos, eliminamos nulos y nos quedamos con las columnas que nos interesan
auxDemografico = pd.read_excel(pathDemografico)
print(df.columns)
#Si algun dato es nulo se elimina
auxDemografico = auxDemografico.dropna(axis=0)
#Eliminamos acentos
auxDemografico.rename(columns={'Código':'Codigo'}, inplace=True)

#Split dataframe por visita y limpiar para hacer append
#Demografico de visita 1
demografico1 = auxDemografico[['Codigo',
                               'Sexo',
                               'Edad',
                               'peso_1',
                               'IMC_1',
                               'masa grasa_1',
                               'masa magra_1',
                               'Producto']].copy()

#Estandarizamos el nombre de las columnas para el append
demografico1.rename(columns={'peso_1':'Peso'}, inplace=True)
demografico1.rename(columns={'IMC_1':'IMC'}, inplace=True)
demografico1.rename(columns={'masa grasa_1':'masa_grasa'}, inplace=True)
demografico1.rename(columns={'masa magra_1':'masa_magra'}, inplace=True)
demografico1['Visita'] = 1
#Demograficos de visita 5
demografico5 = auxDemografico[['Codigo',
                               'Sexo',
                               'Edad',
                               'peso_5',
                               'IMC_5',
                               'masagrasa_5',
                               'masamagra_5',
                               'Producto']].copy()
#Estandarizamos el nombre de las columnas para el append
demografico5.rename(columns={'peso_5':'Peso'}, inplace=True)
demografico5.rename(columns={'IMC_5':'IMC'}, inplace=True)
demografico5.rename(columns={'masagrasa_5':'masa_grasa'}, inplace=True)
demografico5.rename(columns={'masamagra_5':'masa_magra'}, inplace=True)

demografico5['Visita'] = 5

#Append
demografico = demografico1.append(demografico5)
demografico.to_csv(path + "datos_demograficos_loaded.csv", sep =';', index=False)
# -

demografico


# +
#################CARGA DE DATOS DE HOLTER####################
#Path
path = os.getcwd() + "/data/"
pathHolter = os.getcwd() + '/data/Holter1.xlsx'
#Read data of all visit
dfHolter = pd.read_excel(pathHolter, sheet_name='V1')
# dfHolter = dfHolter.append(pd.read_excel(pathHolter, sheet_name='V2'))
# dfHolter = dfHolter.append(pd.read_excel(pathHolter, sheet_name='V3'))
# dfHolter = dfHolter.append(pd.read_excel(pathHolter, sheet_name='V4'))
# dfHolter = dfHolter.append(pd.read_excel(pathHolter, sheet_name='V5'))

#Delete useless columns
del dfHolter['Tipo']
del dfHolter['Código']
del dfHolter['Diario de actividades']

#Normalize subject
#Parse subject of dataframes
def parseSubject(subject):
        aux = str(subject)
        r = aux.replace('T','').replace('.','').lstrip('A')
        return r
dfHolter['Paciente'] = dfHolter['Paciente'].apply(lambda x: parseSubject(x))

#Delete row with errors
dfHolter = dfHolter[dfHolter.Estado != 'EE']
del dfHolter['Estado']

#Parse letterSubject' of Holter
def parseLetterSubject(subject):
    aux = str(subject).upper()
    tam = len(aux)-1
    if (aux[tam] == 'A') or (aux[tam] == 'B') or (aux[tam] == 'C') or (aux[tam] == 'D') or (aux[tam] == 'E') or (aux[tam] == 'S'):
        aux = aux[:-1]
    return aux
dfHolter['Paciente'] = dfHolter['Paciente'].apply(lambda x: parseLetterSubject(x))

# #Filtramos por visita
dfHolter = dfHolter[(dfHolter['Visita'] == 1) | (dfHolter['Visita'] == 5)]
#Reseteamos indices
dfHolter = dfHolter.reset_index(drop=True)
#Save data
# dfHolter.to_csv(path + "holter_loaded.csv", sep=';', index=False)
dfHolter
# -

holter = pd.read_csv(path + "holter_loaded.csv", delimiter=';')
holter

# +
#############LOAD TDA DATA.xlsx#################
#Relative path
path = os.getcwd()
data_path = path + "/data/"
df = pd.read_excel(data_path + "HTA Data.xlsx")

#Interpolate specific columns with the last value. In this case we are interpolate the columns:
#'Subject', 'Product', 'Visit', 'Gender (1: Male, 2: Female)',
df['Subject'] = df['Subject'].interpolate(method='pad')
df['Visit'] = df['Visit'].interpolate(method='pad')

#Count nulls
print("Número de nulos por variable")
print(df.isna().sum())

# #Drops nulls
df = df.dropna(axis=0,subset=df.columns)

#Numero de pacientes unicos
print("Número de pacientes unicos")
print(len(df['Subject'].unique()))

#Parse columns names to lowercase
df.columns = df.columns.str.strip().str.lower()
#Save data
df.to_csv(data_path + "holter_acelerometria.csv", sep=';', index=False)

df

# +
####Union de TDA Data con datos demograficos
path = os.getcwd() + "/data/"
#Load TDA data cleaned
df_holter_acelerometria = pd.read_csv(path + "holter_acelerometria.csv", sep=';')
#Load datos demograficos
df_demografico = pd.read_csv(path + "datos_demograficos_loaded.csv", sep=';')

#left join
df = df_holter_acelerometria.merge(df_demografico, how='left', left_on=["subject", "visit"], right_on=["Codigo","Visita"])
#drop right column
df = df.drop(['Codigo',"Visita"], axis=1)
#Conteo de nulos por columnas
df_with_null = df[df.isna().any(axis=1)]
print("Cantidad de nulos por columnas")
print(df_with_null.isna().sum())
#Comprobacion de los pacientes a los que pertenece
print("Lista de pacientes con valores nulos")
print(df_with_null['subject'].unique())
#Parse columns names to lowercase
df.columns = df.columns.str.strip().str.lower()
#Drop nulls by row
df = df.dropna(axis=0,subset=df.columns)
df.to_csv(path + "hta_cleaned.csv", sep=';', index=False)
df
aux = df[df['visit']==1]
aux2 = aux[aux['subject']==5]
aux2




# +
#############LOAD TDA DATARevisado.xlsx#################
#Relative path
path = os.getcwd()
data_path = path + "/data/"
df = pd.read_excel(data_path + "HTA DataRevisado.xlsx")

#Interpolate specific columns with the last value. In this case we are interpolate the columns:
#'Subject', 'Product', 'Visit', 'Gender (1: Male, 2: Female)',
df['Subject'] = df['Subject'].interpolate(method='pad')
df['Visit'] = df['Visit'].interpolate(method='pad')
# print(df)
#Count nulls
print("Número de nulos por variable")
print(df.isna().sum())
# #Drops nulls
print(len(df))
df = df.dropna(axis=0,subset=df.columns)
print(len(df))
#Numero de pacientes unicos
print("Número de pacientes unicos")
print(len(df['Subject'].unique()))
#Parse columns names to lowercase
df.columns = df.columns.str.strip().str.lower()
#Save data
df.to_csv(data_path + "holter_acelerometria.csv", sep=';', index=False)


####Union de HTA DataRevised con datos demograficos
path = os.getcwd() + "/data/"
#Load HTA data cleaned
df_holter_acelerometria = pd.read_csv(path + "holter_acelerometria.csv", sep=';')
#Load datos demograficos
df_demografico = pd.read_csv(path + "datos_demograficos_loaded.csv", sep=';')

#left join
df = df_holter_acelerometria.merge(df_demografico, how='left', left_on=["subject", "visit"], right_on=["Codigo","Visita"])
#drop right column
df = df.drop(['Codigo',"Visita"], axis=1)
#Conteo de nulos por columnas
df_with_null = df[df.isna().any(axis=1)]
print("Cantidad de nulos por columnas")
print(df_with_null.isna().sum())
#Comprobacion de los pacientes a los que pertenece
print("Lista de pacientes con valores nulos")
print(df_with_null['subject'].unique())
#Parse columns names to lowercase
df.columns = df.columns.str.strip().str.lower()
#Drop nulls by row
df = df.dropna(axis=0,subset=df.columns)
#delete rows with kcal = 0 mets = 1
df = df[~((df['kcal'] == 0) & (df['met'] == 1)) ]

df.to_csv(path + "hta_revised_cleaned.csv", sep=';', index=False)

print(len(df))
df
# +
# #############LOAD TDA DATARevisado.xlsx#################
# #Relative path
# path = os.getcwd()
# data_path = path + "/data/"
# df = pd.read_excel(data_path + "HTA DataRevisado.xlsx")
# df
# -




