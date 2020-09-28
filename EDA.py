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
import seaborn as sns
#Library to work with dates
from datetime import datetime
#Library to operate(subtract and sum) with dates
from datetime import timedelta

#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"

# +
##########ESTADISTICAS POR COLUMNAS##############
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Columnas numericas
numerical_columns = ['kcal', 'met', '% sedentary', '% light', '% moderate', '%vigorous', 'steps counts', 'systolic', 'diastolic', 'pam', 'pp', 'heart rate', 'edad','peso','imc','masa_grasa','masa_magra']
#Columnas categoricas
categorical_columns = ['subject','visit', 'sexo', 'producto']

# #Datos por paciente
aux = df[['subject','visit']].copy()
aux = aux.rename(columns={"visit": "count"})
aux = aux.groupby('subject').count()


# -

#Plot el conteo de datos por paciente
plt.figure(figsize=(20,5))
ax = sns.countplot(x="subject", data=df)
ax.set(ylim=(0, 50))
ax.set_title('Datos por paciente')


# +
#Plot el conteo de datos por paciente y visita
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Visita 1
#Filter
aux = df[df['visit']==1]
#Count data
print(len(aux))
#Plot
plt.figure(figsize=(20,5))
ax = sns.countplot(x="subject", data=aux)
#Configure plot
ax.set(ylim=(0, 50))
ax.set_title('Datos por paciente en visita 1')

#Visita 5
#Filter
aux = df[df['visit']==5]
#count data
print(len(aux))
#Plot
plt.figure(figsize=(20,5))
ax = sns.countplot(x="subject", data=aux)
#Configure plot
ax.set(ylim=(0, 50))
ax.set_title('Datos por paciente en visita 5')

# +
#Distribución por sexo
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Cambiamos la etiqueta de sexo para hacer mas representativo los graficos
aux = df.copy()
aux['sexo_cat'] = aux['sexo'].apply(lambda x: "masculino" if x==1 else "femenino")
#Todos los datos
#Tamaño del ploteo
plt.figure(figsize=(10,5))
#Datos para plotear
ax = sns.countplot(x="sexo_cat",hue="sexo_cat",data=aux)
#Configuración
ax.set_title('Datos por sexo para todas las visitas')

#Filtro para la visita 1
aux_vis1 = aux[aux['visit']==1]
plt.figure(figsize=(5,2.5))
ax = sns.countplot(x="sexo_cat",hue="sexo_cat",data=aux_vis1)
#Configuración
ax.set_title('Datos por sexo para la visita 1')
#Filtro para la visita 5
aux_vis5 = aux[aux['visit']==5]
plt.figure(figsize=(5,2.5))
ax = sns.countplot(x="sexo_cat",hue="sexo_cat",data=aux_vis5)
#Configuración
ax.set_title('Datos por sexo para la visita 5')

# +
#Distribución por producto
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Cambiamos la etiqueta producto para hacer mas representativo los graficos
aux = df.copy()
aux['producto_cat'] = aux['producto'].apply(lambda x: "placebo" if x==1 else "experimental")
#Todos los datos
#Tamaño del ploteo
plt.figure(figsize=(10,5))
#Datos para plotear
ax = sns.countplot(x="producto_cat",hue="producto_cat",data=aux)
#Configuración
ax.set_title('Datos por producto para todas las visitas')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

#Filtro para la visita 1
aux_vis1 = aux[aux['visit']==1]
plt.figure(figsize=(5,2.5))
ax = sns.countplot(x="producto_cat",hue="producto_cat",data=aux_vis1)
#Configuración
ax.set_title('Datos por producto para la visita 1')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

#Filtro para la visita 5
aux_vis5 = aux[aux['visit']==5]
plt.figure(figsize=(5,2.5))
ax = sns.countplot(x="producto_cat",hue="producto_cat",data=aux_vis5)
#Configuración
ax.set_title('Datos por producto para la visita 5')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# +
#Variables categoricas
#Distribución por producto
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Cambiamos la etiqueta producto para hacer mas representativo los graficos
aux = df.copy()
aux['producto_cat'] = aux['producto'].apply(lambda x: "placebo" if x==1 else "experimental")
aux['sexo_cat'] = aux['sexo'].apply(lambda x: "hombre" if x==1 else "mujer")


#Filtramos por pacientes que toman placebo
placebo = aux[aux['producto'] == 0]

#Tamaño del ploteo
plt.figure(figsize=(10,5))
#Datos para plotear
ax = sns.countplot(x="sexo_cat",hue="producto_cat",data=aux)
#Configuración
ax.set_title("Placebo y experimental en función del sexo")
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)



# +
#Stadisticals by numerical columns
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

df[numerical_columns].describe()

# +
#Calculo de agregados para validación de datos
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Ejercicio total
df['total_exercise'] = df['% sedentary'] + df['% light'] + df['% moderate'] + df['%vigorous']
aux = df[df['total_exercise'] <= 0.99]
print(len(aux))
aux[['subject','visit', 'hour','% sedentary','% light','% moderate','%vigorous','total_exercise']]


# +
#Presion de pulso como diferencia entre sistolico y diastolico
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#PP = PresionSistolica - PresionDiastolica
df['pp_check'] = df['systolic'] - df['diastolic']
aux = df[df['pp_check'] != df['pp']]
aux[['subject','visit', 'hour','systolic','diastolic','pam','pp','pp_check']]


# +
#Ploteo por pares de las variables numericas con sexo
#read data
df = pd.read_csv(path + "hta_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

aux = df[numerical_columns].copy()
aux['sexo'] =  df['sexo'].copy()
aux['sexo'] = aux['sexo'].apply(lambda x: "masculino" if x==1 else "femenino")
graph = sns.pairplot(aux, hue="sexo", corner = True)
#add legend
graph = graph.add_legend()
#Save fig
plt.savefig(path_graficas + 'sexo.png')
#Show
plt.show()

# +
#La primera columna (eje X) representa la variable kcals en función de las diferentes variables (eje y), 
#y se destaca a priori que hay una serie de puntos (que se corresponden al sexo masculino) que sigue una 
#distribución diferente al resto de puntos, pudiendo ser producido por un outlier real (valor anómalo según
#la distribución del resto de puntos) o un fallo producido por los dispositivos.
#read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

aux = df[['subject', 'sexo','kcal']].copy()
aux = aux[aux['sexo'] == 1]
aux = aux.sort_values(by=['kcal'],ascending=False).reset_index(drop=True)

aux[0:30]


# +
#Referente a la columna vigorous (sexta columna), se destaca una serie de puntos reducida del sexo 
#femenino que realizan ejercicio vigoroso durante un tiempo mucho mayor que el resto de los puntos del dataset.
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

aux = df[['subject', 'sexo','%vigorous']].copy()
aux = aux[aux['sexo'] == 2]
aux = aux.sort_values(by=['%vigorous'],ascending=False).reset_index(drop=True)

aux[0:30]

# +
#Analisis de ejercicio y PP
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

aux = df[['subject','sexo','systolic','diastolic','pp', '% sedentary', '% light', '% moderate', '%vigorous']].copy()
aux = aux.sort_values(by=['pp'],ascending=False).reset_index(drop=True)

aux[0:30]

# +
#Ploteo por pares de las variables numericas con producto
aux = df[numerical_columns].copy()
aux['producto'] =  df['producto'].copy()
aux['producto'] = aux['producto'].apply(lambda x: "placebo" if x==1 else "experimental")

graph = sns.pairplot(aux, hue="producto", corner = True)
#add legend
graph = graph.add_legend()
#Save fig
plt.savefig(path_graficas + 'producto.png')
plt.show()

# +
#En cuanto a la variable vigorous (sexta columna), podemos ver que caen en su mayoría en pacientes que toman 
#el producto experimental, aunque también existe algún paciente que toma placebo con un registro superior a la 
#mayoría de los puntos
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

aux = df[['subject', 'producto','%vigorous']].copy()
aux = aux.sort_values(by=['%vigorous'],ascending=False).reset_index(drop=True)

aux[0:30]

# +
#Lectura de datos
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)
df['pp_check'] = df['systolic'] - df['diastolic']

aux = df.copy()
aux = aux.sort_values(by=['pp'],ascending=False).reset_index(drop=True)

aux = aux[aux['pp'] >= 70]
aux

# +
#Preparacion de anomalias encontradas
df = pd.read_csv(path + "hta_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)
df['total_exercise'] = df['% sedentary'] + df['% light'] + df['% moderate'] + df['%vigorous']
df['pp_check'] = df['systolic'] - df['diastolic']
print(len(df['subject'].unique()))

aux = df.copy()
#Anomalias de kcal, cuando kcal es 0 y met 1
aux_kcal = aux[(aux['kcal'] == 0) & (aux['met'] == 1) ]
#Anomalias de ejercicio, cuando el ejercicio total realizado en una hora es inferior a 0.99
aux_exercise = aux[aux['total_exercise'] <= 0.99]
#Anomalias relacionadas a PP, cuando la diferencia no coincide con el dato proporcionado
aux_pp = aux[aux['pp_check'] != aux['pp']]
#Anomalias de kcal elevadas del paciente 10
aux_kcal_alta = aux[(aux['kcal'] >= 500)]

#Anomalias en las presiones de pulso elevadas
aux_pp_elevada = aux.sort_values(by=['pp'],ascending=False).reset_index(drop=True)
aux_pp_elevada = aux[aux['pp'] >= 70]

to_revise = pd.DataFrame()
to_revise = to_revise.append(aux_kcal)
to_revise = to_revise.append(aux_exercise)
to_revise = to_revise.append(aux_pp)
to_revise = to_revise.append(aux_kcal_alta)
to_revise = to_revise.append(aux_pp_elevada)

to_revise = to_revise.sort_values(by=['subject','visit','hour'],ascending=True).reset_index(drop=True)
to_revise.to_csv(path + "hta_to_revise.csv", sep=';', index=False)
print(len(to_revise['subject'].unique()))
to_revise

# +
#Se preparan los datos finales para ML
#Preparacion de anomalias encontradas
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)
df['total_exercise'] = df['% sedentary'] + df['% light'] + df['% moderate'] + df['%vigorous']

aux = df.copy()
#Anomalias de ejercicio, cuando el ejercicio total realizado en una hora es inferior a 0.99
aux_exercise = aux[aux['total_exercise'] >= 0.90]
aux_exercise = aux_exercise.drop(['total_exercise'], axis=1)

aux_exercise = aux_exercise.sort_values(by=['subject','visit','hour'],ascending=True).reset_index(drop=True)
aux_exercise.to_csv(path + "data.csv", sep=';', index=False)
aux_exercise

# +
#Diagramas de violin en funcion de las variables categoricas sexo y producto
df = pd.read_csv(path + "hta_cleaned.csv", sep=';')
df['subject'] = df['subject'].astype(int)
df['sexo'] = df['sexo'].astype(int)
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Cambiamos la etiqueta de sexo para hacer mas representativo los graficos
aux = df.copy()
aux['sexo_cat'] = aux['sexo'].apply(lambda x: "masculino" if x==1 else "femenino")
aux['producto_cat'] = aux['producto'].apply(lambda x: "placebo" if x==1 else "producto")

catg_list = ['sexo_cat','producto_cat']

for catg in catg_list :
    #sns.catplot(x=catg, y=target, data=df_train, kind='boxen')
    sns.violinplot(x=catg, y='pp', data=aux)
    plt.savefig(path_graficas + 'pp_' + catg + '.png')
    plt.show()
    sns.violinplot(x=catg, y='systolic', data= aux)
    plt.savefig(path_graficas + 'systolic_' + catg + '.png')
    plt.show()
    sns.violinplot(x=catg, y='diastolic', data= aux)
    plt.savefig(path_graficas + 'diastolic_' + catg + '.png')
    plt.show()
    #sns.boxenplot(x=catg, y=target, data=df_train)
    #bp = df_train.boxplot(column=[target], by=catg)


# -




