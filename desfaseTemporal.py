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
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker

print(os.getcwd())

# +
# Selección de datos de paciente 5 únicamente para testeo
# y limpieza de datos para centrarnos en datos temporales
# los días se convierten en el índice
data_input_acel = pd.read_csv("data/input/Acelerometria Visitas 1 y 5.csv",sep=";")
data_input_holt = pd.read_csv("data/input/desfaseHolter.csv",sep=";")

data_clean_acel = data_input_acel[["date","hour","subject","visita"]]
data_clean_holt = data_input_holt[["DiaInicio","Hora24","Paciente","Visita"]]

# +
# Convertimos las dos columnas de tiempo a timestamp
auxhol = pd.to_datetime(data_clean_holt["Hora24"],format='%H:%M:%S')
auxace = pd.to_datetime(data_clean_acel["hour"],format='%H:%M')

# A continuación extraemos el tiempo
timeholt =list(map(lambda x: x.time(),auxhol))
timeacel =list(map(lambda x: x.time(),auxace))

# Después susttituímos las columnas de hora por las columnas time
data_clean_holt["Hora24"]=timeholt
data_clean_acel["hour"]=timeacel

# Repetimos la operación para date (se pasa a timestamp)
auxhol = pd.to_datetime(data_clean_holt["DiaInicio"],format='%d/%m/%Y')
auxace = pd.to_datetime(data_clean_acel["date"],format='%d/%m/%Y')

# A continuación extraemos la fechga
dateholt =list(map(lambda x: x.date(),auxhol))
dateacel =list(map(lambda x: x.date(),auxace))

# Sustituimos las columnas de fecha (en string) por las nuevas (en datetime.date)
data_clean_holt["DiaInicio"]=dateholt
data_clean_acel["date"]=dateacel

# +
# Unimos las dos datasets (Holter y Acelerometría), pero para eso
# necesitamos poner en común el nombre de la columnas de fecha y hora
# pacientes, y visita
data_clean_holt["Tipo"]="Holter"
data_clean_acel["Tipo"]="Acelerometría"
data_clean_acel.rename(columns={'DiaInicio':'date'}, inplace=True)
data_clean_holt.rename(columns={'DiaInicio':'date'}, inplace=True)
data_clean_acel.rename(columns={'hour':'Hora24'}, inplace=True)
data_clean_acel.rename(columns={'subject':'Paciente'}, inplace=True)
data_clean_holt.rename(columns={'Visita':'visita'}, inplace=True)

data_clean_holt.set_index("date")
data_clean_acel.set_index("date")

final=data_clean_holt.append(data_clean_acel)
final.to_csv("data/output/desfaseFinal.csv")

# +
# Celda de preparación y configuración de las gráficas
# Divisiones menores (cada 10 min)
minor_ticks = []
for i in range (0,24):
    for j in range (0,60,10):
        if(i < 10):
            hora = "0" + str(i)
        else:
            hora = str(i)
        if (j== 0):
            minutos = "00"
        else:
            minutos = str(j)
        minor_ticks.append(hora+":"+minutos)
        
# Divisiones mayores (cada 1 hora)
major_ticks=["00:00","1:00","2:00","3:00","4:00","5:00","6:00","7:00","8:00","9:00","10:00","11:00",
             "12:00","13:00","14:00","15:00","16:00","17:00","18:00","19:00","20:00","21:00","22:00","23:00"]
# -

# Extraemos el listado de todos los pacientes del archivo final
lista_pacientes = final["Paciente"].unique()
# Buscamos cada uno de los pacientes para dibujar la gráfica que toca
for pac in lista_pacientes:
    datos_pac = final[final["Paciente"]==pac]
    datos_pac.sort_values(by=["date"], ascending="True")
    sns.set(rc={'figure.figsize':(10, 20)})
    
    # Dibjuado de la gráfica, y configuración de la rejilla
    ax = sns.scatterplot(y="Hora24", x="date", hue="Tipo", style= "Tipo", data=datos_pac)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='both')                                                                                           
    ax.grid(which='minor', alpha=0.6)                                                
    ax.grid(which='major', alpha=1)
    ax.invert_yaxis()
    plt.xlabel("Fecha")
    plt.ylabel("Hora del día")
    for i in range(len(datos_pac)):
        ax.text(datos_pac.iloc[i]["date"], datos_pac.iloc[i]["Hora24"], datos_pac.iloc[i]["Hora24"], horizontalalignment='left', size='small')
    
    # Añadimos el título
    plt.title('Paciente '+str(pac), weight="bold", size="x-large")
    # Guardamos la imagen creada
    plt.savefig("data/output/Paciente"+str(pac)+".jpg")
    # Limpiamos el dibujo para que no se solape con el de después
    plt.clf()


# +
#################### A PARTIR DE AQUÍ, TODO SON PRUEBAS Y EXPERIMENTOS ########################3

facetest = final[final["Paciente"]==27]
facetest1 = facetest[facetest["visita"]==1]
facetest5 = facetest[facetest["visita"]==5]

# +
sns.set(rc={'figure.figsize':(10, 20)})
fig, axos =plt.subplots(1,2)

axos[0]=ax
# Dibjuado de la gráfica, y configuración de la rejilla
ax = sns.scatterplot(y="Hora24", x="date", hue="Tipo", style= "Tipo", data=facetest1)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)
ax.grid(which='both')                                                                                           
ax.grid(which='minor', alpha=0.6)                                                
ax.grid(which='major', alpha=1)
ax.invert_yaxis()
plt.xlabel("Fecha")
plt.ylabel("Hora del día")
for i in range(len(facetest1)):
    ax.text(facetest1.iloc[i]["date"], facetest1.iloc[i]["Hora24"], facetest1.iloc[i]["Hora24"], horizontalalignment='left', size='small')

# Añadimos el título
plt.title('Paciente '+str(pac), weight="bold", size="x-large")
# Guardamos la imagen creada
plt.savefig("data/output/Paciente"+str(pac)+".jpg")
# Limpiamos el dibujo para que no se solape con el de después


# Dibjuado de la gráfica, y configuración de la rejilla
axos[1]=ax1
ax1 = sns.scatterplot(y="Hora24", x="date", hue="Tipo", style= "Tipo", data=facetest5)
ax1.set_yticks(major_ticks)
ax1.set_yticks(minor_ticks, minor=True)
ax1.grid(which='both')                                                                                           
ax1.grid(which='minor', alpha=0.6)                                                
ax1.grid(which='major', alpha=1)
ax1.invert_yaxis()
plt.xlabel("Fecha")
plt.ylabel("Hora del día")
for i in range(len(facetest5)):
    ax1.text(facetest5.iloc[i]["date"], facetest5.iloc[i]["Hora24"], facetest5.iloc[i]["Hora24"], horizontalalignment='left', size='small')

# Añadimos el título
plt.title('Paciente '+str(pac), weight="bold", size="x-large")
# Guardamos la imagen creada
plt.savefig("data/output/Paciente"+str(pac)+".jpg")
# Limpiamos el dibujo para que no se solape con el de después

plt.show()
# -




