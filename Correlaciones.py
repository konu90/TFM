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


def plot_pearson_heatmap(df, name):
    path_graficas = os.getcwd() + '/graficas/'
    #Significancia estadistica
    significancia = 0.05
    
    df_corr = pd.DataFrame() # Correlation matrix
    df_p = pd.DataFrame()  # Matrix of p-values
    df_with_significance = pd.DataFrame() #Matriz de correlacion con significancia
    #Create matrix of correlation and pvalues
    for x in df.columns:
        for y in df.columns:
            corr = stats.pearsonr(df[x], df[y])
            df_corr.loc[x,y] = corr[0]
            df_p.loc[x,y] = corr[1]
            df_with_significance.loc[x,y] = corr[0] if corr[1] < significancia else np.nan
    
    #Prepare mask to plot as matrix
    mask = np.zeros_like(df_with_significance)
    mask[np.triu_indices_from(mask)] = True
    #Size figure
    plt.figure(figsize=(15,15))
    #Plot heatmap
    g = sns.heatmap(df_with_significance, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
    #Set lim
    bottom, top = g.get_ylim()
    #Configure
    g.set_ylim(bottom + 0.5, top - 0.5)
    #Name
    plt.title(name, fontsize =20)
    plt.savefig(path_graficas + name + '.png')
    g.plot()
    
    return df_with_significance



# +
#Dado el coefiente de spearman y el pvalue obtenido, devuelve el coeficiente si pvalue tiene un nivel de confianza del 95%
#En caso contrario devuelve 0
def confianza_spearman(spearman, pvalue):
    significancia = 0.05
    if(pvalue < significancia):
        return spearman
    else:
        return np.nan

#Calculo de la matriz de confianza
def matriz_confianza_spearman(dataset):
    spearman, pvalue = spearmanr(dataset)
    resultado = []
    fila = []
    for i in range(len(dataset.columns)):
        fila = []
        for j in range(len(dataset.columns)):
            fila.append(confianza_spearman(spearman[i][j], pvalue[i][j]))
        resultado.append(fila)
    df = pd.DataFrame(resultado, columns=dataset.columns)
    df.set_index(df.columns, inplace=True)   
    return df

#Metodo a la que dado un dataframe te calcula spearman y lo plotea
def plot_spearman_heatmap(df,name):
    path_graficas = os.getcwd() + '/graficas/'

    #Construimos matriz
    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True
    aux = matriz_confianza_spearman(df)
    plt.figure(figsize=(15,15))
    g = sns.heatmap(aux, mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            annot=True);
    bottom, top = g.get_ylim()
    g.set_ylim(bottom + 0.5, top - 0.5)
    plt.title(name, fontsize =20)
    plt.savefig(path_graficas + name + '.png')
    g.plot()
    
    return aux


# +
#PEARSON Analisis de correlacion población completa con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_pearson_heatmap(aux,"Pearson - Todos los pacientes")
graf

# +
#Spearman Analisis de correlacion población completa 

#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)
#plot
graf = plot_spearman_heatmap(aux,"Spearman - Todos los pacientes")
graf

# +
#PEARSON Analisis de correlacion grupo placebo con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['producto'] == 1]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_pearson_heatmap(aux,"Pearson - Grupo Placebo")
graf

# +
#Spearman Analisis de correlacion grupo placebo con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['producto'] == 1]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_spearman_heatmap(aux,"Spearman - Grupo placebo")
graf

# +
#PEARSON Analisis de correlacion grupo experimental con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['producto'] == 2]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_pearson_heatmap(aux,"Pearson - Grupo Experimental")
graf

# +
#Spearman Analisis de correlacion grupo experimental con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['producto'] == 2]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_spearman_heatmap(aux,"Spearman - Grupo experimental")
graf

# +
######CALCULO DE LA DIFERENCIA DE MATRICES DE CORRELACION DE SPEARMAN########
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Data grupo placebo
aux_placebo = df.copy()
#Filter by placebo
aux_placebo = aux_placebo[aux_placebo['producto'] == 1]
#Drop categorical variables
aux_placebo = aux_placebo.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Data grupo experimental
aux_experimental = df.copy()
#Filter by experimental
aux_experimental = aux_experimental[aux_experimental['producto'] == 2]
#Drop categorical variables
aux_experimental = aux_experimental.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Matriz del grupo placebo
spearman_placebo = plot_spearman_heatmap(aux_placebo,"Spearman - Grupo experimental")

#Matriz del grupo experimental
spearman_experimental = plot_spearman_heatmap(aux_experimental,"Spearman - Grupo experimental")

#Diferencia
dif = spearman_placebo - spearman_experimental

#Plot heatmap with dif
#Construimos matriz
mask = np.zeros_like(dif.corr())
mask[np.triu_indices_from(mask)] = True
aux = matriz_confianza_spearman(df)
plt.figure(figsize=(15,15))
g = sns.heatmap(dif, mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Diferencia de correlaciones entre grupo placebo y experimental", fontsize =20)
plt.savefig(path_graficas + 'Diferencia Spearman entre placebo y experimental.png')

g.plot()


# +
#Spearman Analisis de correlacion para el sexo masculino con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['sexo'] == 1]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_spearman_heatmap(aux,"Spearman - Sexo Masculino")

# +
#Spearman Analisis de correlacion para el sexo masculino con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Get copy
aux = df.copy()
#Filter by placebo
aux = aux[aux['sexo'] == 2]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)

#plot
graf = plot_spearman_heatmap(aux,"Spearman - Sexo Femenino")

# +
######CALCULO DE LA DIFERENCIA DE MATRICES POR SEXO DE LA CORRELACION DE SPEARMAN########
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Data grupo placebo
aux_masculino = df.copy()
#Filter by placebo
aux_masculino = aux_masculino[aux_masculino['producto'] == 1]
#Drop categorical variables
aux_masculino = aux_masculino.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Data grupo experimental
aux_femenino = df.copy()
#Filter by experimental
aux_femenino = aux_femenino[aux_femenino['producto'] == 2]
#Drop categorical variables
aux_femenino = aux_femenino.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Matriz del grupo masculino
spearman_masculino = plot_spearman_heatmap(aux_masculino,"Spearman - Grupo masculino")

#Matriz del grupo femenino
spearman_femenino = plot_spearman_heatmap(aux_femenino,"Spearman - Grupo femenino")

#Diferencia
dif = spearman_masculino - spearman_femenino

#Plot heatmap with dif
#Construimos matriz
mask = np.zeros_like(dif.corr())
mask[np.triu_indices_from(mask)] = True
aux = matriz_confianza_spearman(df)
plt.figure(figsize=(15,15))
g = sns.heatmap(dif, mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Diferencia de correlaciones entre grupo masculino y femenino", fontsize =20)
plt.savefig(path_graficas + 'Diferencia Spearman entre masculino y femenino.png')

g.plot()

# +
#Spearman Analisis de correlacion para el dia con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Parseo a datetime
df['hour'] = df['hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))
#Establecemos las horas del dia
hora_inicio_dia = pd.to_datetime("8:00", format='%H:%M')
hora_fin_dia = pd.to_datetime("20:00", format='%H:%M')
#Get copy
aux = df.copy()
#Filter by dia
aux = aux[(aux['hour'] >= hora_inicio_dia) & (aux['hour'] <= hora_fin_dia)]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)
#plot
graf = plot_spearman_heatmap(aux,"Spearman - Durante el día")

# +
#Spearman Analisis de correlacion para la noche con un nivel de significancia < 0.05
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Parseo de la hora a formato datetime
df['hour'] = df['hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))
#Establecemos las horas del dia
hora_inicio_dia = pd.to_datetime("8:00", format='%H:%M')
hora_fin_dia = pd.to_datetime("20:00", format='%H:%M')
#Get copy
aux = df.copy()
#Filter by placebo #Los que no sean de dia, son de noche
aux = aux[~(aux['hour'] >= hora_inicio_dia) & (aux['hour'] <= hora_fin_dia)]
#Drop categorical variables
aux = aux.drop(['subject','hour','producto','sexo','visit'], axis=1)
#plot
graf = plot_spearman_heatmap(aux,"Spearman - Durante la noche")

# +
######CALCULO DE LA DIFERENCIA DE MATRICES ENTRE DIA Y NOCHE PARA LA CORRELACION DE SPEARMAN########
#Path
path = os.getcwd() + "/data/"
path_graficas = os.getcwd() + "/graficas/"
#Read data
df = pd.read_csv(path + "hta_revised_cleaned.csv", sep=';')
#Convert types
df['% sedentary'] = df['% sedentary'].astype(float)
df['% light'] = df['% light'].astype(float)
df['% moderate'] = df['% moderate'].astype(float)
df['%vigorous'] = df['%vigorous'].astype(float)

#Parseo de la hora a formato datetime
df['hour'] = df['hour'].apply(lambda x: pd.to_datetime(x, format='%H:%M'))

#Establecemos las horas del dia
hora_inicio_dia = pd.to_datetime("8:00", format='%H:%M')
hora_fin_dia = pd.to_datetime("20:00", format='%H:%M')

#Data durante el dia
aux_dia = df.copy()
#Filter durante el dia
aux_dia = aux_dia[(aux_dia['hour'] >= hora_inicio_dia) & (aux_dia['hour'] <= hora_fin_dia)]
#Drop categorical variables
aux_dia = aux_dia.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Data durante la noche
aux_noche = df.copy()
#Filter durante la noche
aux_noche = aux_noche[~(aux_noche['hour'] >= hora_inicio_dia) & (aux_noche['hour'] <= hora_fin_dia)]
#Drop categorical variables
aux_noche = aux_noche.drop(['subject','hour','producto','sexo','visit'], axis=1)

#Matriz de correlaciones durante el dia
spearman_dia = plot_spearman_heatmap(aux_dia,"Spearman - Dia")

#Matriz de correlaciones durante la noche
spearman_noche = plot_spearman_heatmap(aux_noche,"Spearman - Noche")

#Diferencia
dif = spearman_dia - spearman_noche

#Plot heatmap with dif
#Construimos matriz
mask = np.zeros_like(dif.corr())
mask[np.triu_indices_from(mask)] = True
aux = matriz_confianza_spearman(df)
plt.figure(figsize=(15,15))
g = sns.heatmap(dif, mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True);
bottom, top = g.get_ylim()
g.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Diferencia de correlaciones entre el día y la noche", fontsize =20)
plt.savefig(path_graficas + 'Diferencia Spearman entre dia y noche.png')

g.plot()
# -


