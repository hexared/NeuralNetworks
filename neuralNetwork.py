import pandas as ps
import quandl as ql
import math
import numpy as np #libreria utile per introdurre array in python
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

#preprocessing: utile per la scalabilita dei dati
# cross_validation: utilizzata per esempi di training & testing e separare dati
# Support Vector Machine: utile per la regressione

df = ql.get('WIKI/GOOGL')

#listo tutte le colonne che mi interessano
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Percentuale di cambio giornaliera
df['HL_PCT']= (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
#percentuale di cambio generica
df['PCT_change']= (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
#Volume giornaliero
df= df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume' ]]

#inizializzazione previsione
forecast_col = 'Adj. Close'
#'fillna(val, inplace)' serve a fillare le posizioni con perdite di dati
df.fillna(-99999, inplace=True)

#output della predizione sul cambiamento di prezzo (circa il 10% del dataframe)
forecast_out = int(math.ceil(0.01*len(df)))

#shifto le colonne "negativamente" (verso l'alto)
#in modo da avere l'Adj. Close di 10 giorni nel futuro
df['Label'] = df[forecast_col].shift(-forecast_out)

#definisco x ed y, le features vengono identificate con lettere maiuscole,
#le labels invece sono lettere minuscole

#ritorna un nuovo dataframe convertito in un numpy-array
X = np.array(df.drop(['Label'], 1))
y = np.array(df['Label'])

#il valore di X viene "normalizzato" per essere incluso (insieme agli altri dati)
#tra i vaolri di training
X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['Label'])

print(len(X), len(y))
