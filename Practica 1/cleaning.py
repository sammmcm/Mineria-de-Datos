import pandas as pd
import numpy as np
from pathlib import Path


# leemos el csv
# ! IMPORTANTE: primero hay que correr extraction.py para obtener los datos crudos y que este codigo funcione
df = pd.read_csv(Path("dataset/raw_StormEvents_details.csv"))

print("Tamaño actual del dataset: ", len(df))
print("\nColumnas del dataset: ", df.columns) # vemos las columnas del dataset

print("\nPara que sea más cómodo y para seguir el estilo snake_case, cambiaré el nombre de las columnas a minúsculas")
df.columns = df.columns.str.lower()
print(df.columns)

print("\nReduciré aún más el dataset a 100k de datos filtrandolo por 8-10 tipos de evento")
print("Vemos qué tipos de eventos hay")
print(df['event_type'].value_counts())

events = [
    'Tornado', 'Drought', 'Heavy Snow', 'Flood', 
    'Marine Thunderstorm Wind', 'Heat', 'Heavy Rain', 
    'Strong Wind', 'Excessive Heat', 'Extreme Cold/Wind Chill'
] # eventos de interés

df_filtered = df[df['event_type'].isin(events)]
print("\nTamaño del dataset después del filtrado:", len(df_filtered))

# ahora que tenemos aprox. 100k de datos y que definimos en los tipos de eventos en los que nos vamos a enfocar, vamos a elegir qué columnas nos sirven y cuales no
dropcolumns = [
    # estas columnas no las ocupamos ya que begin_date_time y end_date_time ya tiene todo lo que necesitamos
    'begin_yearmonth', 'begin_day', 'begin_time',
    'end_yearmonth', 'end_day', 'end_time',
    'year', 'month_name',

    # no es necesario, con el nombre del state basta
    'state_fips',

    # da información que no es necesaria a lo que me quiero enfocar, lo cual es sobre los patrones climáticos
    # nos quedamos con cz_name ya que será una referencia del lugar en donde ocurrió, en vez de solo mostrar números (begin_lat, begin_lon) 
    'cz_type', 'cz_fips', 'wfo', 'cz_timezone', 'source', 'data_source',

    # toda la columna está vacía
    'category',

    # no son necesarios por la misma razón que cz_type, cz_fips, etc.
    'tor_other_wfo', 'tor_other_cz_state',
    'tor_other_cz_fips', 'tor_other_cz_name',

    # da más información de dónde empezó el evento pero no es necesario ya que begin_lat y begin_lon son suficientes
    'begin_range', 'begin_azimuth', 'begin_location',
    'end_range', 'end_azimuth', 'end_location'
]

df_filtered = df_filtered.drop(columns=dropcolumns)
print("\nLe cambiaremos el nombre a cz_name a region_name para que quede más claro")
df_filtered = df_filtered.rename(columns={"cz_name": "region_name"})
print("Columnas finales: ", df_filtered.columns)


print("\nVemos si los tipos de datos del dataset estan bien casteados")
print(df_filtered.info())
""" 
! PD: las columnas como 
! 13  magnitude          19531 non-null   float64       
! 14  magnitude_type     19531 non-null   str           
! 15  flood_cause        13018 non-null   str          
! 16  tor_f_scale        8369 non-null    str           
! 18  tor_width          8369 non-null    float64       
! 19  begin_lat          40519 non-null   float64       
! 20  begin_lon          40519 non-null   float64       
! 21  end_lat            40519 non-null   float64       
! 22  end_lon            40519 non-null   float64   
! se mantuvieron a pesar de tener tantos valores nulos ya que más adelante se piensan usar para analisis más especificos 
"""  

print("\nEl episode_id está como float, hay que cambiarlo")
print("Vemos qué datos hay")
print(df_filtered['episode_id'].unique())
df_filtered['episode_id'] = df_filtered['episode_id'].astype(int)

print("\nEl begin_date_time está como str, queremos que sea datetime para que sea más manejable")
print("Vemos qué datos hay")
print(df_filtered['begin_date_time'].unique())
df_filtered['begin_date_time'] = pd.to_datetime(df_filtered['begin_date_time'], errors='coerce')

print("\nLo mismo que para el begin_date_time")
print("Vemos qué datos hay")
print(df_filtered['end_date_time'].unique())
df_filtered['end_date_time'] = pd.to_datetime(df_filtered['end_date_time'], errors='coerce')

print("\nDAMAGE_PROPERTY está como str, ya que los datos están como 10.00K, 10.00M, etc")
print("Verificamos el formato de la celda")
print(df_filtered['damage_property'].dropna().str.contains(r'^\d+\.\d{2}[KMB]$').value_counts())

print("\nGuardamos el sufijo, es decir si está en miles, millones")
df_filtered['damage_property_suffix'] = df_filtered['damage_property'].str.extract(r'\d+\.\d{2}([KMB])')
print(df_filtered[['damage_property', 'damage_property_suffix']][df_filtered['damage_property'] != '0.00K'].dropna())

# convertimos la columna a float
df_filtered['damage_property'] = df_filtered['damage_property'].str.extract(r'(\d+\.\d{2})[KMB]').astype(float)

print("\nChecamos qué sufijos hay para hacer nuestro diccionario para multiplicarlo a damage_property")
print(df_filtered['damage_property_suffix'].dropna().unique())

multiplicador = {'K': 1000, 'M': 1000000, 'B': 1000000000}
df_filtered['damage_property'] = df_filtered['damage_property'] * df_filtered['damage_property_suffix'].map(multiplicador)

print("\nVerificamos los cambios")
print(df_filtered['damage_property'][df_filtered['damage_property'] != 0.0].dropna())

print("\nLo mismo para la columna damage_crops")
print(df_filtered['damage_crops'].dropna().str.contains(r'^\d+\.\d{2}[KMB]$').value_counts())
df_filtered['damage_crops_suffix'] = df_filtered['damage_crops'].str.extract(r'\d+\.\d{2}([KMB])')
print(df_filtered[['damage_crops', 'damage_crops_suffix']][df_filtered['damage_crops'] != '0.00K'].dropna())

df_filtered['damage_crops'] = df_filtered['damage_crops'].str.extract(r'(\d+\.\d{2})[KMB]').astype(float)
print(df_filtered['damage_crops_suffix'].dropna().unique())

multiplicador = {'K': 1000, 'M': 1000000}
df_filtered['damage_crops'] = df_filtered['damage_crops'] * df_filtered['damage_crops_suffix'].map(multiplicador)
print(df_filtered['damage_crops'][df_filtered['damage_crops'] != 0.0].dropna())

# borramos las columnas que ya no necesitamos
df_filtered = df_filtered.drop(columns=['damage_property_suffix', 'damage_crops_suffix'])

print("\nVerificamos los tipos de datos de nuevo")
print(df_filtered.info())

print("\nChecamos que no haya typos en las columnas tipo str")
print(sorted(df_filtered['state'].unique())) # todo bien
print(sorted(df_filtered['magnitude_type'].dropna().unique())) # todo bien
print(sorted(df_filtered['flood_cause'].dropna().unique())) # todo bien
print(sorted(df_filtered['tor_f_scale'].dropna().unique())) # todo bien

print("\nVerificamos que no esté mal registrado las fechas del evento")
print(df_filtered[df_filtered['end_date_time'] < df_filtered['begin_date_time']]) # todo bien

print("\nGuardamos toda la limpieza en un csv")
path = Path("dataset") / "clean_StormEvents_details.csv"
df_filtered.to_csv(path, index=False)
print("Listo.")