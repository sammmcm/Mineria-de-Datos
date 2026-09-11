import pandas as pd
import numpy as np

# usamos la formula de haversine par calcular la distancia recorrida de los eventos a millas
def haversine(lat1, lon1, lat2, lon2, r=3959): # r puede ser 6371 si quiere convertirse a km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * r * np.arcsin(np.sqrt(a))

df = pd.read_csv("dataset/clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

# para que la salida se vea bien y no se muestren tablas truncadas
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', None)

# vista estadística general de los datos
print(df.drop(columns=['episode_id', 'event_id']).describe())

# vemos el min, avg y max de lo que han durado los eventos
# esto se hizo por curiosidad, por si solo no es tan útil, pero pienso usarlo en la practica 5 de correlacion para ver si la duracion del evento causa mayor daño
print()
print((df['end_date_time'] - df['begin_date_time']).min())
print((df['end_date_time'] - df['begin_date_time']).mean())
print((df['end_date_time'] - df['begin_date_time']).max())

# vemos el min, avg y max de lo que han recorrido los eventos, tambien por curiosidad y se puede usar tambien en la practica 5
print()
df['distance_miles'] = haversine(df['begin_lat'], df['begin_lon'], df['end_lat'], df['end_lon'])
print(df['distance_miles'].min())
print(df['distance_miles'].mean())
print(df['distance_miles'].max())

# contamos cuantos eventos han pasado en cada estado
# esto sirve para ver qué estados son los que más sufren este tipo de eventos
print()
print(df.groupby("state")["event_type"].count().sort_values(ascending=False))

# agrupamos ahora por tipo de evento tambien, asi podemos ver cuales son los tipos de eventos que mas afectan en cada estado, no solo el conteo de este
# nos ayuda a saber si cada estado tiene un patron climatico
print()
print(df.groupby(["state", "event_type"])["event_type"].count().sort_values(ascending=False))

# al igual que podemos ver cuáles estados han sido los que mas han tenido daños de propiedad (mayor impacto económico)
print()
print(df.groupby("state")["damage_property"].sum().sort_values(ascending=False))

# ahora vemos la frecuencia de los eventos en cada año
# nos ayuda a ver si hay tendencia temporal
conteo = df.groupby(["event_type", df['begin_date_time'].dt.year]).size().reset_index(name='count')
print()
print(conteo)

# lo mismo que lo anterior, pero filtamos el año donde más presencia tuvo el evento
# nos ayuda a ver si el pico de cada evento coincide con otro, lo que nos puede indicar un patron climatico o algo más
conteo.columns = ['event_type', 'year', 'count']
idx = conteo.groupby('event_type')['count'].idxmax()
resultado = conteo.loc[idx]
print()
print(resultado)

# vemos cuantos heridos, muertes, daños de propiedad y daños de cultivos han hecho cada uno de los eventos
# importante para ver como afectan, porque puede que haya muchas apariciones en un estado, año pero no causar un daño realmente
print()
print(df.groupby("event_type")["injuries_direct"].sum().sort_values(ascending=False))
print(df.groupby("event_type")["deaths_direct"].sum().sort_values(ascending=False))
print(df.groupby("event_type")["damage_property"].sum().sort_values(ascending=False))
print(df.groupby("event_type")["damage_crops"].sum().sort_values(ascending=False))

# por último, vemos que tan anchos han sido los tornados, primero viendo su avg y después viendo el máximo, cada uno por año
# nos ayuda a ver como han evolucionado los tornados, si se han mantenido, si han sido mas peligrosos, etc
print()
print(df.groupby(df['begin_date_time'].dt.year)["tor_width"].mean().sort_values(ascending=False))
print(df.groupby(df['begin_date_time'].dt.year)["tor_width"].max().sort_values(ascending=False)) 