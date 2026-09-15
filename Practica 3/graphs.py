import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 3" / "graphs"
graficas_dir.mkdir(exist_ok=True)

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_csv("dataset/clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

histograma_config = [
    {"data": df[df["event_type"] == 'Tornado'], "x": 'tor_width', "xlim": (0, 1500), "title": "Distribución del ancho de tornados", "xlabel": "Ancho del tornado (yd)", "ylabel": "Cantidad de tornados", "archivo": "hist_tornado.png"},
    {"data": df, "x": 'magnitude', "title": "Distribución de la magnitud del viento", "xlabel": "Magnitud", "ylabel": "Cantidad de eventos", "archivo": "hist_magnitud.png"}
]

for cfg in histograma_config:
    sns.histplot(data=cfg["data"], x=cfg["x"])
    if cfg.get("xlim"): 
        plt.xlim(cfg["xlim"])
    plt.title(cfg["title"])
    plt.xlabel(cfg["xlabel"])
    plt.ylabel(cfg["ylabel"])
    plt.tight_layout()
    plt.savefig(graficas_dir / cfg["archivo"])
    # plt.show()
    plt.close()

# primero hice unos histogramas solo para ver la tendencia del ancho del tornado y de la magnitud, de las cuales
# podemos ver que una esta sesgada a la izq y la otra esta mas o menos centralizada
# para el histograma del ancho del tornado le puse un limite ya que se hacia muy grande la grafica y no se apreciaba bien la tendencia

scatterplot_config = [
    {"data": df[df["event_type"] == 'Tornado'], "x": 'tor_width', "title": "Daño a propiedad según ancho del tornado", "xlabel": "Ancho del tornado (yd)", "archivo": "scatter_tornado.png"},
    {"data": df, "x": 'magnitude', "title": "Daño a propiedad según la magnitud del viento", "xlabel": "Magnitud", "archivo": "scatter_magnitud.png"}
]

for cfg in scatterplot_config:
    sns.scatterplot(data=cfg["data"], x=cfg["x"], y='damage_property')
    plt.yscale('log')
    plt.title(cfg["title"])
    plt.xlabel(cfg["xlabel"])
    plt.ylabel("Daño a propiedad (USD)")
    plt.tight_layout()
    plt.savefig(graficas_dir / cfg["archivo"])
    # plt.show()
    plt.close()

# de la misma manera vemos la tendencia de el ancho del tornado y la magnitud con el daño de propiedad, en donde 
# podemos ver que los tornados de 0 a 100 yd causan entre 10^3 y 10^7 dolares en daños y en la magnitud 
# esta entre 25 a 50 de este que causan de 10^2 a 10^5 dolares en daños

top10 = df.groupby("state")["event_type"].count().sort_values(ascending=False).head(10)
top5 = df.groupby("state")["damage_property"].sum().sort_values(ascending=False).head()

barplot_config = [
    {"top": top10, "title": "Top 10 estados con más eventos climáticos", "ylabel": "Cantidad de eventos", "archivo": "bar_top_estados_eventos.png"},
    {"top": top5, "yscale": 'log', "title": "Top 5 estados con daños en propiedades", "ylabel": "Daños en propiedades (USD)", "archivo": "bar_top_estados_daños.png"}
]

for cfg in barplot_config:
    sns.barplot(x=cfg["top"].index, y=cfg["top"].values)
    if cfg.get("yscale"):
        plt.yscale(cfg["yscale"])
    plt.title(cfg["title"])
    plt.xlabel("Estado")
    plt.ylabel(cfg["ylabel"])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(graficas_dir / cfg["archivo"])
    # plt.show()
    plt.close()

# en la grafica de barras de los top 5 estados con daños en propiedades me di cuenta que algunos estados no están en
# la grafica de los top 10 estados con más eventos climaticos, lo cual me llamó la atención porque el hecho que un 
# estado tenga muchos eventos, nos hace esperar que es el que más daños tiene, pero no es asi

florida_top_events = df[df["state"] == 'FLORIDA'].groupby("event_type")["event_type"].size().sort_values(ascending=False).head(5)
otros = (df[df["state"] == 'FLORIDA'].groupby("event_type")["event_type"].size().sort_values(ascending=False).tail(4)).sum()
florida_top_events["Otros"] = otros

plt.pie(florida_top_events.values, labels=florida_top_events.index, autopct='%1.1f')
plt.title("Tipos de evento en florida")
plt.tight_layout()
plt.savefig(graficas_dir / "pie_florida_eventos.png")
# plt.show()
plt.close()

# por eso mismo, creé esta grafica para ver qué tipo de eventos tiene florida que hace que tenga tanto daño de 
# propiedad y como vemos son las inundaciones, lluvias y tornados

print(df[df["state"] == 'FLORIDA'].groupby("event_type")["damage_property"].sum().sort_values(ascending=False))
print()

# pero eso no significa que el hecho de que haya muchas inundaciones, lluvia o tornado, significa que son las
# causantes de los daños, por eso lo checo aqui con un groupby
 
top3 = df.groupby("event_type")["deaths_direct"].sum().sort_values(ascending=False).head(3)
otros = (df.groupby("event_type")["deaths_direct"].sum().sort_values(ascending=False).tail(7)).sum()
top3["Otros"] = otros

plt.pie(top3.values, labels=top3.index, autopct='%1.1f')
plt.title("Muertes directas por tipo de evento")
plt.tight_layout()
plt.savefig(graficas_dir / "pie_muertes_eventos.png")
# plt.show()
plt.close()

# aqui simplemente veo cuales son los eventos que más muertes causan, lo cual es importante en mi opinión

tornados = df[df["event_type"] == 'Tornado']
cols = ['injuries_direct', 'injuries_indirect', 
        'deaths_direct', 'deaths_indirect',
        'damage_property', 'damage_crops',
        'tor_length', 'tor_width']
corr = tornados[cols].corr()

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlación entre variables numéricas")
plt.tight_layout()
plt.savefig(graficas_dir / "heatmap_var_num.png")
# plt.show()
plt.close()

# hice un heatmap para ver si el ancho o largo del tornado tiene que ver con los daños, muertes o heridos, y como 
# vemos no tienen tanto que ver, las relaciones más fuertes son los daños de propiedad y las muertes directas con los
# heridos directos, el ancho del tornado con el largo de este y otras relaciones no tan fuertes pero un poco 
# destacable son el ancho del tornado con los heridos directos, muertes directas y daños de cosecha

evento_por_anio = df.groupby(["event_type", df['begin_date_time'].dt.year]).size().reset_index(name='count')
evento_por_anio.columns = ['event_type', 'year', 'count']

plt.figure(figsize=(10, 5))
sns.lineplot(data=evento_por_anio, x='year', y='count', hue='event_type')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("Eventos por año y tipo de evento")
plt.xlabel("Año")
plt.ylabel("Cantidad de eventos")
plt.tight_layout()
plt.savefig(graficas_dir / "line_año_evento.png", bbox_inches='tight')
# plt.show()
plt.close()

# esto solo es para ver como han aumentado o disminuido los eventos conforme los años

df["year"] = df['begin_date_time'].dt.year
muertes_tornado = df[df["event_type"] == 'Tornado'].groupby("year")["deaths_direct"].sum()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.lineplot(data=df[df["event_type"] == 'Tornado'], x='year', y='tor_width', ax=ax1, color='tab:blue')
sns.lineplot(x=muertes_tornado.index, y=muertes_tornado.values, ax=ax2, color='tab:red')

ax1.set_xlabel("Año")
ax1.set_ylabel("Ancho del tornado (yd)", color='tab:blue')
ax2.set_ylabel("Muertes directas", color='tab:red')
plt.title("Evolución del ancho de tornados vs muertes por año")
plt.tight_layout()
plt.savefig(graficas_dir / "line_tornados_muertes_por_año.png")
# plt.show()
plt.close()

# aqui decidi hacer una grafica de lineas doble ya que primero lo hice solo con el ancho del tornado y su evolucion 
# con los años a ver si se habian hecho mas violentos o no conforme los años pero luego se me ocurrió convinarlo con 
# las muertes por tornados para ver si conforme los tornados se volvian mas grandes, a mas gente afectaba o si al 
# contrario disminuian las muertes y como vemos el hecho de que el tornado en promedio crezca conforme a los años, no 
# está ligado a cuantas muertes causa de hecho en 2021 bajó el promedio (de hecho es pico más bajo de esta linea) del 
# ancho del tornado pero fue el año en donde más muertes hubo

print(df["event_type"].where((df["begin_lat"].notna()) & (df["begin_lon"].notna())).value_counts()) # eventos que tienen coordenadas

geo_configs = [
    {"filtro": ['Marine Thunderstorm Wind', 'Flood', 'Tornado', 'Heavy Rain'], "titulo": "Eventos climáticos por ubicación", "archivo": "map.png"},
    {"filtro": ['Marine Thunderstorm Wind'], "titulo": "Marine Thunderstorm Wind por ubicación", "archivo": "marine_thunderstorm_wind_map.png"},
    {"filtro": ['Flood'], "titulo": "Floods por ubicación", "archivo": "flood_map.png"},
    {"filtro": ['Tornado'], "titulo": "Tornados por ubicación", "archivo": "tornado_map.png"},
    {"filtro": ['Heavy Rain'], "titulo": "Heavy Rains por ubicación", "archivo": "heavy_rain_map.png"}
]

for cfg in geo_configs:
    fig = px.scatter_geo(
        df[df["event_type"].isin(cfg["filtro"])],
        lat='begin_lat',
        lon='begin_lon',
        color='event_type',
        scope='usa',
        title=cfg["titulo"]
    )
    fig.update_geos(
        showcountries=True,
        showsubunits=True
    )
    fig.update_layout(
        title_font_size=32,
        font=dict(size=26),
        legend=dict(font=dict(size=24)),
        showlegend=len(cfg["filtro"]) > 1
    )

    fig.write_image(graficas_dir / cfg["archivo"], width=1600, height=1000, scale=2)

# por último decidí hacer unas graficas extra, las cuales son mapear la localizacion en los evento en un mapa mas que 
# nada por curiosidad, para ver en donde se concentran mas los eventos