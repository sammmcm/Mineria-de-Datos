import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections.abc import Iterable
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 7" / "graphs"
graficas_dir.mkdir(exist_ok=True, parents=True)

def evaluar_ks_y_graficar(X: np.ndarray | pd.DataFrame, archivo: str, ks: Iterable[int] = range(2, 11), sample_size: int = 10000) -> None:
    inercias, siluetas = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inercias.append(km.inertia_)
        siluetas.append(silhouette_score(X, km.labels_, sample_size=sample_size, random_state=42))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ks, inercias, marker='o')
    axes[0].set_title("Método del codo")
    axes[1].plot(ks, siluetas, marker='o')
    axes[1].set_title("Silhouette")
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

def resumen_cluster(df: pd.DataFrame, cluster: str) -> pd.DataFrame:
    g = df.groupby(cluster)
    ang = np.arctan2(g['mes_sin'].mean(), g['mes_cos'].mean())
    return pd.DataFrame({
        'n': g.size(),
        'damage_property': np.expm1(g['log_damage_property'].mean()),
        'duracion_min': np.expm1(g['log_duracion_min'].mean()),
        'mes': (ang % (2 * np.pi)) / (2 * np.pi) * 12,
    }).round(2)

def graficar_pca(X: np.ndarray, k: int, archivo: str) -> None:
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', s=5, alpha=0.5)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title(f"Clusters (k={k}) en espacio PCA")
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

def graficar_clusters(df: pd.DataFrame, ncols: int, figsize: tuple[float, float], ks: Iterable[int], X: np.ndarray | pd.DataFrame, s: float, archivo: str) -> None:
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    for ax, k in zip(axes, ks):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        ax.scatter(df['begin_lon'], df['begin_lat'], c=labels, cmap='tab10', s=s)
        ax.set_title(f"k={k}")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

def graficar_mapa_zonas(df: pd.DataFrame, k: int, archivo: str) -> None:
    # mapa de las zonas de k-means con el número de cada zona sobre su centro
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df['begin_lon'], df['begin_lat'], c=df[f'zona_{k}'], cmap='tab10', s=3)
    for z, g in df.groupby(f'zona_{k}'):
        ax.text(g['begin_lon'].mean(), g['begin_lat'].mean(), str(z), fontsize=16, weight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    df = pd.read_csv("dataset/clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

    df = df.assign(duracion_min=lambda df: (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 60)
    mes = df['begin_date_time'].dt.month
    df = df.assign(
        mes_sin=np.sin(2 * np.pi * mes / 12),
        mes_cos=np.cos(2 * np.pi * mes / 12),
        log_duracion_min=lambda df: np.log1p(df['duracion_min']),
        log_damage_property=lambda df: np.log1p(df['damage_property']),
        log_damage_crops=lambda df: np.log1p(df['damage_crops']),
    )

    cols = ['log_damage_property', 'log_duracion_min', 'mes_sin', 'mes_cos'] # columnas que usaremos para k-means

    df_km = df.dropna(subset=cols).copy() # quitamos los nulos porque k-means no los acepta
    print(len(df), len(df_km)) # pasamos de tener 108370 a 86752 filas, lo cual es suficiente para el análisis
    print()
    # veamos qué tanto se perdió por tipo de evento
    comparacion = pd.concat(
        [df['event_type'].value_counts(),
        df_km['event_type'].value_counts()],
        axis=1, keys=['original', 'tras_dropna']
    )
    comparacion['pct_retenido'] = (comparacion['tras_dropna'] / comparacion['original'] * 100).round(1)
    print(comparacion.sort_values('pct_retenido'))
    # las filas para cada tipo de evento se mantuvo bien, ninguna se quedó con muy pocos datos

    X = StandardScaler().fit_transform(df_km[cols]) # escalamos las variables para que la escala de damage no domine el mes, ya que esta es mucho menos que damage

    evaluar_ks_y_graficar(X, "evaluacion_ks_general.png")
    # vemos que en el metodo del codo hay un cambio (se suaviza la grafica) en k=4, pero en silhouette se ve más claro que los mejores k fueron k=4 y k=9
    print()

    for k in (4, 9): # vemos los grupos de cada k para ver cual divide los grupos mejor
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        df_km[f'cluster_{k}'] = km.labels_
        print(f"k={k}")
        print(df_km.groupby(f'cluster_{k}')[cols].mean().round(2))
        print(pd.crosstab(df_km[f'cluster_{k}'], df_km['event_type'], normalize='index').round(2))
        print()

    print(resumen_cluster(df_km, 'cluster_4'))
    print()
    print(resumen_cluster(df_km, 'cluster_9'))
    print()

    for k in (4, 9):
            print(k, round(adjusted_rand_score(df_km['event_type'], df_km[f'cluster_{k}']), 3))

    """
    en el k=4:
        - lo que más destaca es que el grupo 3 es el que más daño tiene (~11 mil) mientras que los demas tienen ~0, dura ~47 minutos, y está compuesto principalmente por Flood, Strong Wind y Tornado
        - del grupo 1 y 2, duran ~42 horas, sin embargo el grupo 1 (Drought, Extreme Cold, Flood y Heavy Snow) ocurre a mitades de Enero, mientras que el grupo 2 (Drought, Excessive Heat, Heat y Flood) ocurre a mitades de Julio
        - el grupo 0 son eventos muy cortos (~1 min) y está compuesto más que nada por Marine Thunderstorm
    en el k=9:
        - el grupo que más daño tiene es el grupo 6 (Tornado, Flood y Strong Wind) con ~25 mil el cual dura ~21 min y ocurre a finales de Abril y el que le sigue es el grupo 3 (Strong Wind, Flood, Tornado) con ~6 mil con una duracion de ~1.7 hrs y ocurre a principios de Enero, todos los demás son ~0
        - el grupo que dura más es el grupo 4 con ~103 horas, después los grupos 7, 2 y 0 duran entre ~59 a ~33 horas
    en ambos salieron con un ARI bajo, lo que significa que el daño, duracion y mes no son suficientes para saber el tipo de evento
    sin embargo k=9 es mas detallado que k=4, ya que por ejemplo en k=4 veiamos que solo un grupo era el que más hacía daño, pero con k=9 hay 2 grupos que causan más daño
    esta separación nos da información valiosa, ya que podemos ver que el grupo 6 hace mucho daño, dura poco y ocurre en Abril y el grupo 3 hace menos daño que el gpo 6, dura más y ocurre en Enero, cosa que en k=4 no veiamos
    """

    graficar_pca(X, 4, "clusters_k4_pca.png")
    graficar_pca(X, 9, "clusters_k9_pca.png")
    # en k=4 los eventos con daño se dividen en un grupo (celeste) mientras que en k=9 son dos grupos (gris y rojo) como habiamos visto anteriormente
    # los demás grupos (que se ven como lineas) son eventos sin daños, donde lo unico que varia es la duracion

    # hacemos un k-means para los tornados de acuerdo a la longitud y latitud
    torn = df[df['event_type'] == 'Tornado']
    print(len(torn), torn[['begin_lat', 'begin_lon']].notna().all(axis=1).sum()) # solo se pierden 9 filas
    print()

    torn_geo = torn.dropna(subset=['begin_lat', 'begin_lon']).copy() # nos quedamos con las que solo tienen coordenadas

    fuera = torn_geo[(torn_geo['begin_lon'] < -125) | (torn_geo['begin_lon'] > -70)] # solo nos enfocaremos en el territorio principal de USA
    print(len(fuera)) # solo hay 6 filas fuera del territorio principal
    print()
    print(fuera[['state', 'begin_lat', 'begin_lon']])
    print()

    torn_geo = torn_geo[~torn_geo['state'].isin(['PUERTO RICO', 'ALASKA'])].copy()
    Xg = torn_geo[['begin_lat', 'begin_lon']]

    evaluar_ks_y_graficar(Xg, "evaluacion_ks_tornado.png", sample_size=5000) 
    # en el metodo del codo no queda muy en claro el mejor k pero en silhouette vemos k=2, 6 y 8

    graficar_clusters(torn_geo, 3, (16, 4), (2, 6, 8), Xg, 4, "clusters_tornado.png") 
    # k=8 a mi parecer se ve mejor separado, se ve que cada grupo tiene sentido, por ejemplo el celeste de k=6 son 2 
    # grupos en k=8 y tiene sentido porque la parte celeste de este grupo, son todos más separados y en el verde estan 
    # mas juntos

    km = KMeans(n_clusters=8, n_init=10, random_state=42).fit(Xg)
    torn_geo['zona'] = km.labels_

    print(torn_geo['zona'].value_counts().sort_index()) # veamos cuantos valores tiene cada grupo
    # vemos que hay un desbalance en cuanto al grupo 7 y los demás, este tiene muy pocos valores, por lo tanto hay que 
    # tener cuidado con lo que descubramos de ese grupo, ya que hay muy poca evidencia
    print()
    print(torn_geo.groupby('zona')[['begin_lat', 'begin_lon']].mean().round(2))
    print()
    print(torn_geo.groupby('zona')[['tor_width', 'tor_length', 'deaths_direct', 'injuries_direct', 'damage_property']].median().round(2))
    # se eligió la mediana porque el daño es muy sesgado, muchos de los registros tienen un daño de 0 y los registros
    # que tienen valores extremos arrastran al promedio a un valor muy grande, como se verá más adelante
    print()
    print(pd.crosstab(torn_geo['zona'], torn_geo['tor_f_scale'], normalize='index').round(2))
    print()
    """
        vemos que el grupo con más daño es el 3 con 40 mil, este grupo se conforma más que nada de tornados tipo EF0, EF1 y EF2
        los grupos 6 y 1 tienen un daño de 15 mil y ambos se parecen en las escalas de los tornados, en donde si se nota la diferencia son los tornados tipo EFU, donde en el grupo 6 son un 15% de los tornados y el en grupo 1 solo un 3%
        lo mismo pasa con los grupos 5 y 0 que tienen un daño similar, solo difieren en una o dos escalas de tornados
        los demás grupos aunque tienen un daño de 0, si hay mucha diferencia entre las escalas de los tornados, por ejemplo el grupo 2 tiene que el 52% de los tornados son EFU mientras que los demás tienen un 17%, 20%
        otra cosa que podemos notar es que en el grupo 0 y 3, aunque sus escalas se parecen, el daño es muy diferente (5 mil y 40 mil) por lo que podemos pensar que es porque una zona es remota y la otra es más urbanizada, pero no es un dato que tengamos lamentablemente
    """
    for zona, datos_zona in torn_geo.groupby('zona'):
        print(zona, datos_zona['state'].value_counts(normalize=True).head(3).round(2).to_dict())
    # aqui podemos ver los estados en donde cae cada zona 
    print()
    print(torn_geo.groupby('zona')['damage_property'].agg(['mean', 'median']).round(0))
    # como dijimos antes, el promedio se hace muy grande por los valores extremos que tiene el grupo, cuando en 
    # realidad la mayoria de los valores no son tan grandes
    print()
    print(torn_geo.groupby('zona')['damage_property'].apply(lambda s: s.isna().mean()).round(3))
    # otra cosa que aclarar es que entre ~9% y ~36% de los tornados de cada zona no tienen un registro del daño, por lo que las medianas y promedios de arriba se calcularon sin ellos
    print()

    # como casi todos los muertos y heridos son 0, arriba nos aparecía como tal, pero veamos los porcentajes
    resumen = torn_geo.groupby('zona').agg(
        pct_con_muertes=('deaths_direct', lambda s: (s > 0).mean() * 100),
        pct_con_heridos=('injuries_direct', lambda s: (s > 0).mean() * 100),
    ).round(1)
    print(resumen)
    # los grupos en donde más se registran heridos y muertes son los grupos 1, 6, 5 y 0
    # esos grupos tienen en su mayoria escalas EF1 y EF0
    print()
    print(torn_geo.groupby('tor_f_scale').agg(
        n=('tor_f_scale', 'size'),
        pct_muertes=('deaths_direct', lambda s: (s > 0).mean() * 100),
        pct_heridos=('injuries_direct', lambda s: (s > 0).mean() * 100),
    ).round(1))
    # viendo los heridos y muertes de cada escala de tornado, vemos que EF4 tiene un porcentaje muy alto, pero muy 
    # pocos registros, lo cual lo hace poco confiable, el que le sigue es EF3 que si tiene un poco mas de registros, 
    # entonces es más sólido
    # tambien otra cosa que podemos notar es que el % de heridos aumenta conforme a la escala del tornado
    print()
    # el hecho de que en los grupos que mas heridos y muertes tienen sean en su mayoría EF0 y EF1, estos no son los 
    # principales causantes de las tragedias, estos grupos tambien cuentan con tornados tipo EF3 y solo el grupo 1 de 
    # EF4 y estos son los que son más propensos a causar daño, por lo tanto, esos tornados son los que elevan el numero
    # aunque el grupo 4 se parece al los grupos 0, 3 y 5, este tiene menos heridos y muertos, pero ninguna de las variables lo explica

    # ahora hacemos un k-means por ubicación pero para las inundaciones
    flood = df[df['event_type'] == 'Flood']
    print(len(flood), flood[['begin_lat', 'begin_lon']].notna().all(axis=1).sum()) # no se pierde ninguna fila
    print()

    flood_geo = flood.dropna(subset=['begin_lat', 'begin_lon']).copy() # nos quedamos solo con las que tienen coordenadas

    # nos quedamos con los registros del territorio principal como en los tornados
    fuera = flood_geo[(flood_geo['begin_lon'] < -125) | (flood_geo['begin_lon'] > -70) | (flood_geo['begin_lat'] < 24)]
    print(fuera['state'].value_counts())
    print()

    excluir = ['PUERTO RICO', 'ALASKA', 'HAWAII', 'GUAM', 'AMERICAN SAMOA', 'VIRGIN ISLANDS'] 
    flood_geo = flood_geo[~flood_geo['state'].isin(excluir)].copy() # quitamos esos estados/territorios
    Xf = flood_geo[['begin_lat', 'begin_lon']]
    print(len(flood_geo)) # solo perdimos ~ mil filas, lo cual está bien
    print()

    evaluar_ks_y_graficar(Xf, "evaluacion_ks_flood.png", sample_size=5000)
    # en el método del codo el que tiene la inercia mas alta es el 2 y el cambio se presenta en el 3 y en silhouette los picos después de la caída fueron 6 y 7

    graficar_clusters(flood_geo, 4, (20, 4), (2, 3, 6, 7), Xf, 3, "clusters_flood.png")
    # al ver los mapas ninguno me pareció mejor que el otro, en cuanto al 3, 6 y 7, cada uno separa algunas zonas bien 
    # y otras no, por ejemplo en el k=6 la zona amarilla y morado podrían ser uno solo, por lo tanto analizaré los 3 
    # para ver si vale la pena hacer muchos o pocos grupos
    # no agregaré el 2, ya que aunque hace una buena división, los demás dan más detalle

    for k in (3, 6, 7):
        flood_geo[f'zona_{k}'] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xf)
        print(f"k={k}")
        print(flood_geo.groupby(f'zona_{k}').agg(
            n=(f'zona_{k}', 'size'),
            lat=('begin_lat', 'mean'),
            lon=('begin_lon', 'mean'),
            pct_con_dano=('damage_property', lambda s: (s > 0).mean() * 100),
            pct_con_heridos=('injuries_direct', lambda s: (s > 0).mean() * 100),
            duracion_mediana=('duracion_min', 'median'),
        ).round(1))
        print()

    for k in (6, 7):
            graficar_mapa_zonas(flood_geo, k, f"zonas_flood_k{k}.png")

    """
    podemos ver que k=3 da poco detalle, k=6 lo mejora ya que da más detalle en cuanto a grupos por la duración y la proporcion del daño
    por ejemplo en k=3 el unico que se distingue de los demás es el grupo dos con 759 minutos y el daño no varía tanto como en k=6, ahi se tienen grupos de 140, 355, y 4,700 minutos, algo que no veiamos en k=3, igual con el daño donde ahí varía de un ~11% a un ~24% y en k=6 de un ~8% a un ~30%
    con respecto a k=6 y k=7 este da aun mas detalle util, y algo en donde se nota mucho es que en el grupo 4, este nos dice el 33% de esa zona tuvo daño, lo cual es un detalle que se perdía en k=6 ya que esa zona estaba contenida en el grupo 2
    """
    
    print(pd.crosstab(flood_geo['zona_6'], flood_geo['zona_7'])) # aqui vemos que efectivamente, el grupo 2 de k=6 se dividió entre los grupos 4 y 5 de k=7 en donde el grupo 4 son solo valores del grupo 2 de k=6