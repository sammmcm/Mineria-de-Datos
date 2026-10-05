import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from matplotlib.axes import Axes

pd.set_option('display.max_columns', None)

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 5" / "graphs"
graficas_dir.mkdir(exist_ok=True, parents=True)

def haversine(lat1: float, lon1: float, lat2: float, lon2: float, r=3959) -> float: # para calcular la distancia
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * r * np.arcsin(np.sqrt(a))

def generar_df(df_original: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return (
        df_original
        .assign(
            duracion_min=lambda df: (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 60,
            distancia_millas=lambda df: haversine(df['begin_lat'], df['begin_lon'], df['end_lat'], df['end_lon']),
        )
        .assign(
            log_duracion_min=lambda df: np.log1p(df['duracion_min']),
            log_distancia_millas=lambda df: np.log1p(df['distancia_millas']),
            log_damage=lambda df: np.log1p(df['damage_property']),
            log_magnitude=lambda df: np.log1p(df['magnitude']),
            log_tor_width=lambda df: np.log1p(df['tor_width']),
            log_tor_length=lambda df: np.log1p(df['tor_length'])
        )
        .dropna(subset=cols)
    )

def generar_modelo(formula: str, df: pd.DataFrame) -> RegressionResultsWrapper:
    return smf.ols(formula, data=df).fit()

def generar_real_vs_predicho(ax: Axes, df: pd.DataFrame, modelo: RegressionResultsWrapper, title: str) -> None:
    sns.scatterplot(x=modelo.fittedvalues, y=df['log_damage'], alpha=0.3, ax=ax)
    lims = [df['log_damage'].min(), df['log_damage'].max()]
    ax.plot(lims, lims, color='red')
    ax.set_title(f"{title} | R² = {modelo.rsquared:.3f}")
    ax.set_xlabel("log(daño) predicho")
    ax.set_ylabel("log(daño) real")

def generar_comparacion(nombre_modelo: str, casos: dict, nombre_png: str) -> None:
    fig, axes = plt.subplots(1, len(casos), figsize=(6 * len(casos), 5))
    for ax, (titulo, (df, modelo)) in zip(axes, casos.items()):
        generar_real_vs_predicho(ax, df, modelo, titulo)
    fig.suptitle(f"Modelo {nombre_modelo}: real vs. predicho")
    fig.tight_layout()
    fig.savefig(graficas_dir / nombre_png)
    plt.close(fig)

def generar_diagnostico_escala(df: pd.DataFrame, var: str, etiqueta: str, name: str) -> None:
    columnas = [(var, etiqueta), (f"log_{var}", f"log(1 + {etiqueta})")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for i, (col, label) in enumerate(columnas):
        sns.histplot(df[col], ax=axes[0, i])
        axes[0, i].set_xlabel(label)
        axes[0, i].set_ylabel("Cantidad de eventos")

        sns.scatterplot(x=df[col], y=df["log_damage"], alpha=0.3, ax=axes[1, i])
        axes[1, i].set_xlabel(label)
        axes[1, i].set_ylabel("log(daño)")

    fig.suptitle(f"Escala de {etiqueta}: original vs. log")
    fig.tight_layout()
    fig.savefig(graficas_dir / name)
    plt.close(fig)

def resumir_modelo(modelo: RegressionResultsWrapper) -> dict[str, float]:
    fila = {
        'nobs': int(modelo.nobs), # filas usadas por el modelo
        'r2': modelo.rsquared, # qué proporcion de la variacion explica el modelo
        'r2_adj': modelo.rsquared_adj, # r2 penalizado por cantidad de variables
        'f_pvalue': modelo.f_pvalue, # dice si el modelo en conjunto es mejor que no tener variables
        'rmse': np.sqrt(modelo.mse_resid), # cuánto se equivoca en promedio
        'aic': modelo.aic, # para comparar modelos con el mismo nobs, menor es mejor
    }
    for var in modelo.params.index.drop('Intercept'):
        fila[f'coef_{var}'] = modelo.params[var] # efecto estimado en cada variable, qué tanto afecta
        fila[f'p_{var}'] = modelo.pvalues[var] # si el coeficiente es distinguible de 0, si el efecto es real o es ruido
    return fila

if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "dataset" / "clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

    cols_date = ['begin_date_time', 'end_date_time']
    cols_coords = ['begin_lat', 'begin_lon', 'end_lat', 'end_lon']

    cols = {
        'impacto': ['injuries_direct', 'injuries_indirect', 'deaths_direct', 'deaths_indirect', 'damage_property', 'damage_crops'],
        'ancho': ['damage_property', 'tor_width'],
        'largo': ['damage_property', 'tor_length'],
        'ancho_largo': ['damage_property', 'tor_width', 'tor_length'],
        'duracion': ['damage_property', *cols_date],
        'distancia': ['damage_property', *cols_coords],
        'duracion_distancia': ['damage_property', *cols_date, *cols_coords],
        'magnitude': ['damage_property', 'magnitude']
    }

    pct_nulos = (
        df[cols['impacto'] + cols_date + cols_coords].isna()
        .groupby(df['event_type']).mean().mul(100).round(2)
    )
    print("\t\tpct_nulos\n", pct_nulos)
    print()
    # vemos que en cuestión de coordenadas, el analisis solo se puede hacer con Flood, Heavy Rain, Marine Thunderstorm Wind y Tornado
    # esto es importante ya que se pierden filas, y un analisis con pocas filas no suele ser bueno

    pct_ceros = (
        (df[cols['impacto']] == 0)
        .where(df[cols['impacto']].notna()) # los nulos quedan fuera del promedio
        .groupby(df['event_type']).mean().mul(100).round(2)
    )
    print("\t\tpct_ceros\n", pct_ceros)
    print()
    # ahora con los ceros, lo importante es que no sean muchos, ya que aunque se puede hacer el analisis, de nada sirve 
    # hacerlo si el 100% de los datos son 0s
    # en este caso, solo podemos hacer el analisis en damage_property, pero solo en los tipos de eventos Flood, Strong Wind y Tornado

    # otra cosa que aclarar es que, aunque se podia hacer un analisis de daño ~ distancia Heavy Rain y Marine 
    # Thunderstorm Wind se descartan ya que el daño es cero en 97 y 99% de sus datos
    # Flood lo conservo ya que aunque hay 80% donde el daño es cero, no es tan extremo como las otras variables

    diagnostico = [
        ("duracion_min", "duración (min)", "Flood"),
        ("distancia_millas", "distancia (mi)", "Flood"),
        ("magnitude", "magnitud", "Strong Wind"),
        ("tor_width", "ancho del tornado", "Tornado"),
        ("tor_length", "largo del tornado", "Tornado"),
    ]

    # antes de hacer la regresión, como se vio en la practica 3, muchos de los datos son muy sesgados, por lo que 
    # veremos visualmente si es mejor transformar las variables con log para que los extremos no dominen el ajuste de la regresión
    for var, etiqueta, evento in diagnostico:
        df_diag = generar_df(df[df["event_type"] == evento], ["damage_property", var])
        generar_diagnostico_escala(df_diag, var, etiqueta, f"escala_{var}.png")
    # en este caso, la única variable que no necesita ser transformada con log es magnitude

    # lo siguiente lo hice para ver si hacer un analisis con duracion y distancia juntos, como lo haré con ancho y largo de los tornados
    for evento in ["Tornado", "Flood"]:
        df_corr = generar_df(df[df["event_type"] == evento], cols['duracion_distancia'])
        print(f"Corr {evento}: ", df_corr[['log_duracion_min', 'log_distancia_millas']].corr())

    formulas = {
        'ancho': 'log_damage ~ log_tor_width',
        'largo': 'log_damage ~ log_tor_length',
        'ancho_largo': 'log_damage ~ log_tor_width + log_tor_length',
        'duracion': 'log_damage ~ log_duracion_min',
        'distancia': 'log_damage ~ log_distancia_millas',
        'duracion_distancia': 'log_damage ~ log_duracion_min + log_distancia_millas',
        'magnitude': 'log_damage ~ magnitude'
    }

    modelos_por_evento = {
        'Tornado': ['ancho', 'largo', 'ancho_largo', 'duracion', 'distancia', 'duracion_distancia'],
        'Strong Wind': ['duracion', 'magnitude'],
        'Flood': ['duracion', 'distancia', 'duracion_distancia']
    }

    resultados = {}

    for evento, tipo_analisis in modelos_por_evento.items():
        for tipo in tipo_analisis:
            formula, columnas = formulas[tipo], cols[tipo]
            df_modelo = generar_df(df[df['event_type'] == evento], columnas)
            df_modelo_sin_0 = df_modelo[df_modelo['damage_property'] > 0]

            casos = {
                'Con ceros': (df_modelo, generar_modelo(formula, df_modelo)),
                'Sin ceros': (df_modelo_sin_0, generar_modelo(formula, df_modelo_sin_0))
            }

            resultados[(evento, tipo)] = casos

            generar_comparacion(f"{evento.lower().replace(' ', '_')}_{tipo}", casos, f"modelo_{evento.lower().replace(' ', '_')}_{tipo}.png")

    filas = []
    for (evento, tipo), casos in resultados.items():
        for caso, (_, modelo) in casos.items():
            filas.append({
                'evento': evento,
                'modelo': tipo,
                'caso': caso,
                **resumir_modelo(modelo)
            })

    resumen = pd.DataFrame(filas)
    resumen.to_csv(BASE_DIR / "Practica 5" / "resumen.csv", index=False)

    """ 
    Resultados
    Cada uno de los modelos explica solamente hasta un 30% del daño de propiedad, lo más destacable fue:
    En el tornado, cuando este sí hace daño explica un 30% del daño con respecto a ancho_largo, al igual que la duracion_distancia explica un 25%
    En las inundaciones, cuando este sí hace daño explica un 30% del daño con respecto a la duracion_distancia
    En los vientos fuertes, no se encontró algo destacable, lo que sugiere que otro factor/variable explica el daño, pero no es una información que tenga al momento (hacen falta variables)
    Para tornados en general (hagan daño o no), las variables mencionadas llegan a explicar hasta un 15% del daño
    Para las inundaciones, solo explica un 4%, pero esto es porque 80% de los daños registrados son 0
    """