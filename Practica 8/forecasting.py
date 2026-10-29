import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

pd.set_option('display.max_rows', None)

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 8" / "graphs"
graficas_dir.mkdir(exist_ok=True, parents=True)

def calcular_mae(test: pd.DataFrame, pred: pd.Series, nombre: str) -> None:
    mae = mean_absolute_error(test[nombre], pred) 
    print(f"{nombre} - MAE: {mae}, Error Absoluto: {(mae / test[nombre].mean() * 100).round(1)}%")

def graficar_pronostico(train: pd.Series, test: pd.Series, predicciones: dict[str, pd.Series], evento: str, archivo: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train, label="Train")
    ax.plot(test.index, test, label="Test (real)", color="black")
    for nombre, serie in predicciones.items():
        ax.plot(serie.index, serie, linestyle="--", label=nombre)
    ax.set_title(f"Pronóstico mensual - {evento}")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad de eventos")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(axis="x", which="minor", alpha=0.2)
    ax.grid(axis="x", which="major", alpha=0.5)
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300)
    plt.close(fig)

def pronosticar(serie: pd.Series, meses: int) -> pd.Series:
    modelo = ExponentialSmoothing(
        serie, trend=None, seasonal='add', seasonal_periods=12
    ).fit()

    return modelo.forecast(meses).clip(lower=0)

def graficar_prediccion(serie: pd.Series, futuro: pd.Series, evento: str, meses: int, archivo: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(serie.index, serie, label="Histórico")
    ax.plot(futuro.index, futuro, linestyle="--", color="tab:green", label="Pronóstico Holt-Winters")
    ax.set_title(f"Pronóstico a {meses} meses - {evento}")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Cantidad de eventos")
    ax.legend()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(axis="x", which="minor", alpha=0.2)
    ax.grid(axis="x", which="major", alpha=0.5)
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "dataset" / "clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

    print(df.groupby('event_type')['damage_property'].sum().sort_values(ascending=False).head(3))
    print()

    eventos = ['Flood', 'Strong Wind', 'Tornado'] # elegimos predecir estos eventos ya que son los que más hacen daño

    df_date = df[df['event_type'].isin(eventos)].groupby([pd.Grouper(key='begin_date_time', freq='MS'), 'event_type']).size().unstack(fill_value=0) # contamos cada event_type por mes del año

    print(df_date)
    print()

    train = df_date.loc[:'2023-12']
    test = df_date.loc['2024-01':]

    base = df_date.shift(12).loc['2024-01':]

    for evento in eventos:
        # vemos el error al predecir los eventos más recientes con los de el año pasado, esto para cuando hagamos la 
        # predicción con sarima u otros modelos, ver si vale la pena usar ese modelo o es peor que esta predicción simple
        calcular_mae(test, base[evento], evento)
        # como vemos todos tienen un error de más del 50% lo cual significa que ninguno es bueno
    print()

    pred_hw = {}
    for evento in eventos:
        modelo = ExponentialSmoothing( # predecimos con otro modelo a ver si mejora
            train[evento], trend=None, seasonal='add', seasonal_periods=12
        ).fit()
        pred = pronosticar(train[evento], len(test))

        calcular_mae(test, pred, evento)
        # como vemos la predicción mejoró, ya todos los eventos tienen un error de 49% a un 47%, pero sigue siendo algo
        # malo porque el error sigue siendo mucho, llegandose a equivocar casi la mitad de los datos

        pred_hw[evento] = pred
    print()

    print(train.groupby(train.index.month).mean().round(0)) 
    # aqui vemos que efectivamente, flood y tornado disminuyen en noviembre y aumentan en diciembre
    print()
    print(train.corr()) 
    # aqui se tiene una correlación baja entre todos los tipos de eventos (~0.25), ya que estamos comparando todos 
    # los años y en cada año cada evento tiene sus propios picos, que son diferentes casi siempre
    print()
    print(train.groupby(train.index.month).mean().corr()) 
    # pero si hacemos un promedio de cada mes del año, la correlación aumenta, más que nada en flood-tornado (0.68), que respalda lo que dijimos que ocurren en meses parecidos
    # los que casi no se parecen es tornado-strong wind con (0.068)
    print()

    for evento in eventos:
        configs = [
            ((p, 0, q), (P, 0, Q, 12))
            for p, q, P, Q in product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1])
        ]

        resultados = []
        for order, seasonal in configs:
            modelo = SARIMAX(train[evento], order=order, seasonal_order=seasonal).fit(disp=False)
            pred = modelo.forecast(len(test))
            mae = mean_absolute_error(test[evento], pred)
            resultados.append((mae, order, seasonal))

        mae, order, seasonal = min(resultados)
        modelo = SARIMAX(train[evento], order=order, seasonal_order=seasonal).fit(disp=False)
        pred_sarima = modelo.forecast(len(test))

        print(f"\t{order} x {seasonal}")
        calcular_mae(test, pred_sarima, evento)
        # como vemos, a comparación de holt, sarima empata en los eventos flood y strong wind pero holt gana en tornado,
        # por lo tanto eligiremos el modelo de holt para predecir datos futuros ya que es mucho más sencillo y es un poco mejor

        predicciones = {
            "Base (año pasado)": base[evento],
            "Holt-Winters": pred_hw[evento],
            "SARIMA": pred_sarima,
        }

        graficar_pronostico(train[evento], test[evento], predicciones, evento, f"forecast_{evento.lower().replace(' ', '_')}.png")
        # en las graficas vemos algo muy interesante, lo cual es que flood y tornado ocurren en meses parecidos, o al 
        # menos lo que predijo holt-winters se parecen, ya que en meses similares ambos aumentan y bajan, el más claro 
        # es al final de año donde en diciembre ambos aumentan, cuando en noviembre ambos habian disminuido

        # otra cosa que podemos notar es que lo que hace holt hace un promedio de todos los años anteriores para 
        # predecir, y se nota mucho en tornado, ya que en mayo hubo un pico de ~600 y holt predijo ~220 pero esto es 
        # porque en otros años no se habia presentado un pico tan grande, eso es una limitación, que tenemos pocos años para que el modelo prediga bien

        # algo que podemos decir de strong wind es que en promedio en mayo a septiembre se presentan pocos casos, el 
        # evento ocurre mas que nada en la primera parte del año y aumenta en noviembre-diciembre de año

        # vemos que efectivamente, sarima se parece mucho a lo que predijo holt, en la mayoria de veces es practicamente la misma forma

        # el modelo acierta la forma en general pero no la magnitud, se queda algo corto en ese aspecto
    print()

    futuro = {}
    for evento in eventos:
        futuro[evento] = pronosticar(df_date[evento], 12)
        graficar_prediccion(df_date[evento], futuro[evento], evento, 12, f"prediccion_{evento.lower().replace(' ', '_')}.png")
        # como vemos, cada grafica sigue una forma similar de otros años, por ejemplo como dijimos anteriormente en 
        # strong wind, su actividad baja por la mitad de año, en este caso predijo julio y luego aumenta, por lo que la 
        # predicción tiene sentido, solo que recordemos que falla en promedio la mitad de los casos (49%-47%), por lo 
        # tanto no podemos confiar tanto en la magnitud que predijo, y digo esto ya que en todas las graficas tienen 
        # picos históricos pero en lo que predijo holt no hay ningun pico así, ya que holt hace como un promedio de los 
        # datos, entonces se suaviza la cantidad de eventos, donde si nos ayuda es en ver en qué meses el evento puede tener más actividad o menos 

    print(pd.DataFrame(futuro).round(0))