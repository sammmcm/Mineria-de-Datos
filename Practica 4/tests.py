import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from pathlib import Path

pd.set_option('display.max_rows', None)

BASE_DIR = Path(__file__).resolve().parent.parent

def generar_qq_plots(df: pd.DataFrame) -> None:
    qq_dir = BASE_DIR / "Practica 4" / "qq_plots"
    qq_dir.mkdir(parents=True, exist_ok=True)

    eventos_validos = df.groupby('event_type')['deaths_direct'].count()
    eventos_validos = eventos_validos[eventos_validos > 0].index

    for evento in eventos_validos:
        subset = df[df['event_type'] == evento]['deaths_direct'].dropna()
        stats.probplot(subset, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot - {evento} (deaths_direct)")
        plt.savefig(qq_dir / f"qq_{evento.lower().replace('/', ' ').replace(' ', '_')}.png")
        plt.close()

def clasificar_efecto(e: float) -> str:
    if e < 0.01: return "despreciable"
    if e < 0.06: return "pequeño"
    if e < 0.14: return "moderado"
    return "grande"

def formatear_p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

def kruskal_por_grupo(df: pd.DataFrame, grupo_col: str, variables: list[str]) -> pd.DataFrame:
    filas = []
    for var in variables:
        grupos = [df[df[grupo_col] == valor][var].dropna() for valor in df[grupo_col].unique()]
        grupos = [g for g in grupos if len(g) >= 5] # descartamos los grupos con menos de 5 datos ya que la 
        # aproximacion chi-cuadrada es poco confiable con grupos menores a 5

        h, p = stats.kruskal(*grupos)
        n = sum(len(g) for g in grupos)
        epsilon2 = h / (n - 1) # como con muchos datos (n) p casi siempre sale significativo, usaremos epsilon para saber qué tan grande es la diferencia entre las categorias

        filas.append({"variable": var, "H": h, "p_value": p, "epsilon2": epsilon2, "efecto": clasificar_efecto(epsilon2)})
    return pd.DataFrame(filas)

def holm_correction(df: pd.DataFrame) -> None:
    reject, p_adj, _, _ = multipletests(df["p_value"], alpha=0.05, method="holm")
    df["p_adj"] = p_adj
    df["significativo"] = reject

def posthoc(df: pd.DataFrame, grupo_col: str, var: str) -> pd.DataFrame:
    conteos = df.groupby(grupo_col)[var].count()
    grupos = conteos[conteos >= 5].index
    df_filtrado = df[df[grupo_col].isin(grupos) & df[var].notna()]

    dunn = sp.posthoc_dunn(df_filtrado, val_col=var, group_col=grupo_col, p_adjust='holm')

    pares = dunn.stack().reset_index()
    pares.columns = ['grupo_1', 'grupo_2', 'p_adj']
    pares = pares[pares['grupo_1'] < pares['grupo_2']]
    pares['significativo'] = pares['p_adj'] < 0.05

    return pares.sort_values('p_adj')

def imprimir_resultados(df: pd.DataFrame, nombre_grupo: str) -> None:
    print(f"Por {nombre_grupo}")
    for fila in df.sort_values("epsilon2", ascending=False).itertuples():
        if fila.significativo:
            texto = f"Sí hay diferencias y tiene un efecto {fila.efecto}"
        else:
            texto = "No hay diferencias significativas"
        print(f"{fila.variable}: {texto} ({formatear_p(fila.p_adj)}, ε² = {fila.epsilon2:.3f})")

if __name__ == '__main__':
    df = pd.read_csv("dataset/clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

    # para ver si mis datos (en este caso con la variable de muertes directas) sigue una distribución normal o no para ver 
    # si usar anova + prueba t o kruskal-wallis en este caso nos daremos cuenta mediante q-q plot que lo que hace es 
    # comparar los cuantiles de mis datos contra los de una distribución normal

    # generar_qq_plots(df)

    # como vemos en las imagenes, ningun tipo de evento sigue una distribución normal, por lo tanto usaremos kruskal-wallis 
    # nota: no chequé la normalidad para todas las variables ya que además de que es mucho cómputo, no es necesario ya 
    # que nos podemos dar cuenta que todas las variables siguen este patrón, es decir, la mayoria de estas, por no 
    # decir todas, tienen muchos valores en 0 y luego tienen valores extremos, como vemos que pasa en todas las graficas

    variables_et = ['deaths_direct', 'injuries_direct', 'deaths_indirect', 'injuries_indirect', 'damage_property', 'damage_crops'] # no se añaden las variables de magnitud ni de tornado porque será para el analisis por tipo de evento, no tiene mucho sentido usarlas ahi
    variables_state = [*variables_et, 'magnitude', 'tor_length', 'tor_width']

    # hacemos la prueba de kruskal
    df_et = kruskal_por_grupo(df, 'event_type', variables_et)
    df_state = kruskal_por_grupo(df, 'state', variables_state)

    # ajustamos el valor de p con holm ya que por la cantidad de pruebas que hacemos, el error en cada una va aumentando
    holm_correction(df_et)
    holm_correction(df_state)

    imprimir_resultados(df_et, 'tipo de evento')
    print()
    imprimir_resultados(df_state, 'estado')

    # aqui vemos que en todas las variables si hay diferencias menos en injuries indirect de los estados, y esto es porque como vemos a continuacion

    totales = df.groupby('state')['injuries_indirect'].sum().sort_values(ascending=False)
    print()
    print(f"{(totales > 0).sum()} estados de {len(totales)} con injuries indirect > 0")
    print(f"{(df['injuries_indirect'] > 0).sum()} de {len(df)} datos con injuries indirect > 0")

    # como vemos casi todos los valores de injuries indirect son 0, como kruskal ordena los datos en rangos, con 
    # tantos 0s casi todos quedan empatados en el mismo rango, por eso la prueba no detecta diferencias
    
    # las variables en donde más diferencias se vieron fueron damage property para tipo de evento y tor_width para estados
    # usaremos la prueba de post-hoc de dunn para ver cuáles categorias difieren en esas variables donde hubo más diferencias que las demás (mayor ε²)
    print()
    pares_et = posthoc(df, 'event_type', 'damage_property')
    print(f"Por tipo de evento: {pares_et['significativo'].sum()} de {len(pares_et)} pares son significativos")

    print()
    print(df.groupby('event_type')['damage_property'].agg(['mean', 'median', lambda s: (s > 0).mean()]))
    print()
    print(pares_et.head(8))

    # como vemos en el dataframe, vemos que categorias como drought, excessive heat, extreme cold difieren de flood, strong wind y tornado ya que entre el 20% y 63% de sus eventos causa daño frente a casi 0% en los otros

    print()
    pares_estados = posthoc(df, 'state', 'tor_width')
    print(f"Por estado: {pares_estados['significativo'].sum()} de {len(pares_estados)} pares son significativos")

    resumen = df.groupby('state')['tor_width'].agg(
        n='count',
        media='mean',
        mediana='median',
    )
    print()
    print(resumen.loc[['COLORADO', 'MISSISSIPPI']])
    print()
    print(pares_estados.head(8))

    # por esta parte vemos que pares como mississippi y colorado difieren, esto puede depender por la mediana del 
    # ancho de los tornados de mississippi (200 yd) contra la mediana de colorado (30 yd) pero no significa que uno 
    # sea mas peligroso que el otro

    ruta_csv = BASE_DIR / "Practica 4" / "resultados_csv"
    ruta_csv.mkdir(parents=True, exist_ok=True)

    df_et.to_csv(ruta_csv / "kruskal_event_type.csv", index=False)
    df_state.to_csv(ruta_csv / "kruskal_states.csv", index=False)
    pares_et.to_csv(ruta_csv / "dunn_event_type.csv", index=False)
    pares_estados.to_csv(ruta_csv / "dunn_states.csv", index=False)