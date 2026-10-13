import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 6" / "graphs"
graficas_dir.mkdir(exist_ok=True, parents=True)

pd.set_option('display.max_columns', None)

K_BASE = 5
KS = [1, 3, 5, 7, 9, 15, 25, 35, 50, 75]
K_FINAL = 15

def preparar_modelo(X, cols_numericas, idx_train, idx_test):
    X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]

    # escalamos las columnas, ya que estas tienen escalas muy distintas y no queremos que numeros grandes dominen la distancia
    scaler = StandardScaler()
    X_train_sc, X_test_sc = X_train.copy(), X_test.copy()
    X_train_sc[cols_numericas] = scaler.fit_transform(X_train[cols_numericas])
    X_test_sc[cols_numericas] = scaler.transform(X_test[cols_numericas])
    return X_train_sc, X_test_sc

def guardar_matriz_confusion(y_true, pred, titulo, archivo):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, normalize='true', values_format='.2f', ax=ax
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluar_cv(X_train_sc, y_train, grupos_train, k, nombre):
    modelo = KNeighborsClassifier(n_neighbors=k)
    pred_cv = cross_val_predict(
        modelo, X_train_sc, y_train,
        cv=GroupKFold(n_splits=5), groups=grupos_train, n_jobs=-1
    )
    print(f"Modelo {nombre} (k={k}) | columnas: {X_train_sc.shape[1]} | filas: {len(y_train)}")
    print(classification_report(y_train, pred_cv))
    guardar_matriz_confusion(
        y_train, pred_cv,
        f'Matriz de confusión - Modelo {nombre} (k={k})',
        f'matriz_confusion_modelo_{nombre.lower()}.png'
    )
    return classification_report(y_train, pred_cv, output_dict=True)

def buscar_k(X_train_sc, y_train, grupos_train, nombre):
    print(f"Búsqueda de k - Modelo {nombre}")
    for k in KS:
        modelo = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(
            modelo, X_train_sc, y_train,
            groups=grupos_train, cv=GroupKFold(n_splits=5), scoring='f1_macro', n_jobs=-1
        )
        # lo hacemos con cross para evaluar cada k sin usar el test, se usa todo el train para entrenar y validar
        print(k, scores.mean().round(4), scores.std().round(4))

def evaluar_test(X_train_sc, y_train, X_test_sc, y_test, k, nombre):
    modelo_final = KNeighborsClassifier(n_neighbors=k)
    modelo_final.fit(X_train_sc, y_train)
    pred = modelo_final.predict(X_test_sc)

    print(f"Modelo {nombre} en test (k={k})")
    print(classification_report(y_test, pred))
    guardar_matriz_confusion(
        y_test, pred,
        f'Matriz de confusión - Modelo {nombre} (test)',
        f'matriz_confusion_test_modelo_{nombre.lower()}.png'
    )

if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "dataset" / "clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])
    df = df.assign(duracion_min=lambda df: (df['end_date_time'] - df['begin_date_time']).dt.total_seconds() / 60)

    cols = [
        'injuries_direct', 'injuries_indirect', 'deaths_direct', 'deaths_indirect',
        'damage_property', 'damage_crops',
        'duracion_min'
    ]
    print(df[cols].isna().groupby(df['event_type']).mean().mul(100).round(2))  # checamos nulos
    print()

    print(
        (df[cols] == 0)
        .where(df[cols].notna())
        .groupby(df['event_type']).mean().mul(100).round(2)
    )  # checamos ceros
    print()

    print(df['event_type'].value_counts().sort_index())
    print()

    # como de damage property y crops hay nulos, hay que ver cuántas filas perdemos y si nos afecta quitarlas
    print(df.dropna(subset=['damage_property'])['event_type'].value_counts().sort_index())
    print()

    print(df.dropna(subset=['damage_crops'])['event_type'].value_counts().sort_index())
    print()
    # si hay filas suficientes para knn

    # para decidir si llenar los nulos de damage con 0s o borrar las filas (dropna), vemos si cambian los injuries y deaths dependiendo si el damage es nulo o no
    for col_dano in ['damage_property', 'damage_crops']:
        nulos = df[df[col_dano].isna()]
        no_nulos = df[df[col_dano].notna()]
        for nombre, grupo in [('nulos', nulos), ('no nulos', no_nulos)]:
            for par in (['injuries_direct', 'deaths_direct'], ['injuries_indirect', 'deaths_indirect']):
                print(col_dano, nombre, (grupo[par] == 0).mean().round(3).to_dict())
            print()
    # todos dieron un 99% de 0s, entonces es muy probable que los nulos signifique que simplemente no se registró el daño
    # por lo tanto quitaremos los nulos

    print(df[['damage_property', 'damage_crops', 'duracion_min']].skew())  # vemos la asimetria de las variables para ver si aplicar log para que estén más simétricas
    print()

    print(np.log1p(df[['damage_property', 'damage_crops', 'duracion_min']]).skew())  # en duracion min no hay tanta asimetria, pero vemos si mejora con el log
    # todos las variables mejoran, entonces aplicaremos log a estos 3
    print()

    mes = df['begin_date_time'].dt.month
    df = df.assign(
        mes_sin=np.sin(2 * np.pi * mes / 12),
        mes_cos=np.cos(2 * np.pi * mes / 12),
        # esto es para solucionar la problematica de las distancias, ya que en knn puede parecer que diciembre (12) está
        # muy lejos de enero (1) que no es así, por lo tanto hacemos que los meses formen un círculo
        log_duracion_min=lambda df: np.log1p(df['duracion_min']),
        log_damage_property=lambda df: np.log1p(df['damage_property']),
        log_damage_crops=lambda df: np.log1p(df['damage_crops']),
    )

    features = {
        "a": ['injuries_direct', 'injuries_indirect', 'deaths_direct', 'deaths_indirect', 'log_damage_property', 'log_damage_crops', 'log_duracion_min', 'mes_sin', 'mes_cos'],
        "b": ['log_damage_property', 'log_duracion_min', 'mes_sin', 'mes_cos']
    }

    df = df.dropna(subset=features['a'])
    # por último, verificamos si en un episodio se repiten muchos tipos eventos muy similares, ya que si es asi el modelo puede hacer trampa, si una fila cae en train y otra en test, el modelo encuentra esos eventos similares y acierta por copia
    print(df.groupby('episode_id').size().describe())
    print()
    print((df.groupby('episode_id')['event_type'].nunique() == 1).mean())
    print()
    # vemos que en efecto, en promedio (93%) cada uno de los episodios tiene eventos que son de un solo tipo, por lo 
    # tanto haremos el analisis con grupos
    
    y = df['event_type']
    grupos = df['episode_id']  # un episodio = un grupo    

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(gss.split(df, y, groups=grupos))
    y_train, y_test = y.iloc[idx_train], y.iloc[idx_test]
    grupos_train = grupos.iloc[idx_train]

    reportes = {}

    # Modelo A: ver qué tan bien predice knn el event type con todas las columnas numéricas
    X_train_sc_a, X_test_sc_a = preparar_modelo(df[features['a']], features['a'], idx_train, idx_test)
    reportes['A'] = evaluar_cv(X_train_sc_a, y_train, grupos_train, K_BASE, 'A')
    # se obtuvo un 0.59 de f1-score con k=5

    # como vemos en la matriz de confusión, los que mejor predijo el modelo fueron drought, marine thunderstorm y tornado
    # con el que más batalló fue extreme cold con heavy snow y heavy rain con flood y marine thunderstorm wind, y esto puede ser porque sus caracteristicas pueden parecerce, que duren similar, mes del año, etc. esto tambien se ve un poco con heat, el cual el modelo a veces predijo que era extreme heat

    # para comprobar esto vemos el promedio de los meses y la duracion en los tipos de evento
    print(df.groupby('event_type')[['log_duracion_min', 'mes_sin', 'mes_cos']].mean())
    # vemos que en heat y extreme heat, las 3 variables si se acercan mucho entre sí, lo que explica por qué lo llega a
    # a confundir el modelo, por lo tanto para extreme cold y heavy snow si se parecen pero más en cuanto a la duración 
    # y para heavy rain, flood y marine thunderstorm no se parecen mucho, solo heavy rain y flood en cuanto a duración 
    # y mes sin, pero marine thunderstorm con los demás no, por lo que puede ser por otro factor
    print()

    # vamos a probar con diferentes valores de k a ver si mejora el f1-score
    buscar_k(X_train_sc_a, y_train, grupos_train, 'A')
    # como vemos el mejor fue 35, que es el que tiene el mayor f1-score (0.6338) y se mantiene estable
    print()

    # Modelo B: ver si damage property, duracion min y mes ayudan a distinguir más los tipos de eventos
    # quitamos todas las demas columnas ya que como los 0s en esas variables son constantes (casi todos tienen un 90% o
    # 100%), no varian entre tipos de eventos, no le sirve al modelo para distinguirlos
    X_train_sc_b, X_test_sc_b = preparar_modelo(df[features['b']], features['b'], idx_train, idx_test)
    reportes['B'] = evaluar_cv(X_train_sc_b, y_train, grupos_train, K_BASE, 'B')
    buscar_k(X_train_sc_b, y_train, grupos_train, 'B')
    # en este modelo el k=35 se mantiene como el mejor, pero su f1-score reduce un poco (0.6245), por lo tanto las 
    # variables que quitamos aportan un poco al modelo
    print()

    # Modelo C: ver si la localizacion (state) ayuda a distinguir más los tipos de eventos
    dummies = pd.get_dummies(df['state'], prefix='state', dtype=int) # como son categorias, usamos dummies para convertirlos a numericas
    df_c = pd.concat([df[features['a']], dummies], axis=1) # añadimos las columnas de dummies
    assert len(df_c) == len(y) == len(grupos) and df_c.isna().sum().sum() == 0 
    # assert de seguridad para que df_c, y y grupos tengan las mismas filas y el modelo no use datos desalineados, al 
    # igual verificamos que df_c no tenga ningun nulo
    X_train_sc_c, X_test_sc_c = preparar_modelo(df_c, features['a'], idx_train, idx_test)
    reportes['C'] = evaluar_cv(X_train_sc_c, y_train, grupos_train, K_BASE, 'C')
    buscar_k(X_train_sc_c, y_train, grupos_train, 'C')
    # a diferencia de los modelos anteriores, este mejora considerablemente en cuanto a f1-score, el mejor k fue 15 con un f1-score de 0.7472
    # cabe aclarar que hubo empates entre 5, 7, 9 y 15, pero elegimos 15 ya que es el mas estable de todos, además apartir de ahi el rendimiento baja
    print()

    # comparacion entre modelos con k=5
    tabla = pd.DataFrame({
        nombre: {clase: r[clase]['f1-score'] for clase in r if clase not in ('accuracy', 'weighted avg')}
        for nombre, r in reportes.items()
    })
    tabla['diferencia_C_A'] = tabla['C'] - tabla['A'] # para ver como los states aportaron al modelo
    print(tabla.round(3))
    print()

    # reporte más detallado del mejor modelo (C)
    evaluar_test(X_train_sc_c, y_train, X_test_sc_c, y_test, K_FINAL, 'C')
    # vemos que los mejores fueron drought y marine thunderstorm wind
    # el modelo sigue confundiendo heavy rain con flood, excessive heat con heat y extreme cold con heavy snow, pero al
    # menos ya todos los eventos tienen un recall mayor o igual a 0.5

    # ya que states le ayudó mucho al modelo a distinguir marine thunderstorm de flood y heavy rain, veamos por qué
    prop_marine = (df['event_type'] == 'Marine Thunderstorm Wind').groupby(df['state']).mean()# proporcion de filas de marine en cada state
    print(prop_marine[prop_marine > 0.85].sort_values(ascending=False)) 
    # elegimos los que son > 0.85 para ver qué zonas maritimas hay, no usamos 1 ya que esas son exclusivas de marine thunderstorm
    print()
    # todas las zonas parecen pertenecerle a marine thunderstorm
    zonas = prop_marine[prop_marine == 1.0].index
    marine = df[df['event_type'] == 'Marine Thunderstorm Wind']
    print("Proporcion marine avg: ", marine['state'].isin(zonas).mean())
    # y en efecto, todas son de marine thunderstorm, entonces cuando el modelo ve esas zonas sabe que es ese evento
    print()

    # veamos el rendimiento de los modelos sin marine thunderstorm
    sin_marine = tabla.drop(['Marine Thunderstorm Wind', 'macro avg'])
    print(sin_marine[['A', 'C']].mean()) # no disminuye casi nada, por lo tanto el modelo no depende del atajo

""" Nota: no grafiqué knn porque es muy complicado con mi dataset y los modelos que hice, por ejemplo el C tiene masomenos 
60 columnas y en una grafica 2D no se aprecia muy bien, por esa razón me apoyé más que nada con la matriz de confusión """