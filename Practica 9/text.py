import pandas as pd
from pathlib import Path
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
lemmatizer = WordNetLemmatizer() # esto es para palabras como wind y winds, se convierten en una sola que es wind

BASE_DIR = Path(__file__).resolve().parent.parent
graficas_dir = BASE_DIR / "Practica 9" / "graphs"
graficas_dir.mkdir(exist_ok=True, parents=True)

# palabras genéricas que no sirven ya que no nombran un fenómeno
STOPWORDS_DOMINIO = {
    'county', 'area', 'inch', 'low', 'hour', 'month',
    'produced', 'afternoon', 'central', 'condition',
    'moved', 'morning', 'severe', 'portion', 'high'
}
STOPWORDS = set(ENGLISH_STOP_WORDS) | STOPWORDS_DOMINIO

def limpiar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r'\b\d+(st|nd|rd|th)\b', '', texto) # para ordinales como 3rd, 4th, etc
    texto = re.sub(r'[^a-z\s]', '', texto)
    palabras = [lemmatizer.lemmatize(p) for p in texto.split()]
    palabras = [p for p in palabras if p not in STOPWORDS and len(p) > 2]
    # 'd' venía de las categorías del drought monitor (d0-d4), se descartó porque el 99.6% de esos episodios ya tiene 
    # la palabra drought, por lo tanto no lo necesitamos para distinguir el fenómeno (por eso el len(p) > 2)

    return ' '.join(palabras)

def graficar_heatmap(tabla: pd.DataFrame, archivo: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(tabla, annot=True, cmap='OrRd', fmt='.1f')
    ax.set_title("Porcentaje de episodios de cada tipo de evento que mencionan la palabra")
    ax.set_xlabel("Palabra")
    ax.set_ylabel("Tipo de evento")
    fig.tight_layout()
    fig.savefig(graficas_dir / archivo, dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    df = pd.read_csv(BASE_DIR / "dataset" / "clean_StormEvents_details.csv", parse_dates=['begin_date_time', 'end_date_time'])

    print(df[['episode_narrative', 'event_narrative']].isna().mean().mul(100).round(2)) # vemos que tantos nulos tienen
    # como vemos, solo event narrative tiene y tiene un 19% de nulos
    print()
    for col in ['episode_narrative', 'event_narrative']:
        print(df[col].str.len().describe())
        # vemos que la longitud promedio de las narrativas en episode es de ~532 caracteres y de event es de ~220, 
        print()
    print(df[['episode_narrative', 'event_narrative']].nunique()) # vemos los registros únicos
    print()
    print(df[['episode_narrative', 'event_narrative']].count()) 
    # al contar los registros totales que tenemos, podemos ver que si hay muchos registros duplicados, donde más se nota es en episode, ya que pasa de ~108 mil a ~17 mil filas
    print()
    print(df['event_narrative'].isna().groupby(df['event_type']).mean().mul(100).round(2)) # vemos que cantidad de nulos tiene cada tipo de evento
    # algunos llegan a tener hasta un ~39% de nulos, lo cual si es bastante a comparacion de otros que no tienen o tienen un 2%, por lo tanto está algo sesgado
    print()

    # haremos un analisis por episodio, ya que no tiene nulos y tiene 19 mil episodios (con 17 mil textos únicos) con 
    # ~500 caracteres cada una es mas que suficiente
    print(df.groupby('episode_id')['episode_narrative'].nunique().value_counts()) # vemos que solo 19 episodios tienen 2 textos distintos
    print()

    df_episodes = df.drop_duplicates(subset='episode_id')
    print(len(df_episodes))
    print()
    print(df_episodes['episode_narrative'].value_counts().head()) # vemos que lo maximo que se repite una narrativa es 40 veces
    # dejaremos las narrativas duplicadas para que la nube refleje cuantas veces se repite cada fenomeno
    print()

    print(df_episodes['episode_narrative'].str.split().explode().value_counts().head(15)) # vemos las palabras que dominan
    # como vemos hay muchas palabras innecesarias, como the, of, and
    print()

    df_episodes['clean_text'] = df_episodes['episode_narrative'].apply(limpiar_texto) # quitamos estas palabras que no necesitamos

    print(df_episodes['clean_text'].str.split().explode().value_counts().head(15)) 
    # ahora si vemos las palabras que dominan, las cuales nos ayudarán en la nube
    print()

    texto = ' '.join(df_episodes['clean_text'])
    nube = WordCloud(width=1600, height=900, background_color='white', collocations=False).generate(texto) # creamos la nube de palabras
    nube.to_file(graficas_dir / "wordcloud_episodios.png")
    # como vemos las palabras que más destacan son thunderstorm, drought, snow, wind, strong, storm, heavy, rainfall y
    # cold veamos en que tipos de eventos se repiten más esas palabras, ya que por ejemplo wind se puede repetir en varios eventos

    palabras = ['thunderstorm', 'drought', 'snow', 'wind', 'strong', 'storm', 'heavy', 'rainfall', 'cold']

    tabla = pd.DataFrame({
        p: df_episodes['clean_text']
            .str.contains(rf'\b{p}\b', na=False)
            .groupby(df_episodes['event_type'])
            .mean()
            .mul(100)
            .round(1)
        for p in palabras
    })
    graficar_heatmap(tabla, "heatmap_palabras_evento.png")
    # podemos notar que drought, snow y cold casi identifican un tipo, por ejemplo drought aparece 96.6% en drought, 
    # snow un 88.4% en heavy snow y cold 76.1% en extreme cold
    # sin embargo en todas las demás palabras (thunderstorm, heavy, strong, storm, etc) ya no solo se concentran en un
    # solo evento, sino en varias, similar a como pasaba en knn, el cual confundia tipos parecidos

    # ya que extreme cold aparece con una frecuencia alta en wind aunque este evento sea más que nada de frio, veamos si pasa eso por su nombre (wind chills)
    sin_chill = df_episodes['clean_text'].str.replace(r'\bwind chills?\b', '', regex=True)
    con_wind = sin_chill.str.contains(r'\bwind\b', na=False)
    print(con_wind.groupby(df_episodes['event_type']).mean().mul(100).round(1))
    # vemos que baja de 90.4% a 66.1%, por lo tanto si era en parte por su nombre