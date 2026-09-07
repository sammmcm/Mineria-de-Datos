# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import pandas as pd
from pathlib import Path

download_path = kagglehub.dataset_download("bwandowando/noaa-storm-events-database")

df_list = []
for df_chunk in pd.read_csv(Path(download_path) / "StormEvents_details.csv" / "StormEvents_details.csv", chunksize=100000):
    # filtramos a que los datos sean a partir de 2020 para que sea mas rapido la extraccion, ya que son aprox 2 millones de datos
    df_chunk = df_chunk[df_chunk['YEAR'] >= 2020]
    df_list.append(df_chunk)

df_complete = pd.concat(df_list, ignore_index=True)
print()
print(len(df_complete))

BASE_DIR = Path(__file__).resolve().parent.parent
dataset_dir = BASE_DIR / "dataset"
dataset_dir.mkdir(exist_ok=True)
dataset_path = dataset_dir / "raw_StormEvents_details.csv"
df_complete.to_csv(dataset_path, index=False)