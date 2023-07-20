# Se importan las librerías necesarias
import pandas as pd
import re
import utils.functions2 as fn


# Se importa el dataset con los datos
df_init = pd.read_csv("src/data/raw/df_alquiler_2020_21_processed.csv", sep=";", index_col=0)

# Se crea una lista con las columnas cuyo nombre termina en "2020"
col_to_drop = [col for col in df_init.columns if col.endswith("2020")]

# Se eliminan las variables con 2020 al final
df_2021 = df_init.drop(col_to_drop, axis=1)

df_2021.columns = [re.sub(r"_20\d{2}$", '', col) for col in df_2021.columns]

df_2021.drop(["Provincia", "Codigo_municipio","Codigo_provincia"],axis=1, inplace=True)

df_features = pd.read_csv("src/app/data/df_features.csv", sep=";", index_col=0)
df_2021.drop(df_features[6:].index, axis=1, inplace=True)

final_model = fn.get_final_model(df_2021, "Comunidad_autonoma", "Alquiler_mes_vu_m2", "src/models/my_model.pkl")