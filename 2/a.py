from sklearn.datasets import load_wine
import pandas as pd

# load data
data = load_wine()

# create DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)


#### Aufgabe 1: Neues Attribut erstellen ####
df["substracted_phenols"] = df["total_phenols"] - df["nonflavanoid_phenols"]

#### Aufgabe 2: Werte von "alcohol" um 1.0 erhöhen ####
df["alcohol"] += 1.0

# Ergebnisse anzeigen
print("Neue Spalte 'substracted_phenols' hinzugefügt:")
print(df[["total_phenols", "nonflavanoid_phenols", "substracted_phenols"]].head())

print("\nWerte von 'alcohol' nach Erhöhung um 1.0:")
print(df)