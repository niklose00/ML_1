import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler

# Daten laden
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # Zielwerte hinzufügen

#### Aufgabe a) Neues Attribut erstellen ####

# a.1) Neues Attribut "substracted_phenols" erstellen
substracted_phenols = df["total_phenols"] - df["nonflavanoid_phenols"]
insert_pos = df.columns.get_loc("nonflavanoid_phenols") + 1
df.insert(loc=insert_pos, value=substracted_phenols, column="substracted_phenols")

# a.2) Alle Werte von "alcohol" um 1.0 erhöhen
df["alcohol"] += 1.0

#### Aufgabe b) Datenaugmentation ####

# b.1) Daten shufflen
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# b.2) Daten mit Gaussian Noise augmentieren
np.random.seed(42)
random_indices = np.random.choice(len(df), size=20, replace=False)  # 20 zufällige Indizes
augmented_examples = df.iloc[random_indices].copy()
augmented_examples.iloc[:, :-1] += np.random.normal(loc=0, scale=1, size=augmented_examples.iloc[:, :-1].shape)
df_augmented = pd.concat([df_shuffled, augmented_examples], ignore_index=True)
print(df)

#### Aufgabe c) Daten skalieren ####

# c.1) Skaliere das Attribut "ash" in den Bereich [0,1]
scaler = MinMaxScaler()  # MinMaxScaler gewählt, weil er Werte in den Bereich [0,1] transformiert und einfach interpretierbar ist
df_augmented["ash_scaled"] = scaler.fit_transform(df_augmented[["ash"]])

#### Ergebnisse anzeigen ####

print("Neue Attribute hinzugefügt:")
print(df_augmented[["alcohol", "total_phenols", "nonflavanoid_phenols", "substracted_phenols", "ash_scaled"]].head())

print("\nAnzahl der Beispiele nach Datenaugmentation:", len(df_augmented))
print("Originale Anzahl der Beispiele:", len(df))
