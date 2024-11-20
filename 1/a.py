from sklearn.datasets import load_wine
import pandas as pd

# Daten laden
data = load_wine()

# DataFrame erstellen
df = pd.DataFrame(data.data, columns=data.feature_names)

#### Teil 1: Überblick über den Datensatz ####
print("#### Teil 1: Überblick über den Datensatz ####")

# Anzahl der Beispiele
print("Anzahl der Beispiele:", data.data.shape[0])

# Anzahl der Attribute
print("Anzahl der Attribute:", len(data.feature_names))

# Namen der Attribute
attribute_names = ", ".join(data.feature_names)
print("Namen der Attribute:", attribute_names)

# Datentypen der Attribute
print("\nDatentypen der Attribute:")
print(df.dtypes)

# Beschreibung des Datensatzes (optional)
#print(data.DESCR)


#### Teil 2: Analyse einzelner Attribute ####
print("\n#### Teil 2: Analyse einzelner Attribute ####")

# Werte des Attributs "ash" extrahieren
ash = df["ash"]
print("\nAttribut 'ash' ausgewählt.")

# Statistische Werte für "ash"
print("Maximalwert von 'ash':", max(ash))
print("Minimalwert von 'ash':", min(ash))
print("Median von 'ash':", ash.median())
print("Mittelwert von 'ash':", ash.mean())


#### Teil 3: Analyse des label Attribute ####
print("\n#### Teil 3: Analyse des label Attribute ####")
# Anzahl der Klassen
num_classes = len(data.target_names)
print("Anzahl der Klassen:", num_classes)

# Anzahl der Beispiele pro Klasse
value_counts = pd.Series(data.target).value_counts()
for value, count in value_counts.items():
    print(f"Klasse {value}: {count}")



