import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd

# Daten laden
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Werte des Attributs "ash" extrahieren
ash = df["ash"]

#### Teil 1: Histogramm ####
plt.hist(ash, bins=10)

# Achsenbeschriftungen
plt.xlabel("Wertebereiche (Bins)")
plt.ylabel("Häufigkeit")
plt.title("Histogramm der Werte von 'ash' mit 10 Bins")

# Bin-Bereiche auf der x-Achse anzeigen
bin_edges = plt.hist(ash, bins=10)[1]  # Bin-Kanten
plt.xticks(bin_edges, rotation=45)  # Werte auf der x-Achse

# Histogramm anzeigen
plt.tight_layout()  # Layout verbessern
plt.show()

#### Teil 2: Boxplot ####
plt.boxplot(ash)

# Achsenbeschriftungen
plt.ylabel("Wert von 'ash'")
plt.title("Boxplot des Attributs 'ash'")

# Boxplot anzeigen
plt.tight_layout()  # Layout verbessern
plt.show()


# Kommentar:
# Der Boxplot zeigt:
# - Den Interquartilsabstand (IQR), der den Bereich von Q1 (25. Perzentil) bis Q3 (75. Perzentil) darstellt.
# - Den Median (mittlerer Wert), dargestellt durch die Linie innerhalb der Box.
# - Potenzielle Ausreißer als Punkte außerhalb der "Whiskers" (die maximal 1.5 * IQR von Q1 und Q3 entfernt sind).
# - Die Werte von 'ash' liegen größtenteils zwischen ca. 1.8 und 2.8.