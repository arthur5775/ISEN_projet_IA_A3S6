import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# Charger le fichier CSV
file_path = 'Data_Arbre.csv'
df = pd.read_csv(file_path)

# Sélectionner les caractéristiques pour le clustering
X = df[['haut_tot', 'tronc_diam']].values

# Initialiser et ajuster le modèle K-means avec n_clusters=2 et random_state=42
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Extraire les centroides
centroids = kmeans.cluster_centers_

# Afficher les centroides
print("Centroids:")
print(centroids)

# Convertir les centroides en DataFrame
centroids_df = pd.DataFrame(centroids, columns=['haut_tot_centroid', 'tronc_diam_centroid'])

# Sauvegarder les centroides dans un fichier CSV
output_file = 'centroids.csv'
centroids_df.to_csv(output_file, index=False)

print(f"Centroids saved to {output_file}.")
