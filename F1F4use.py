import os
import sys
import json
import pandas as pd
import numpy as np

def load_centroids(file_path):
    df = pd.read_csv(file_path)
    return df[['haut_tot_centroid', 'tronc_diam_centroid']].values

def predict_cluster(centroids, haut_tot, tronc_diam):
    # Calcule la distance entre le nouveau point et chaque centroid
    distances = np.linalg.norm(centroids - np.array([haut_tot, tronc_diam]), axis=1)
    # Trouve le cluster le plus proche
    cluster = np.argmin(distances)
    return cluster

def main():
    # On récupère les arguments de la ligne de commande
    if len(sys.argv) != 3:
        print("Usage: python F1F4use.py <haut_tot> <tronc_diam>")
        sys.exit(1)
    
    haut_tot = float(sys.argv[1])
    tronc_diam = float(sys.argv[2])

    # Defini le chemin du fichier contenant les centroids
    centroids_file_path = 'centroids.csv'

    try:
        # Charge les centroids depuis le fichier CSV
        centroids = load_centroids(centroids_file_path)
        print("Centroids loaded from disk.")
    except (FileNotFoundError, EOFError):
        print(f"Error: {centroids_file_path} not found or is corrupted.")
        sys.exit(1)

    # Predie le cluster pour le nouveau point de données
    cluster = predict_cluster(centroids, haut_tot, tronc_diam)
    
    # Ecrit le cluster prédit dans un fichier JSON
    with open('predicted_cluster.json', 'w') as json_file:
        json.dump(int(cluster), json_file)

    print(f"Cluster number {int(cluster)} saved to predicted_cluster.json")

if __name__ == "__main__":
    main()
