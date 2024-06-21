import os
import sys
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

def train_kmeans(df, n_clusters):
    X = df[['haut_tot', 'tronc_diam']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_scaled)
    return kmeans, scaler

def save_model(model, scaler, model_path, scaler_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

def load_model(model_path, scaler_path):
    kmeans = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return kmeans, scaler

def predict_cluster(kmeans, scaler, longitude, latitude, clc_secteur, haut_tot, tronc_diam, age_estim):
    X_new = scaler.transform([[longitude, latitude, clc_secteur, haut_tot, tronc_diam, age_estim]])
    cluster = kmeans.predict(X_new)
    return cluster[0]

def main():
    if len(sys.argv) != 7:
        print("Usage: python F3F4use.py <longitude>   <latitude>  <clc_secteur>  <haut_tot>  <tronc_diam>  <age_estim>")
        sys.exit(1)

    longitude = float(sys.argv[1])
    latitude = float(sys.argv[2])
    clc_secteur = float(sys.argv[3])
    haut_tot = float(sys.argv[4])
    tronc_diam = float(sys.argv[5])
    age_estim = float(sys.argv[6])

    # Defini le chemin du fichier contenant les centroids
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'RandomForest_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler2.pkl')

    try:
        # Essaie de charger le modèle et le scaler depuis le disque 
        kmeans, scaler = load_model(model_path, scaler_path)
        print("Model and scaler loaded from disk.")
    except (FileNotFoundError, EOFError):
        print("pas de .pkl")

    # Prediction du cluster pour le nouveau point de données
    cluster = predict_cluster(kmeans, scaler, longitude, latitude, clc_secteur, haut_tot, tronc_diam, age_estim)

    # Ecrit le cluster prédit dans un fichier JSON
    with open('predicted_cluster.json', 'w') as json_file:
        json.dump(int(cluster), json_file)

    print(f"Cluster number {int(cluster)} saved to predicted_cluster.json")

if __name__ == "__main__":
    main()
