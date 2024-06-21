import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler  
import pandas as pd
import folium
from folium.plugins import FloatImage

# Chargement des données depuis le fichier CSV
file_path = 'Data_Arbre.csv'
df = pd.read_csv(file_path)

print(f'\nValeurs pour les valeurs d entree non normalisees')
# Sélection des caractéristiques pour le clustering (hauteur totale et diamètre du tronc)
X = df[['haut_tot', 'tronc_diam']].values

# Determine k, le nombre optimal de cluster en utilisant la méthode du coude
inertia = []
for k in range(1, 11): 
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Affiche la courbe du coude
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choisi automatiquement le k optimal basé sur la méthode du coude
diff = [inertia[i] - inertia[i + 1] for i in range(len(inertia) - 1)]
optimal_k = diff.index(max(diff)) + 2  # +2 car range commence à 1, k commence à 2
#optimal_k=3 # Vous ajustez ce chiffre et enlevez le commentaire pour entrer un k à la main
print(f'Optimal number of clusters: {optimal_k}')

# Entrainement de KMeans avec k optimal sur les données non normalisées
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_

# Ajout des étiquettes de cluster à votre DataFrame
df['cluster'] = labels

# Print cluster validation scores
silhouette = silhouette_score(X, labels)
calinski_harabasz = calinski_harabasz_score(X, labels)
davies_bouldin = davies_bouldin_score(X, labels)

print(f'methode KMeans')
print(f'Silhouette Score: {silhouette}')
print(f'Calinski-Harabasz Index: {calinski_harabasz}')
print(f'Davies-Bouldin Index: {davies_bouldin}')









# Entrainez Agglomerative Clustering sur les données non normalisées
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(X)

# Ajout des étiquettes de cluster à votre DataFrame pour Agglomerative Clustering
df['agg_cluster'] = agg_labels

# Print cluster validation scores pour Agglomerative Clustering
agg_silhouette = silhouette_score(X, agg_labels)
agg_calinski_harabasz = calinski_harabasz_score(X, agg_labels)
agg_davies_bouldin = davies_bouldin_score(X, agg_labels)
print(f'methode Agglomerative Clustering')
print(f'Agglomerative Clustering - Silhouette Score: {agg_silhouette}')
print(f'Agglomerative Clustering - Calinski-Harabasz Index: {agg_calinski_harabasz}')
print(f'Agglomerative Clustering - Davies-Bouldin Index: {agg_davies_bouldin}')









# Entrainez DBSCAN sur les données non normalisées
dbscan = DBSCAN(eps=0.5, min_samples=5) 
dbscan_labels = dbscan.fit_predict(X)

# Ajout des étiquettes de cluster à votre DataFrame pour DBSCAN
df['dbscan_cluster'] = dbscan_labels

# Print cluster validation scores pour DBSCAN
# Note: Silhouette and Calinski-Harabasz scores ont besoin de plus d'un cluster pour être calculés
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X, dbscan_labels)
    dbscan_calinski_harabasz = calinski_harabasz_score(X, dbscan_labels)
else:
    dbscan_silhouette = -1
    dbscan_calinski_harabasz = -1

dbscan_davies_bouldin = davies_bouldin_score(X, dbscan_labels)
print(f'methode DBSCAN')
print(f'DBSCAN - Silhouette Score: {dbscan_silhouette}')
print(f'DBSCAN - Calinski-Harabasz Index: {dbscan_calinski_harabasz}')
print(f'DBSCAN - Davies-Bouldin Index: {dbscan_davies_bouldin}')









print(f'\nValeurs pour les valeurs d entree normalisees')
# Normalisation des données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine k, le nombre optimal de cluster en utilisant la méthode du coude
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Affiche la courbe du coude 
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Choisir automatiquement le k optimal basé sur la méthode du coude
diff = [inertia[i] - inertia[i + 1] for i in range(len(inertia) - 1)]
optimal_k = diff.index(max(diff)) + 2  # +2 because range starts from 1, k is from 2
print(f'Optimal number of clusters: {optimal_k}')

# Entrainez KMeans avec k optimal sur les données normalisées
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X_scaled)
labels = kmeans.labels_

# Ajout des étiquettes de cluster à votre DataFrame
df['cluster'] = labels

# Print cluster validation scores
silhouette = silhouette_score(X_scaled, labels)
calinski_harabasz = calinski_harabasz_score(X_scaled, labels)
davies_bouldin = davies_bouldin_score(X_scaled, labels)

print(f'methode KMeans')
print(f'Silhouette Score: {silhouette}')
print(f'Calinski-Harabasz Index: {calinski_harabasz}')
print(f'Davies-Bouldin Index: {davies_bouldin}')

# Entrainement de Agglomerative Clustering sur les données normalisées
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_clustering.fit_predict(X_scaled)

# Ajout des étiquettes de cluster à votre DataFrame pour Agglomerative Clustering
df['agg_cluster'] = agg_labels

# Print cluster validation scores pour Agglomerative Clustering
agg_silhouette = silhouette_score(X_scaled, agg_labels)
agg_calinski_harabasz = calinski_harabasz_score(X_scaled, agg_labels)
agg_davies_bouldin = davies_bouldin_score(X_scaled, agg_labels)
print(f'methode Agglomerative Clustering')
print(f'Agglomerative Clustering - Silhouette Score: {agg_silhouette}')
print(f'Agglomerative Clustering - Calinski-Harabasz Index: {agg_calinski_harabasz}')
print(f'Agglomerative Clustering - Davies-Bouldin Index: {agg_davies_bouldin}')

# Entrainez DBSCAN sur les données normalisées
dbscan = DBSCAN(eps=0.5, min_samples=5) 
dbscan_labels = dbscan.fit_predict(X_scaled)

# Ajout des étiquettes de cluster à votre DataFrame pour DBSCAN
df['dbscan_cluster'] = dbscan_labels

# Print cluster validation scores pour DBSCAN
if len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    dbscan_calinski_harabasz = calinski_harabasz_score(X_scaled, dbscan_labels)
else:
    dbscan_silhouette = -1
    dbscan_calinski_harabasz = -1

dbscan_davies_bouldin = davies_bouldin_score(X_scaled, dbscan_labels)
print(f'methode DBSCAN')
print(f'DBSCAN - Silhouette Score: {dbscan_silhouette}')
print(f'DBSCAN - Calinski-Harabasz Index: {dbscan_calinski_harabasz}')
print(f'DBSCAN - Davies-Bouldin Index: {dbscan_davies_bouldin}')









# Création du scatter plot pour visualiser les clusters
plt.figure(figsize=(10, 6))
colors = ['yellow', 'blue', 'green', 'red', 'purple', 'orange'][:optimal_k]
for cluster in range(optimal_k):
    cluster_points = df[df['cluster'] == cluster]
    plt.scatter(cluster_points['haut_tot'], cluster_points['tronc_diam'], 
                c=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel('Hauteur Totale')
plt.ylabel('Diamètre du Tronc')
plt.title('Diamètre du Tronc en fonction de la Hauteur Totale')
plt.legend()
plt.show()









# Initialisation de la carte centrée sur la moyenne des coordonnées
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Defini les couleurs des clusters
cluster_colors = ['yellow', 'blue', 'green', 'red', 'purple', 'orange'][:optimal_k]

# ajoute les marqueurs de cluster à la carte
for index, row in df.iterrows():
    folium.CircleMarker([row['latitude'], row['longitude']],
                        radius=1,
                        color=cluster_colors[row['cluster']],
                        fill=True,
                        fill_color=cluster_colors[row['cluster']],
                        fill_opacity=0.7).add_to(m)

# Crée la légende des clusters en html
legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 150px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white;
     ">&nbsp; Cluster Légende <br>
     '''
for i in range(optimal_k):
    legend_html += f'&nbsp; Cluster {i} &nbsp; <i class="fa fa-map-marker fa-2x" style="color:{cluster_colors[i]}"></i><br>'

legend_html += '</div>'

m.get_root().html.add_child(folium.Element(legend_html))

# Sauvegarde la carte en HTML
m.save('clustered_trees_map.html')
