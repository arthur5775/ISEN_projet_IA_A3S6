RQ: RandomForest_model.pkl impossile d'upload sur github car "Yowza, that’s a big file. Try again with a file smaller than 25MB."

Github: https://github.com/arthur5775/projet_IA_A3S6/tree/main

Projet de Classification et de Clustering d'Arbres
Ce projet contient des scripts pour la classification et le clustering d'arbres en utilisant des techniques d'apprentissage automatique. 
Les scripts permettent de prédire la classe d'âge d'un arbre et de déterminer à quel cluster il appartient en fonction de diverses caractéristiques.



FICHIER: F1.py
UTILISATION: python F1.py

Ce script exécute plusieurs méthodes de clustering (KMeans, DBSCAN, et Agglomerative Clustering) sur les données des arbres, détermine le nombre optimal de clusters, et affiche une carte interactive des clusters.
Ce script charge les données depuis un fichier CSV, exécute les méthodes de clustering sur les données normalisées et non normalisées, et affiche les scores de validation des clusters. 
Il génère également une carte interactive avec les arbres colorés par cluster.
Fonctionnalités
    Détermination du nombre optimal de clusters : Utilise la méthode du coude pour choisir le nombre optimal de clusters.
    Clustering : Applique KMeans, DBSCAN, et Agglomerative Clustering aux données.
    Évaluation des clusters : Calcule les scores de silhouette, de Calinski-Harabasz et de Davies-Bouldin pour évaluer la qualité des clusters.
    Visualisation : Crée une carte interactive Folium affichant les arbres colorés par cluster.
Exemple de Sortie
    Affiche des courbes du coude pour déterminer le nombre optimal de clusters.
    Affiche les scores de validation pour chaque méthode de clustering.
    Génère une carte HTML interactive clustered_trees_map.html.



FICHIER: F1F4_calcul_centroides.py
UTILISATION: python F1F4_calcul_centroides.py

Ce script calcule les centroides des clusters en utilisant l'algorithme K-means et enregistre les centroides dans un fichier CSV.
Ce script charge les données depuis un fichier CSV, exécute l'algorithme K-means pour un nombre fixe de clusters (2), et sauvegarde les centroides dans un fichier CSV.
Exemple de Sortie
    Affiche les coordonnées des centroides.
    Sauvegarde les centroides dans un fichier centroids.csv.



FICHIER: F1F4use.py
UTILISATION: python F1F4use.py <haut_tot> <tronc_diam>
    haut_tot : Hauteur totale de l'arbre.
    tronc_diam : Diamètre du tronc de l'arbre.
ex: python F1F4use.py 50 50

Ce script prédit à quel cluster appartient un arbre donné en utilisant les centroides calculés par le script F1F4_calcul_centroides.py.
Ce script charge les centroides depuis un fichier CSV, calcule les distances euclidiennes entre l'arbre donné et les centroides, et prédit à quel cluster appartient l'arbre. 
Le résultat est sauvegardé dans un fichier JSON.



FICHIER: F2.py
UTILISATION: python F2.py

Ce script entraîne et évalue différents modèles de classification pour prédire la classe d'âge estimée des arbres en fonction de certaines caractéristiques.
Fonctionnalités
    Prétraitement des Données :
        Lecture des données depuis un fichier CSV.
        Sélection des colonnes pertinentes et création de catégories pour l'âge.
        Encodage des colonnes catégorielles.
        Normalisation des données.
        Division des données en ensembles d'entraînement et de test.
    Entraînement des Modèles :
        Entraîne trois modèles de classification différents : RandomForest, DecisionTree, et KNeighbors.
        Utilisation de GridSearchCV pour trouver les meilleurs hyperparamètres.
    Évaluation des Modèles :
        Évaluation des performances des modèles sur l'ensemble de test en utilisant des métriques telles que l'accuracy, la précision, et le recall.
        Affichage de la matrice de confusion et du rapport de classification.
        Tracé des courbes ROC multi-classes pour chaque modèle et affichage de la valeur de l'aire sous la courbe.
    Sauvegarde des Modèles :
        Sauvegarde les meilleurs modèles, ainsi que l'encodeur et le scaler, dans des fichiers .pkl pour une utilisation ultérieure.
Exemple de Sortie
    Affiche les meilleurs hyperparamètres trouvés pour chaque modèle.
    Affiche les métriques de performance (accuracy, précision, recall) pour chaque modèle.
    Affiche la matrice de confusion pour chaque modèle.
    Affiche les courbes ROC multi-classes pour chaque modèle.



FICHIER: F2F4use.py
UTILISATION: python F2F4use.py <method> <haut_tot> <haut_tronc> <tronc_diam> <fk_stadedev>
    method : Le nom du modèle à utiliser (doit correspondre à un fichier modèle .pkl existant).
    haut_tot : Hauteur totale de l'arbre.
    haut_tronc : Hauteur du tronc de l'arbre.
    tronc_diam : Diamètre du tronc de l'arbre.
    fk_stadedev : Stade de développement de l'arbre (doit être l'une des valeurs suivantes : 'Jeune', 'Adulte', 'Vieux', 'senescent').
ex: python F2F4use.py DecisionTree 27 6 565 Adulte

Ce script permet de prédire la classe d'âge d'un arbre en utilisant un modèle d'apprentissage automatique pré-entraîné.



FICHIER: F3.py
UTILISATION: python F3.py
    
Ce script contient les étapes de préparation des données, l'entraînement de différents modèles de classification, et l'évaluation de leurs performances.



FICHIER: F3F4use.py
UTILISATION: python F3F4use.py <longitude> <latitude> <clc_secteur> <haut_tot> <tronc_diam> <age_estim>
    longitude : Longitude de l'arbre.
    latitude : Latitude de l'arbre.
    clc_secteur : Secteur de localisation de l'arbre.
    haut_tot : Hauteur totale de l'arbre.
    tronc_diam : Diamètre du tronc de l'arbre.
    age_estim : Âge estimé de l'arbre.
ex: python F3F4use.py 3.2932636093638927 50.84050020512298 47 57 123 150

Ce script permet de prédire à quel cluster appartient un arbre en utilisant un modèle de clustering KMeans pré-entraîné.



bibliothèques :
    pandas
    numpy
    scikit-learn
    joblib
    matplotlib
    seaborn
    folium



Organisation des Fichiers
    F1.py : Script pour le clustering et la visualisation des arbres.
    F1F4_calcul_centroides.py : Script pour le calcul des centroides des clusters.
    F1F4use.py : Script pour la prédiction des clusters à partir des centroides.
    F2.py : Script pour l'entraînement et l'évaluation des modèles de classification.
    F2F4use.py : Script pour la prédiction de la classe d'âge des arbres.
    F3.py : Script pour l'entraînement et l'évaluation des modèles de classification.
    F3F4use.py : Script pour la prédiction des clusters d'arbres.
    Data_Arbre.csv : Fichier de données utilisé pour l'entraînement des modèles.
    centroids.csv : Fichier contenant les centroides des clusters calculés.
    predicted_cluster.json : Fichier contenant le résultat de la prédiction de cluster.
    clustered_trees_map.html : Carte interactive générée par le script F.py.
    models/ : Répertoire contenant les modèles et les scalers sauvegardés.

