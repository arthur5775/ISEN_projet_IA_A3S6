import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import classification_report

def fonctionnalite2(features, methode):
    # Lecture du fichier CSV
    data_arbre = pd.read_csv("Data_Arbre.csv")

    # Sélection des colonnes pertinentes
    cible = 'age_estim'
    X = data_arbre[features]
    y_data = data_arbre[cible]

    # Création des catégories pour l'âge
    liste_age = [0, 20, 40, 60, 80, 100, 250]
    y = np.digitize(y_data, liste_age)

    # Identifier les colonnes catégorielles
    categorical_columns = ['fk_stadedev']

    # Création de l'encodeur
    encoder = OneHotEncoder(categories='auto', sparse_output=False,handle_unknown='ignore')

    # On transforme les données catégorielles
    X_encoded = encoder.fit_transform(X[categorical_columns])

    # On supprime l'ancienne colonne catégorielle et on la remplace par la nouvelle encodé
    X = X.drop(columns=['fk_stadedev'])
    X = np.hstack((X, X_encoded))

    # On normalise les données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # On sauvegarde l'encodeur et la normalisation dans un fichier .pkl pour l'utiliser dans la fonctionnalité 4
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    # On divise les données en entrainements et tests
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Choix du modèle
    if methode == 'RandomForest':
        clf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': [2, 3, 4, 5]
        }
    elif methode == 'DecisionTree':
        clf = DecisionTreeClassifier()
        param_grid = {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8, 16]
        }
    elif methode == 'KNeighbors':
        clf = KNeighborsClassifier()
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 30, 50],
            'p': [1, 2]
        }
    else:
        raise ValueError("Méthode non supportée. Choisissez parmi 'RandomForest', 'DecisionTree', ou 'KNeighbors'.")

    # GridSearchCV
    gsc = GridSearchCV(clf, param_grid, scoring='accuracy', cv=5, n_jobs=-1)
    gsc.fit(X_train, y_train)
    meilleur_modele = gsc.best_estimator_
    print(meilleur_modele)
    # Affichage des meilleurs parametres
    print("Meilleurs parametres:", gsc.best_params_)

    # On sauvegarde le modèle dans un fichier .pkl pour l'utiliser dans la foncionnalité 4 
    joblib.dump(meilleur_modele, f'{methode}_model.pkl')

    # On fait la prédiction
    y_pred = meilleur_modele.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test,y_pred)

    #On affiche les métriques
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall score :", recall)
    print("Classification report:",class_report)

    #On affiche la matrice de confusion
    matrice = confusion_matrix(y_test, y_pred)
    print(matrice)

    # Tracé de la matrice de confusion sur un graphique
    plt.figure(figsize=(10, 7))
    sns.heatmap(matrice, annot=True, fmt='d', cmap='Blues', xticklabels=liste_age, yticklabels=liste_age)
    plt.xlabel('Predit')
    plt.ylabel('Actuel')
    plt.title('Matrice de Confusion')
    plt.show()

    # Courbe ROC
    y_score = meilleur_modele.predict_proba(X_test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = len(np.unique(y))

    # Calcule le taux de vrais positifs et faux positifs pour la classe i
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test, y_score[:, i], pos_label=i + 1)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # On trace les courbes ROC pour chaque classe
    plt.figure()
    couleurs = ['blue', 'orange', 'red', 'green', 'brown', 'pink']
    for i, color in zip(range(n_classes), couleurs):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Classe {i + 1} (aire = {roc_auc[i]:.2f})')

    # On l'affiche sur un graphique
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC multi-classe')
    plt.legend(loc="lower right")
    plt.show()

#On lance la fonction pour chaque méthode
fonctionnalite2(features=['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev'],methode='RandomForest')
fonctionnalite2(features=['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev'],methode='DecisionTree')
fonctionnalite2(features=['haut_tot', 'haut_tronc', 'tronc_diam', 'fk_stadedev'],methode='KNeighbors')