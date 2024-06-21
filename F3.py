import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import joblib
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

# Lecture du fichier CSV
data = pd.read_csv("Data_Arbre.csv")

# Affichage des données
data = data.map(lambda x: x.lower() if type(x) == str else x)

# Crée une liste de déracinés
deracines = []

#print(data["fk_situation"])

# affichage des données
for e in data["fk_arb_etat"]:
    if e == "essouché" or e == "non essouché":
        deracines.append(1)
    else:
        deracines.append(0)

# Encodage des variables catégorielles, cela permet de les transformer en valeurs numériques
data[["clc_secteur"]] = OrdinalEncoder().fit_transform(data[["clc_secteur"]])
data[["fk_pied"]] = OrdinalEncoder().fit_transform(data[["fk_pied"]])

"""
# Diviser les données en ensembles d'entraînement et de test
train_data = data.drop('fk_arb_etat', axis=1).values[:6000]
train_labels = deracines[:6000]
test_data = data.drop('fk_arb_etat', axis=1).values[6000:]
test_labels = deracines[6000:]
"""

# Suppression des colonnes inutiles
data = data.drop(
    ['clc_quartier', 'haut_tronc', 'fk_arb_etat', 'fk_stadedev', 'fk_port', 'fk_pied', 'fk_situation', 'fk_revetement',
     'fk_prec_estim',
     'clc_nbr_diag', 'fk_nomtech', 'villeca', 'feuillage', 'remarquable'], axis=1)
print(data)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(data, deracines, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
train_data_normalize = scaler.fit_transform(train_data, train_labels)
test_data_normalize = scaler.fit_transform(test_data, train_labels)

# Entraînement des modèles
sgdc_classifier_normalize = SGDClassifier()
sgdc_classifier_normalize.fit(train_data_normalize, train_labels)

sgdc_classifier = SGDClassifier()
sgdc_classifier.fit(train_data, train_labels)

rd_forest_classifier = RandomForestClassifier()
rd_forest_classifier.fit(train_data, train_labels)

rd_forest_classifier_normalize = RandomForestClassifier()
rd_forest_classifier_normalize.fit(train_data_normalize, train_labels)

# Validation croisée
score = cross_val_score(sgdc_classifier, train_data, train_labels, cv=3, scoring="accuracy")
score2 = cross_val_score(rd_forest_classifier, train_data, train_labels, cv=3, scoring="accuracy")

val_predict = cross_val_predict(sgdc_classifier, train_data, train_labels, cv=3)
matrix = confusion_matrix(train_labels, val_predict)

# Affichage de la matrice de confusion
disp1 = ConfusionMatrixDisplay(matrix, display_labels=np.unique(train_labels))

# Normalisation de la matrice de confusion
matrix_normalized = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
disp2 = ConfusionMatrixDisplay(confusion_matrix=matrix_normalized, display_labels=np.unique(train_labels))

print(matrix)
print("score moyen sgdc classifier", np.mean(score))
print("scores sgdc classifier", score)
print("score moyen scalaire", np.mean(score2))
print("scores scalaire", score2)

score = cross_val_score(sgdc_classifier, train_data, train_labels, cv=3, scoring="accuracy")
score2 = cross_val_score(sgdc_classifier, train_data_normalize, train_labels, cv=3, scoring="accuracy")

# Calcul de la précision
precision = precision_score(train_labels, val_predict, average='weighted')

# Calcul du rappel
recall = recall_score(train_labels, val_predict, average='weighted')

# Calcul du score F1
f1 = f1_score(train_labels, val_predict, average='weighted')

# Affichage des résultats
print("Precision :", precision)
print("Rappel :", recall)
print("Score F1 :", f1)

# Calcul des probabilités de prédiction
val_predict_decision = cross_val_predict(sgdc_classifier, train_data_normalize, train_labels, cv=3,
                                         method="decision_function")

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(train_labels, val_predict_decision)

# Calcul de l'AUC, l'aire sous la courbe ROC
roc_auc = roc_auc_score(train_labels, val_predict_decision)

# Affichage de la courbe ROC
plt.figure()
plt.plot(fpr, tpr, label=f'Courbe ROC (aire = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Taux de faux positif')
plt.ylabel('Taux de vrai positif')
plt.title('Courbe de ROC')
plt.legend(loc="lower right")
plt.show()

# Affichage de l'AUC
print(f'AUC: {roc_auc:.2f}')

print("Coefficients:", sgdc_classifier.coef_)

# Calcul des probabilités de prédiction
val_predict_decision = cross_val_predict(rd_forest_classifier, train_data_normalize, train_labels, cv=3)

# Calcul de la courbe ROC
fpr, tpr, thresholds = roc_curve(train_labels, val_predict_decision)

# Calcul de l'AUC
roc_auc = roc_auc_score(train_labels, val_predict_decision)

# Affichage de la courbe ROC
plt.figure()
plt.plot(fpr, tpr, label=f'Courbe ROC (aire = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Taux de faux positif')
plt.ylabel('Taux de vrai positif')
plt.title('Courbe de ROC')
plt.legend(loc="lower right")
plt.show()

# Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_features': [2, 3, 4, 5]
}

grid_search = GridSearchCV(rd_forest_classifier_normalize, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(train_data_normalize, train_labels)

meilleur_modele = grid_search.best_estimator_
print(meilleur_modele)

# 3. Métriques pour la classification
# Prédictions
y_pred = meilleur_modele.predict(test_data_normalize)

# Matrice de confusion
print("Matrice de confusion:")
print(confusion_matrix(test_labels, y_pred))

# Rapport de classification
print("\nRapport de classification:")
print(classification_report(test_labels, y_pred))

# Courbe ROC 
y_score = meilleur_modele.predict_proba(test_data_normalize)

fpr = dict()
tpr = dict()
roc_auc = dict()

fpr, tpr, thresholds = roc_curve(test_labels, y_score[:, -1])
roc_auc = roc_auc_score(test_labels, y_score[:, -1])

# Tracer la courbe ROC
plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'Classe {1} (aire = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbe ROC multi-classe')
plt.legend(loc="lower right")
plt.show()

# Affichage des paramètres optimaux
print("\nMeilleurs hyperparametres:")
print(grid_search.best_params_)

print("Usage: python F3F4use.py <longitude>   <latitude>  <clc_secteur>  <haut_tot>  <tronc_diam>  <age_estim>")

model_path = os.path.join('models', 'RandomForest_model.pkl')

joblib.dump(meilleur_modele, model_path)


model_path = os.path.join('models', 'scaler2.pkl')

joblib.dump(scaler, model_path)