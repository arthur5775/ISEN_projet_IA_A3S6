import sys
import joblib
import json
import numpy as np
import pandas as pd

def age(method, haut_tot, haut_tronc, tronc_diam, fk_stadedev):
    # On charge le modèle choisis avec le fichier .pkl qui correspond
    model = joblib.load(f'{method}_model.pkl')
    
    # On reprend l'encodeur et la normalisation de la fonctionnalité 2 
    encoder = joblib.load('encoder.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # On vérifie que la valeur fk_stadedev correspond bien à une des valeurs attribuées
    valeur_colonne = ['Jeune', 'Adulte', 'Vieux', 'senescent']
    if fk_stadedev not in valeur_colonne:
        raise ValueError(f"La valeur de fk_stadedev doit être parmi {valeur_colonne}.")
    
    # On créé un tableau avec les données des arbres qu'on a rentré dans le terminal
    arbre = np.array([[float(haut_tot), float(haut_tronc), float(tronc_diam)]])

    # On créé un DataFrame pour l'encodage de fk_stadedev
    col_cat = pd.DataFrame({'fk_stadedev': [fk_stadedev]})
    
    # On encoder la variable catégorielle fk_stadedev
    fk_stadedev_encode = encoder.transform(col_cat[['fk_stadedev']])
    
    # On concaténe fk_stadedev_encoded avec arbre
    arbre_encode = np.hstack((arbre, fk_stadedev_encode))

    # On standardise les données de l'arbre
    arbre_scaled = scaler.transform(arbre_encode)
    
    # On fait la prédiction
    pred = model.predict(arbre_scaled)
    return int(pred[0]) 

if __name__ == "__main__":
    # On récupère les arguments de la ligne de commande
    if len(sys.argv) != 6:
        print("python F2F4use.py method haut_tot haut_tronc tronc_diam fk_stadedev")
        sys.exit(1)
    
    method = sys.argv[1]
    haut_tot = sys.argv[2]
    haut_tronc = sys.argv[3]
    tronc_diam = sys.argv[4]
    fk_stadedev = sys.argv[5]
    
    # On appelle la fonction age avec les arguments de la ligne de commande
    try:
        prediction = age(method, haut_tot, haut_tronc, tronc_diam, fk_stadedev)
        print(f"Classe d'âge prédite: {prediction}")
        
        # On enregistre la prédiction dans un fichier JSON
        with open('class_age.json', 'w') as json_file:
            json.dump(int(prediction), json_file)
            
    except ValueError as e:
        print(e)
        sys.exit(1)


