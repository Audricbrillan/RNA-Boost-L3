import pandas as pd
import numpy as np

# Lecture des données
data = pd.read_csv("obesity_donnee.csv")

data['NObeyesdad'] = data['NObeyesdad'].apply(lambda x: 1 if str(x).lower() in ['obese', '1'] else 0)

features = data.columns.drop('NObeyesdad')
X = data[features].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
y = data['NObeyesdad'].values.astype(int)

# des poids et du biais
w = np.zeros(len(features))
b = 0
learn_rate = 0.01
epochs = 30

# Fonction d'activation
def activation(s):
    if s >= 0 :
        return 1  
    else :
        return 0

# Fonction perceptron
def perceptron(x, w, b):
    s = np.dot(w, x) + b
    return activation(s)

# Entraînement
for epoch in range(epochs):
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]
        pred = perceptron(x_i, w, b)
        error = y_i - pred
        w += learn_rate * error * x_i
        b += learn_rate * error

descriptions = {
    "Gender": "Genre (0 = Femme, 1 = Homme)",
    "Age": "Âge (ex: 25)",
    "family_history_with_overweight": "Antécédents familiaux d'obésité (0 = Non, 1 = Oui)",
    "FAVC": "Alimentation hypercalorique (0 = Non, 1 = Oui)",
    "FCVC": "Fréquence de consommation de légumes (0 à 3)",
    "NCP": "Nombre de repas principaux par jour (1 à 4)",
    "CAEC": "Grignotage entre les repas (0 = Jamais, 3 = Toujours)",
    "SMOKE": "Fume (0 = Non, 1 = Oui)",
    "CH2O": "Consommation d'eau par jour (0 à 3)",
    "SCC": "Surveillance des calories (0 = Non, 1 = Oui)",
    "FAF": "Activité physique hebdo (0 à 3)",
    "TUE": "Temps devant l'écran par jour (0 à 2)",
    "CALC": "Consommation d'alcool (0 = Jamais, 3 = Toujours)",
    "Automobile": "Utilise une voiture (0 = Non, 1 = Oui)",
    "Bike": "Utilise un vélo (0 = Non, 1 = Oui)",
    "Motorbike": "Utilise une moto (0 = Non, 1 = Oui)",
    "Public_Transportation": "Utilise les transports en commun (0 = Non, 1 = Oui)",
    "Walking": "Marche régulièrement (0 = Non, 1 = Oui)"
}

def demander_float(nom):
    while True:
        try:
            print(f"\n{nom} : {descriptions.get(nom, 'Valeur attendue')}")
            val = float(input("Entrez la valeur correspondante : "))
            return val
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre.")

# Demander les données utilisateur
print("\n--- Veuillez entrer les caractéristiques suivantes ---")
nouvel_exemple = []
for feature in features:
    val = demander_float(feature)
    nouvel_exemple.append(val)

nouvel_exemple = np.array(nouvel_exemple, dtype=float)
prediction = perceptron(nouvel_exemple, w, b)

print("\nRésultat de la classification :")
if prediction == 1:
    print("=> Obèse")
else:
    print("=> Non obèse")
