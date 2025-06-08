import pandas as pd
import numpy as np

data = pd.read_csv("obesity_donnee.csv")

data['NObeyesdad'] = data['NObeyesdad'].apply(lambda x: 1 if str(x).lower() in ['obese', '1'] else 0)

features = data.columns.drop('NObeyesdad')
X = data[features].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
y = data['NObeyesdad'].values.astype(int)

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_norm = (X - X_min) / (X_max - X_min + 1e-8)  

input_size = X.shape[1]
hidden_size = 5 
output_size = 1
learning_rate = 0.05
epochs = 20

# Initialisation des poids et biais
np.random.seed(42)
W1 = np.random.randn(hidden_size, input_size) * 0.01
b1 = np.zeros((hidden_size, 1))
W2 = np.random.randn(output_size, hidden_size) * 0.01
b2 = np.zeros((output_size, 1))

# Fonction d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict_single(x):
    z1 = np.dot(W1, x) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(W2, a1) + b2
    a2 = sigmoid(z2)
    return a2

# Entraînement
for epoch in range(epochs):
    for i in range(len(X_norm)):
        x = X_norm[i].reshape(-1,1)
        y_true = y[i]

        z1 = np.dot(W1, x) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = sigmoid(z2)

        error = a2 - y_true

        dz2 = error * a2 * (1 - a2)
        dW2 = np.dot(dz2, a1.T)
        db2 = dz2

        dz1 = np.dot(W2.T, dz2) * a1 * (1 - a1)
        dW1 = np.dot(dz1, x.T)
        db1 = dz1

        # Mise à jour des poids et biais
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

correct = 0
for i in range(len(X_norm)):
    x = X_norm[i].reshape(-1,1)
    a2 = predict_single(x)
    pred = 1 if a2 >= 0.5 else 0
    if pred == y[i]:
        correct += 1
accuracy = correct / len(X_norm)
print(f"\nAccuracy sur l'ensemble d'entraînement : {accuracy*100:.2f}%")


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

def predict_new(x):
    x = np.array(x, dtype=float)
    x_norm = (x - X_min) / (X_max - X_min + 1e-8)
    x_norm = x_norm.reshape(-1,1)
    a2 = predict_single(x_norm)
    return 1 if a2 >= 0.5 else 0

print("\n--- Veuillez entrer les caractéristiques suivantes ---")
nouvel_exemple = []
for feature in features:
    val = demander_float(feature)
    nouvel_exemple.append(val)

prediction = predict_new(nouvel_exemple)

print("\nRésultat de la classification :")
if prediction == 1:
    print("=> Obèse")
else:
    print("=> Non obèse")
