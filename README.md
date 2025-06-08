# 🧠 Obesity Prediction avec Perceptron (CLI)

Ce projet implémente un modèle de **Perceptron binaire** pour prédire si une personne est **obèse** ou **non obèse** à partir de caractéristiques personnelles et comportementales.

---

## 📂 Jeu de données

Le fichier `obesity_donnee.csv` contient les colonnes suivantes :

- `Gender` : Genre (0 = Femme, 1 = Homme)
- `Age` : Âge (ex. : 25)
- `family_history_with_overweight` : Antécédents familiaux d'obésité
- `FAVC` : Alimentation hypercalorique
- `FCVC` : Fréquence de consommation de légumes
- `NCP` : Nombre de repas par jour
- `CAEC` : Grignotage entre les repas
- `SMOKE` : Fume
- `CH2O` : Consommation d'eau quotidienne
- `SCC` : Surveillance des calories
- `FAF` : Activité physique hebdomadaire
- `TUE` : Temps d’écran quotidien
- `CALC` : Consommation d’alcool
- Moyens de transport : `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, `Walking`
- `NObeyesdad` : Variable cible (1 = Obèse, 0 = Non Obèse)

---

## 🧠 Algorithme

L’algorithme repose sur un **Perceptron** entraîné à partir du jeu de données avec :

- Taux d’apprentissage : `0.01`
- Époques (epochs) : `30`
- Fonction d’activation : `step function (seuil)`
- Prédiction finale : 1 (obèse) ou 0 (non obèse)

---

## 🖥️ Fonctionnalité

Après l'entraînement, l'utilisateur est invité à **saisir ses propres caractéristiques** (âge, genre, habitudes...) via le terminal. Le programme effectue alors une prédiction :

```bash
Résultat de la classification :
=> Obèse
ou
=> Non obèse