# ============================================================
# TITANIC - Comparaison de 4 modèles (KNN, SVM, LogReg, MLP)
# Objectif:
#   1) Préparer le dataset Titanic (nettoyage + encodage)
#   2) Entraîner 4 modèles
#   3) Comparer leurs performances (Accuracy, Precision, Recall, F1, RMSE)
#   4) Mesurer le temps d'entraînement et de prédiction
#   5) Visualiser les résultats avec des graphes clairs
#   6) Étudier la "sensibilité" (robustesse) quand on ajoute du bruit
# ============================================================

# -----------------------------
# 1) Importation des bibliothèques
# -----------------------------
import time                      # Pour mesurer les temps d'exécution (train/predict)
import numpy as np               # Pour calculs numériques (RMSE, bruit aléatoire...)
import pandas as pd              # Pour lire / manipuler le dataset CSV
import matplotlib.pyplot as plt  # Pour créer les graphiques

from sklearn.model_selection import train_test_split  # Split train/test

from sklearn.preprocessing import StandardScaler      # Normalisation (important pour KNN/SVM/MLP/LogReg)
from sklearn.pipeline import Pipeline                 # Pipeline = scaling -> modèle

# Modèles ML utilisés
from sklearn.neighbors import KNeighborsClassifier    # KNN : basé sur distances
from sklearn.svm import SVC                           # SVM : marge maximale
from sklearn.linear_model import LogisticRegression    # Régression logistique : modèle linéaire + probabilités
from sklearn.neural_network import MLPClassifier       # Réseau de neurones (perceptron multicouche)

# Métriques pour évaluer les modèles
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ============================================================
# 2) Paramètres généraux
# ============================================================
CSV_PATH = "../data/Titanic-Dataset.csv"  # chemin vers ton dataset Titanic
TEST_SIZE = 0.2                          # 20% test / 80% train
RANDOM_STATE = 42                        # pour avoir des résultats reproductibles

# ============================================================
# 3) Chargement des données
# ============================================================
df = pd.read_csv(CSV_PATH)

# ============================================================
# 4) Nettoyage (prétraitement classique Titanic)
# ============================================================
# Suppression de colonnes qui n'apportent pas de valeur prédictive claire,
# ou qui ont trop de valeurs manquantes (Cabin), ou trop spécifiques (Name, Ticket).
df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True, errors="ignore")

# Remplissage des valeurs manquantes :
# Age -> médiane (robuste aux valeurs extrêmes)
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Fare -> médiane
if "Fare" in df.columns:
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Embarked -> mode (la valeur la plus fréquente)
if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encodage des colonnes catégorielles en numérique (One-Hot Encoding)
# Exemple: Sex -> Sex_male (si drop_first=True)
# drop_first=True évite une redondance (dummy trap).
cat_cols = [c for c in ["Sex", "Embarked"] if c in df.columns]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ============================================================
# 5) Séparation Features / Target
# ============================================================
target = "Survived"          # ce qu'on veut prédire
X = df.drop(columns=[target])  # toutes les colonnes sauf la cible
y = df[target]                 # la cible 0/1

# ============================================================
# 6) Split Train / Test
# stratify=y garde la proportion (survivants/non survivants) identique dans train et test
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# ============================================================
# 7) Définition des modèles
# StandardScaler est utile pour KNN, SVM, LogReg, MLP :
# car ces modèles sont sensibles à l'échelle des variables (Age vs Fare etc.)
# ============================================================
models = {
    "KNN (k=5)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ]),
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "NeuralNet (MLP)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=400,
            random_state=RANDOM_STATE
        ))
    ]),
}

# ============================================================
# 8) Fonction d'évaluation d'un modèle
# ============================================================
def evaluate_model(name, model):
    """
    Cette fonction:
      - entraîne un modèle
      - prédit sur X_test
      - calcule les métriques principales
      - mesure les temps train et predict
    """
    # ----- Mesure du temps d'entraînement -----
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # ----- Mesure du temps de prédiction -----
    t1 = time.perf_counter()
    y_pred = model.predict(X_test)
    pred_time = time.perf_counter() - t1

    # ----- Calcul des métriques -----
    acc = accuracy_score(y_test, y_pred)                 # % de bonnes prédictions
    f1 = f1_score(y_test, y_pred)                        # équilibre précision/rappel
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))   # erreur moyenne quadratique (0/1)
    precision = precision_score(y_test, y_pred)          # parmi prédits "1", combien sont vrais
    recall = recall_score(y_test, y_pred)                # parmi vrais "1", combien détectés (sensibilité)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "RMSE": rmse,
        "TrainTime(s)": train_time,
        "PredTime(s)": pred_time,
        "y_pred": y_pred
    }

# ============================================================
# 9) Exécuter tous les modèles et stocker résultats
# ============================================================
results = []
pred_store = {}

for name, model in models.items():
    r = evaluate_model(name, model)
    pred_store[name] = r.pop("y_pred")  # garder predictions séparées pour confusion matrix
    results.append(r)

# DataFrame des résultats (trié par accuracy)
results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)

print("\n===== Résultats (Titanic) =====")
print(results_df.to_string(index=False))

# ============================================================
# 10) Graphique 1 : Comparaison des métriques (barres + valeurs)
# ============================================================
# On affiche Accuracy, Precision, Recall, F1 sur une même figure
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1"]

plt.figure(figsize=(10, 5))
x = np.arange(len(results_df["Model"]))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    plt.bar(x + (i - 1.5) * width, results_df[metric], width=width, label=metric)

# Ajouter les valeurs au-dessus des barres (plus lisible)
for i, metric in enumerate(metrics_to_plot):
    for j, val in enumerate(results_df[metric]):
        plt.text(j + (i - 1.5) * width, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)

plt.xticks(x, results_df["Model"], rotation=20, ha="right")
plt.title("Comparaison des métriques principales (Titanic)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# 11) Graphique 2 : Temps d'exécution (train vs predict)
# ============================================================
plt.figure(figsize=(8, 4))
plt.bar(results_df["Model"], results_df["TrainTime(s)"], label="Train time")
plt.bar(results_df["Model"], results_df["PredTime(s)"], label="Predict time")
plt.title("Temps d'entraînement vs prédiction")
plt.ylabel("Secondes")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.show()



# ============================================================
# 12) Graphique 3 : Sensibilité au bruit (robustesse)
# Idée: on ajoute du bruit aux colonnes numériques et on regarde si l'accuracy chute.
# ============================================================
num_cols = X_test.select_dtypes(include=[np.number]).columns.tolist()

def add_noise(X, noise_level=0.05):
    """
    Ajoute un bruit gaussien sur chaque colonne numérique.
    noise_level = 0.05 signifie bruit ~ 5% de l'écart-type de la feature.
    """
    Xn = X.copy()
    for c in num_cols:
        std = Xn[c].std()
        if std == 0 or np.isnan(std):
            continue
        Xn[c] = Xn[c] + np.random.normal(0, noise_level * std, size=len(Xn))
    return Xn

noise_levels = [0.0, 0.02, 0.05, 0.10]

plt.figure(figsize=(9, 5))
for name, model in models.items():
    # On ré-entraîne une fois sur le train normal
    model.fit(X_train, y_train)

    acc_list = []
    for nl in noise_levels:
        X_test_noisy = add_noise(X_test, nl)
        y_pred_noisy = model.predict(X_test_noisy)
        acc_list.append(accuracy_score(y_test, y_pred_noisy))

    # Courbe (avec marqueurs + valeur base)
    plt.plot(noise_levels, acc_list, marker="o", label=f"{name} (base={acc_list[0]:.2f})")

plt.title("Sensibilité au bruit (plus la courbe chute, plus le modèle est sensible)")
plt.xlabel("Niveau de bruit ajouté")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



