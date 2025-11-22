import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



df = pd.read_csv('Mall_Customers.csv')

#Afficher les 5 premières lignes du dataframe
print(df.info())
print("------------------------------------ \n")
#Afficher les colonnes et les valeurs manquantes
print(df.head())
print("------------------------------------ \n")

#Afficher le nombre de valeurs manquantes de chaque colonnes
print(df.isnull().sum())


# Analyse de la distribution des variables
#Afficher les statistiques descriptives
print("\nStatistiques descriptives :\n")
print(df.describe())
print("\n---------------------------------\n")

#Visualiser les distributions des variables numériques
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
plt.figure(figsize=(15, 4))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, 3, i)
    plt.hist(df[col], bins=15, edgecolor='black')
    plt.title(f"Distribution de {col}")
    plt.xlabel(col)
    plt.ylabel("Fréquence")

plt.tight_layout()
plt.show()


#Sélection des variables pertinentes
#On garde seulement les colonnes utiles au clustering
features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]
print("\nVariables retenues pour la segmentation :")
print(features)
print("\nAperçu du dataset réduit :\n")
print(X.head())

#Normalisation
scaler = StandardScaler()
# On transforme notre dataset X
X_scaled = scaler.fit_transform(X)
print("\nAperçu des données normalisées :\n")
print(X_scaled[:5])


#Pairplot (relations entre variables)
print("Génération du pairplot (relations Age / Income / Spending Score)...")

sns.pairplot(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]],
    diag_kind="kde"
)
plt.show()

#Heatmap de corrélation
print("\nAffichage de la heatmap des corrélations\n")
plt.figure(figsize=(6, 4))
sns.heatmap(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Matrice de corrélation")
plt.show()

#Hypothèses sur les clusters (à mettre dans le rapport)
print("\nPremières observations et hypothèses :")
print(
    "- Les graphiques montrent des regroupements visibles dans l’espace Revenu / Score de dépense.\n"
    "- On observe plusieurs groupes distincts, notamment :\n"
    "   • Clients haut revenu / faible dépense\n"
    "   • Clients revenu moyen / forte dépense\n"
    "   • Clients jeunes dépensiers\n"
    "   • Clients plus âgés et peu dépensiers\n"
    "- On peut raisonnablement supposer entre 4 et 6 clusters.\n"
)


#Tester plusieurs valeurs de k (de 2 à 10)
inertias = []
silhouettes = []
K_values = range(2, 11)

print("Calcul des inerties et des silhouettes pour k = 2 à 10...\n")

for k in K_values:
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

print("Terminé !\n")

#Méthode du coude
plt.figure(figsize=(8, 5))
plt.plot(K_values, inertias, marker='o')
plt.title("Méthode du coude (Inertie / WCSS)")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie (WCSS)")
plt.xticks(K_values)
plt.grid(True)
plt.show()


#Score de silhouette
plt.figure(figsize=(8, 5))
plt.plot(K_values, silhouettes, marker='o')
plt.title("Score de silhouette selon k")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Silhouette score")
plt.xticks(K_values)
plt.grid(True)
plt.show()

#Choix du meilleur k
best_k = K_values[silhouettes.index(max(silhouettes))]
print(f"\n>>> Meilleur k selon la silhouette = {best_k}\n")


#Modèle final
print(f"Entraînement du modèle K-Means avec k = {best_k}...\n")
final_kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
df["Cluster"] = final_kmeans.fit_predict(X_scaled)
print("Modèle final entraîné ! Voici un aperçu :\n")
print(df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

