import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA


#Partie 1
def charger_dataset(chemin):
    df = pd.read_csv(chemin)
    print("\n===== Aperçu du dataset (10 premières lignes) =====")
    print(df.head(10))
    return df


def verifier_valeurs_manquantes(df):
    print("\n===== Valeurs manquantes =====")
    print(df.isnull().sum())


def statistiques_descriptives(df):
    print("\n===== Statistiques descriptives =====")
    print(df.describe(include="all"))

    colonnes_numeriques = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    df[colonnes_numeriques].hist(bins=20, figsize=(10, 6))
    plt.suptitle("Histogrammes des variables numériques")
    plt.show()


def selection_variables(df):
    print("\n===== Sélection des variables =====")
    colonnes_selectionnees = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    print(f"Variables retenues : {colonnes_selectionnees}")
    return df[colonnes_selectionnees]


def normaliser_donnees(df_selectionnees):
    scaler = StandardScaler()
    donnees_normalisees = scaler.fit_transform(df_selectionnees)

    print("\n= Données normalisées =  ")
    print(pd.DataFrame(donnees_normalisees, columns=df_selectionnees.columns).head(10))

    return donnees_normalisees, scaler


#Partie 2
def scatter_plots(df):
    print("\n===== Scatter Plots =====")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], alpha=0.7)
    plt.xlabel("Revenu annuel (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Revenu annuel vs Spending Score")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Age"], df["Annual Income (k$)"], alpha=0.7)
    plt.xlabel("Age")
    plt.ylabel("Revenu annuel (k$)")
    plt.title("Age vs Revenu annuel")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(df["Age"], df["Spending Score (1-100)"], alpha=0.7)
    plt.xlabel("Age")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Age vs Spending Score")
    plt.tight_layout()
    plt.show()


def pair_plot(df):
    print("\n===== Pairplot =====")
    sns.pairplot(df, height=2)
    plt.suptitle("Pairplot des variables sélectionnées", y=1.02)
    plt.show()


def heatmap_de_correlation(df):
    print("\n===== Heatmap de correlation =====")

    df_num = df.select_dtypes(include=["int64", "float64"])
    print(df_num.corr())

    plt.figure(figsize=(6, 4))
    sns.heatmap(df_num.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.show()



#Partie 3
def evaluer_kmeans(X_scaled):
    inertias = []
    silhouettes = []
    K_values = range(2, 11)

    print("\nÉvaluation K-Means (k = 2 → 10) \n")

    for k in K_values:
        kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    return K_values, inertias, silhouettes


def afficher_methodes_selection(K_values, inertias, silhouettes):
    plt.figure(figsize=(8, 5))
    plt.plot(K_values, inertias, marker='o')
    plt.title("Méthode du coude")
    plt.xlabel("k")
    plt.ylabel("Inertie (WCSS)")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(K_values, silhouettes, marker='o')
    plt.title("Score de silhouette")
    plt.xlabel("k")
    plt.ylabel("Silhouette")
    plt.grid(True)
    plt.show()


def entrainer_kmeans_final(df, X_scaled, K_values, silhouettes):
    best_k = K_values[silhouettes.index(max(silhouettes))]
    print(f"\n Meilleur k selon silhouette = {best_k}\n")

    final_kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    df["Cluster"] = final_kmeans.fit_predict(X_scaled)

    print("Clusters ajoutés au dataset :\n")
    print(df[["Age", "Annual Income (k$)", "Spending Score (1-100)", "Cluster"]].head())

    return best_k



#Partie 4
def interpretation_clusters(df):
    print("\n Interprétation des clusters \n")

    cluster_means = df.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean()
    print(cluster_means)

    print("\nProfils détectés\n")

    for cluster, row in cluster_means.iterrows():
        age, income, spend = row
        print(f"Cluster {cluster} :")
        print(f" - Âge moyen : {age:.1f}")
        print(f" - Revenu moyen : {income:.1f}")
        print(f" - Score de dépense : {spend:.1f}")

        if income > 60 and spend > 60:
            print(" ➤ VIP : Clients haut revenu & très dépensiers.\n")
        elif income > 60 and spend < 40:
            print(" ➤ Haut revenu mais faible dépense.\n")
        elif income < 40 and spend > 60:
            print(" ➤ Petits revenus mais très actifs.\n")
        elif spend < 40:
            print(" ➤ Clients peu engagés.\n")
        else:
            print(" ➤ Clients modérés.\n")


def visualisation_clusters(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="tab10")
    plt.title("Clusters (Revenu vs Spending Score)")
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(df["Age"], df["Annual Income (k$)"], df["Spending Score (1-100)"],
               c=df["Cluster"], cmap="tab10")
    ax.set_xlabel("Âge")
    ax.set_ylabel("Revenu (k$)")
    ax.set_zlabel("Spending Score")
    plt.title("Clusters en 3D")
    plt.show()

#Partie 5 (BONUS) : PCA + K-Means dans un espace réduit
def partie5_bonus(donnees_normalisees, df, best_k):
    print("\n===== Partie 5 : Bonus PCA + K-Means =====\n")

    # On réduit les données à 2 axes
    pca = PCA(n_components=2)
    donnees_reduites = pca.fit_transform(donnees_normalisees)

    variance_totale = pca.explained_variance_ratio_.sum()
    print(f"Part de variance gardée avec 2 axes : {variance_totale:.3f}")

    # K-Means dans cet espace réduit
    kmeans_bonus = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    etiquettes_bonus = kmeans_bonus.fit_predict(donnees_reduites)

    score_silhouette_bonus = silhouette_score(donnees_reduites, etiquettes_bonus)
    print(f"Silhouette dans l'espace réduit (k = {best_k}) : {score_silhouette_bonus:.3f}\n")

    # Préparer les données pour le graphique
    data_graph = df.copy()
    data_graph["axe1"] = donnees_reduites[:, 0]
    data_graph["axe2"] = donnees_reduites[:, 1]
    data_graph["Cluster_bonus"] = etiquettes_bonus

    # Nuage de points dans l'espace réduit
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data_graph,
        x="axe1",
        y="axe2",
        hue="Cluster_bonus",
        palette="tab10"
    )
    plt.title("Clusters K-Means dans l'espace réduit (PCA)")
    plt.tight_layout()
    plt.show()


#Partie 5 (BONUS) : CAH (Classification Ascendante Hiérarchique)
def partie5_cah(donnees_normalisees, df, best_k):
    print("\n===== Partie 5 : Bonus CAH (Agglomératif) =====\n")

    # Modèle de CAH avec la même valeur de k que K-Means
    modele_cah = AgglomerativeClustering(
        n_clusters=best_k,
        linkage="ward"
    )

    etiquettes_cah = modele_cah.fit_predict(donnees_normalisees)

    score_silhouette_cah = silhouette_score(donnees_normalisees, etiquettes_cah)
    print(f"Silhouette pour la CAH (k = {best_k}) : {score_silhouette_cah:.3f}\n")

    # Préparer les données pour visualisation
    data_cah = df.copy()
    data_cah["Cluster_cah"] = etiquettes_cah

    # Moyennes par cluster (CAH)
    print("Moyennes des variables par cluster (CAH) :\n")
    print(data_cah.groupby("Cluster_cah")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())

    # Nuage de points Revenu vs Score de dépense
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data_cah,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Cluster_cah",
        palette="tab10"
    )
    plt.title("Clusters CAH (Revenu vs Spending Score)")
    plt.tight_layout()
    plt.show()







def main():
    #Partie 1
    df = charger_dataset("Mall_Customers.csv")
    verifier_valeurs_manquantes(df)
    statistiques_descriptives(df)
    df_selectionnees = selection_variables(df)
    donnees_normalisees, scaler = normaliser_donnees(df_selectionnees)

    print("\n=== Partie 1 terminée ===")

    #Partie 2
    scatter_plots(df_selectionnees)
    pair_plot(df_selectionnees)
    heatmap_de_correlation(df)

    print("\n=== Partie 2 terminée ===")

    #Partie 3
    K_values, inertias, silhouettes = evaluer_kmeans(donnees_normalisees)
    afficher_methodes_selection(K_values, inertias, silhouettes)
    best_k = entrainer_kmeans_final(df, donnees_normalisees, K_values, silhouettes)

    #Partie 4
    interpretation_clusters(df)
    visualisation_clusters(df)

    #Partie 5 PCA + K-Means
    partie5_bonus(donnees_normalisees, df, best_k)

    #Partie 5  CAH
    partie5_cah(donnees_normalisees, df, best_k)


if __name__ == "__main__":
    main()

