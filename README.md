
Ce document présente le travail réalisé dans le cadre d’un TP consacré à la segmentation de clients à l’aide de l’algorithme K-Means.
L’objectif du projet était de comprendre comment un centre commercial peut regrouper ses clients en fonction de leurs caractéristiques. L'approche adoptée a été simple : observer les données, appliquer le clustering et essayer d'interpréter les résultats de manière claire et logique.

1. Présentation du dataset
Le dataset utilisé, *Mall Customers*, contient 200 clients avec plusieurs informations :
- Âge
- Genre
- Revenu annuel (k$)
- Spending Score (score d’engagement entre 1 et 100)
Pour la partie clustering, seules trois variables ont été conservées : l’âge, le revenu annuel et le spending score. Ces variables sont les plus pertinentes pour distinguer des groupes cohérents.

2. Exploration initiale des données
Avant d’appliquer K-Means, une analyse rapide des données a été réalisée :
- Histogrammes pour observer la distribution
  ![Histogrammes](images/Figure_1.png)
- Scatter plots pour repérer d’éventuels regroupements
  ![Age vs Income](images/Figure_2.png)
  ![Age vs Spending Score](images/Figure_3.png)
  ![Income vs Spending Score](images/Figure_4.png)
- Pairplot
  ![Pairplot](images/Figure_5.png)
- Matrice de corrélation pour voir s’il existe des liens entre variables
  ![Correlation Matrix](images/Figure_6.png)
Cette première étape a permis d’identifier certaines tendances, par exemple des groupes de clients dépensiers ou à fort revenu.

3. Application de K-Means
L’algorithme K-Means nécessite de choisir un nombre de clusters k. Plusieurs valeurs ont été testées (de 2 à 10), comme demandé dans le TP.
Pour choisir le meilleur k, deux méthodes ont été utilisées :
- La méthode du coude
  ![Méthode de coude](images/Figure_7.png)
- Le score de silhouette
  ![Silhouette Score](images/Figure_8.png)
Une fois k déterminé, le modèle final a été entraîné et une colonne “Cluster” a été ajoutée au dataset pour identifier le segment de chaque client.

4. Interprétation des clusters
Après l’obtention des clusters, les moyennes de chaque groupe ont été analysées (âge moyen, revenu moyen, score moyen).
Cela a permis de dégager différents types de clients, par exemple :
- Jeunes dépensiers
- Clients avec haut revenu mais faible dépense
- Clients plus âgés et modérés
- Groupes équilibrés avec revenu moyen et dépenses moyennes
Cette étape a donné du sens aux regroupements observés dans les données.

5. Visualisations des résultats
Plusieurs graphiques ont été générés pour mieux comprendre et illustrer les clusters :
- Visualisation 2D (Revenu vs Spending Score)
  ![Clusters 2D](images/Figure_9.png)
- Visuel 3D avec âge, revenu et score
  ![Clusters 3D](images/Figure_ç.png)
- PCA pour réduire la dimension et visualiser les clusters autrement
  ![Clusters PCA](images/Figure_11.png)
- Comparaison avec la CAH (clustering hiérarchique)
  ![Clusters CAH](images/Figure_12.png)
Ces visualisations facilitent la compréhension des segments obtenus
