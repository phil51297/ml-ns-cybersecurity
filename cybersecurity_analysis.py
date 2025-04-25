import kagglehub
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm

visualization_dir = "visualizations"
if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)
    print(f"Dossier '{visualization_dir}' créé pour sauvegarder les graphiques")

print("Téléchargement du dataset...")
path = kagglehub.dataset_download("atharvasoundankar/global-cybersecurity-threats-2015-2024")
print(f"Dataset téléchargé dans: {path}")

csv_files = glob.glob(os.path.join(path, "*.csv"))
print(f"Fichiers trouvés: {csv_files}")

if csv_files:
    data_path = csv_files[0]
    print(f"Chargement des données depuis: {data_path}")
    df = pd.read_csv(data_path)
    
    print("\nAperçu des données:")
    print(df.head())
    
    print("\nInformations sur le dataset:")
    print(f"Nombre de lignes: {df.shape[0]}")
    print(f"Nombre de colonnes: {df.shape[1]}")
    print("\nTypes des colonnes:")
    print(df.dtypes)
    
    plt.style.use('ggplot')
    sns.set(font_scale=1.2)
    plt.rcParams['figure.figsize'] = (12, 8)
    
    print("\nVérification des valeurs manquantes:")
    print(df.isnull().sum())
    
    print("\nStatistiques descriptives des variables numériques:")
    print(df.describe())
    
    plt.figure(figsize=(14, 8))
    attack_counts = df['Attack Type'].value_counts()
    sns.barplot(x=attack_counts.index, y=attack_counts.values)
    plt.title('Distribution des Types d\'Attaques', fontsize=16)
    plt.xlabel('Type d\'Attaque', fontsize=14)
    plt.ylabel('Nombre d\'Incidents', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'attack_types_distribution.png'))
    
    plt.figure(figsize=(14, 8))
    attack_by_year = df.groupby(['Year', 'Attack Type']).size().unstack()
    attack_by_year.plot(kind='line', marker='o')
    plt.title('Évolution des Types d\'Attaques par Année', fontsize=16)
    plt.xlabel('Année', fontsize=14)
    plt.ylabel('Nombre d\'Incidents', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Type d\'Attaque', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'attack_evolution.png'))
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='Number of Affected Users', y='Financial Loss (in Million $)', 
                    hue='Attack Type', size='Incident Resolution Time (in Hours)',
                    sizes=(50, 200), alpha=0.7, data=df)
    plt.title('Pertes Financières vs Utilisateurs Affectés par Type d\'Attaque', fontsize=16)
    plt.xlabel('Nombre d\'Utilisateurs Affectés', fontsize=14)
    plt.ylabel('Pertes Financières (en Millions $)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'financial_loss_vs_users.png'))
    
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation = df[numeric_cols].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de Corrélation des Variables Numériques', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'correlation_matrix.png'))
    
    print(f"\nGraphiques sauvegardés dans le dossier '{visualization_dir}'.")
    
    print("\n\n--- UNSUPERVISED LEARNING ---")

    print("Préparation des données...")
    features = ['Year', 'Attack Type', 'Target Industry', 'Financial Loss (in Million $)', 
                'Number of Affected Users', 'Attack Source', 'Security Vulnerability Type',
                'Incident Resolution Time (in Hours)']

    numeric_features = ['Year', 'Financial Loss (in Million $)', 'Number of Affected Users', 
                        'Incident Resolution Time (in Hours)']
    categorical_features = ['Attack Type', 'Target Industry', 'Attack Source', 'Security Vulnerability Type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features), 
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    print("Application du préprocesseur...")
    X = preprocessor.fit_transform(df[features])
    print(f"Dimensions des données transformées: {X.shape}")

    print("\nApplication du clustering K-means...")
    inertias = []
    silhouette_scores = []
    range_n_clusters = range(2, 11)

    for n_clusters in range_n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
        if n_clusters > 1:
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(X, labels)
            silhouette_scores.append(silhouette_avg)
            print(f"Pour {n_clusters} clusters, le score de silhouette est : {silhouette_avg:.3f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range_n_clusters, inertias, marker='o')
    plt.title('Méthode du Coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Score de Silhouette')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de Silhouette')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'elbow_silhouette_method.png'))

    optimal_clusters = 4
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df['Cluster'] = clusters
    print(f"\nRépartition des clusters:\n{df['Cluster'].value_counts()}")

    print("\nApplication de PCA pour visualisation...")
    pca = PCA(n_components=2) 
    X_pca = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Variance expliquée par les 2 premières composantes: {explained_variance:.2%}")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', 
                         s=50, alpha=0.8, edgecolors='w')
    plt.title(f'Visualisation des {optimal_clusters} Clusters (PCA)', fontsize=16)
    plt.xlabel('Composante Principale 1', fontsize=14)
    plt.ylabel('Composante Principale 2', fontsize=14)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(visualization_dir, 'clusters_pca_visualization.png'))

    print("\nAnalyse des caractéristiques des clusters...")
    cluster_analysis = df.groupby('Cluster').agg({
        'Financial Loss (in Million $)': 'mean',
        'Number of Affected Users': 'mean',
        'Incident Resolution Time (in Hours)': 'mean',
        'Year': 'mean'
    }).round(2)

    print(cluster_analysis)

    attack_cluster = pd.crosstab(df['Cluster'], df['Attack Type'], normalize='index') * 100
    print("\nDistribution des types d'attaques par cluster (%):")
    print(attack_cluster.round(1))

    plt.figure(figsize=(14, 10))
    cluster_analysis.plot(kind='bar', ax=plt.gca())
    plt.title('Caractéristiques Moyennes par Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Valeur Moyenne', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Métrique', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'cluster_characteristics.png'))

    print("\nApprentissage non supervisé terminé. Tous les graphiques ont été sauvegardés.")
else:
    print("Aucun fichier CSV trouvé dans le dossier téléchargé.")