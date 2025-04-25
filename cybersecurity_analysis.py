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
from sklearn.ensemble import IsolationForest
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
    
    print("\n\n--- ANOMALY DETECTION ---")
    
    anomaly_features = ['Financial Loss (in Million $)', 'Number of Affected Users', 'Incident Resolution Time (in Hours)']
    X_anomaly = df[anomaly_features].values

    scaler = StandardScaler()
    X_anomaly_scaled = scaler.fit_transform(X_anomaly)

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(X_anomaly_scaled)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    anomaly_count = df['anomaly'].sum()
    print(f"Nombre d'anomalies détectées: {anomaly_count} ({(anomaly_count/len(df))*100:.2f}% des données)")

    plt.figure(figsize=(12, 10))
    plt.scatter(df[df['anomaly']==0]['Number of Affected Users'], 
                df[df['anomaly']==0]['Financial Loss (in Million $)'],
                alpha=0.6, c='blue', s=40, label='Normal')
    plt.scatter(df[df['anomaly']==1]['Number of Affected Users'], 
                df[df['anomaly']==1]['Financial Loss (in Million $)'],
                alpha=1, c='red', s=120, edgecolors='black', label='Anomalie')
    plt.title('Détection d\'Anomalies: Incidents de Cybersécurité', fontsize=16)
    plt.xlabel('Nombre d\'Utilisateurs Affectés', fontsize=14)
    plt.ylabel('Pertes Financières (en Millions $)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'anomaly_detection.png'))

    anomaly_by_attack = df.groupby('Attack Type')['anomaly'].mean().sort_values(ascending=False) * 100
    plt.figure(figsize=(14, 8))
    anomaly_by_attack.plot(kind='bar')
    plt.title('Pourcentage d\'Anomalies par Type d\'Attaque', fontsize=16)
    plt.xlabel('Type d\'Attaque', fontsize=14)
    plt.ylabel('Pourcentage d\'Anomalies (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(visualization_dir, 'anomaly_by_attack_type.png'))

    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    df.to_csv(os.path.join(results_dir, 'cybersecurity_analyzed.csv'), index=False)

    print(f"\nDétection d'anomalies terminée. Résultats sauvegardés dans le dossier '{results_dir}'.")
    print("\nAnalyse complète du dataset de cybersécurité terminée!")
    
    print("\n\n--- INTERPRETATION AND SUMMARY ---")

    report_dir = "reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    cluster_summary = pd.DataFrame()

    cluster_summary = df.groupby('Cluster').agg({
        'Financial Loss (in Million $)': ['mean', 'min', 'max'],
        'Number of Affected Users': ['mean', 'min', 'max'],
        'Incident Resolution Time (in Hours)': ['mean', 'min', 'max'],
        'Year': ['mean', 'min', 'max']
    })

    dominant_attack = {}
    for cluster in df['Cluster'].unique():
        attack_counts = df[df['Cluster'] == cluster]['Attack Type'].value_counts()
        dominant_attack[cluster] = attack_counts.idxmax()

    target_industries = {}
    for cluster in df['Cluster'].unique():
        industry_counts = df[df['Cluster'] == cluster]['Target Industry'].value_counts()
        target_industries[cluster] = industry_counts.idxmax()

    defense_mechanisms = {}
    for cluster in df['Cluster'].unique():
        defense_counts = df[df['Cluster'] == cluster]['Defense Mechanism Used'].value_counts()
        defense_mechanisms[cluster] = defense_counts.idxmax()

    with open(os.path.join(report_dir, 'cybersecurity_report.txt'), 'w') as f:
        f.write("RAPPORT D'ANALYSE DES MENACES DE CYBERSÉCURITÉ\n")
        f.write("===========================================\n\n")
        
        f.write("1. RÉSUMÉ DES CLUSTERS IDENTIFIÉS\n")
        f.write("--------------------------------\n\n")
        
        for cluster in sorted(df['Cluster'].unique()):
            cluster_size = len(df[df['Cluster'] == cluster])
            f.write(f"CLUSTER {cluster} ({cluster_size} incidents, {cluster_size/len(df)*100:.1f}% des données)\n")
            f.write(f"- Type d'attaque dominant: {dominant_attack[cluster]}\n")
            f.write(f"- Industrie principale ciblée: {target_industries[cluster]}\n")
            f.write(f"- Mécanisme de défense typique: {defense_mechanisms[cluster]}\n")
            f.write(f"- Perte financière moyenne: {df[df['Cluster'] == cluster]['Financial Loss (in Million $)'].mean():.2f} millions $\n")
            f.write(f"- Utilisateurs affectés en moyenne: {int(df[df['Cluster'] == cluster]['Number of Affected Users'].mean())}\n")
            f.write(f"- Temps de résolution moyen: {int(df[df['Cluster'] == cluster]['Incident Resolution Time (in Hours)'].mean())} heures\n\n")
        
        f.write("2. ANALYSE DES ANOMALIES\n")
        f.write("------------------------\n\n")
        
        anomaly_count = df['anomaly'].sum()
        f.write(f"Nombre total d'anomalies détectées: {anomaly_count} ({anomaly_count/len(df)*100:.1f}% des incidents)\n\n")
        
        f.write("Top 5 des types d'attaques avec le plus d'anomalies:\n")
        anomaly_by_attack_pct = df.groupby('Attack Type')['anomaly'].mean().sort_values(ascending=False) * 100
        for attack_type, pct in anomaly_by_attack_pct.head(5).items():
            f.write(f"- {attack_type}: {pct:.1f}% d'anomalies\n")
        
        f.write("\n3. TENDANCES TEMPORELLES\n")
        f.write("------------------------\n\n")
        
        yearly_trends = df.groupby('Year').agg({
            'Financial Loss (in Million $)': 'mean',
            'Number of Affected Users': 'mean',
            'Incident Resolution Time (in Hours)': 'mean'
        }).round(2)
        
        f.write("Évolution des impacts par année:\n")
        f.write(f"{yearly_trends.to_string()}\n\n")
        
        f.write("4. RECOMMANDATIONS\n")
        f.write("------------------\n\n")
        
        f.write("Sur la base de cette analyse, voici les principales recommandations:\n\n")
        
        top_vulnerabilities = df.groupby('Security Vulnerability Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False).head(3)
        f.write("Vulnérabilités prioritaires à corriger:\n")
        for vuln, impact in top_vulnerabilities.items():
            f.write(f"- {vuln} (Impact financier moyen: {impact:.2f} millions $)\n")
        
        f.write("\nMécanismes de défense les plus efficaces:\n")
        defense_resolution = df.groupby('Defense Mechanism Used')['Incident Resolution Time (in Hours)'].mean().sort_values().head(3)
        for defense, time in defense_resolution.items():
            f.write(f"- {defense} (Temps de résolution moyen: {time:.1f} heures)\n")

    print(f"Rapport d'analyse généré dans {os.path.join(report_dir, 'cybersecurity_report.txt')}")

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 2, 1)
    cluster_counts = df['Cluster'].value_counts().sort_index()
    plt.pie(cluster_counts, labels=[f'Cluster {i}' for i in cluster_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
    plt.title('Distribution des Clusters', fontsize=14)

    plt.subplot(2, 2, 2)
    financial_by_attack = df.groupby('Attack Type')['Financial Loss (in Million $)'].mean().sort_values(ascending=False).head(5)
    financial_by_attack.plot(kind='bar')
    plt.title('Impact Financier par Type d\'Attaque', fontsize=14)
    plt.ylabel('Perte Moyenne (Millions $)')
    plt.xticks(rotation=45, ha='right')

    plt.subplot(2, 2, 3)
    yearly_attacks = df.groupby('Year').size()
    yearly_attacks.plot(kind='line', marker='o', linewidth=2)
    plt.title('Évolution du Nombre d\'Incidents par Année', fontsize=14)
    plt.xlabel('Année')
    plt.ylabel('Nombre d\'Incidents')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    defense_time = df.groupby('Defense Mechanism Used')['Incident Resolution Time (in Hours)'].mean().sort_values().head(5)
    defense_time.plot(kind='barh')
    plt.title('Efficacité des Mécanismes de Défense', fontsize=14)
    plt.xlabel('Temps de Résolution Moyen (Heures)')
    plt.tight_layout()

    plt.savefig(os.path.join(report_dir, 'dashboard_summary.png'))
    print(f"Tableau de bord récapitulatif généré dans {os.path.join(report_dir, 'dashboard_summary.png')}")
    
else:
    print("Aucun fichier CSV trouvé dans le dossier téléchargé.")