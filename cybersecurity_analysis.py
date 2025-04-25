import kagglehub
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
else:
    print("Aucun fichier CSV trouvé dans le dossier téléchargé.")