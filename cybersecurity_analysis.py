import kagglehub
import pandas as pd
import os
import glob

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
else:
    print("Aucun fichier CSV trouvé dans le dossier téléchargé.")