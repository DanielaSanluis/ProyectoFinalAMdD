import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, file_path: str, random_state: int = 42):
        """Clase encargada del flujo completo de Clustering con K-Means."""
        self.file_path = file_path
        self.seed = random_state
        self.df = None
        self.features_scaled = None
        self.scaler = StandardScaler()
        self.features_list = ['EDAD', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']

    def load_and_scale(self):
        """Carga los datos, filtra las variables clínicas y numéricas relevantes y las normaliza."""
        self.df = pd.read_csv(self.file_path, encoding='latin-1')
        
        # Limpieza básica de códigos de nulos ocultos
        df_clean = self.df[self.features_list].copy()
        for col in ['DIABETES', 'HIPERTENSION', 'OBESIDAD']:
            df_clean[col] = df_clean[col].replace([97, 98, 99], np.nan)
        df_clean['EDAD'] = df_clean['EDAD'].replace(999, np.nan)
        
        # Imputación simple por consistencia
        df_clean['EDAD'] = df_clean['EDAD'].fillna(df_clean['EDAD'].median())
        for col in ['DIABETES', 'HIPERTENSION', 'OBESIDAD']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            # Recodificación binaria idéntica (SÍ = 1, NO = 0) para que la distancia tenga sentido
            df_clean[col] = df_clean[col].map({1: 1, 2: 0})
            
        self.data_clean = df_clean
        # Es obligatorio escalar los datos antes de usar K-Means
        self.features_scaled = self.scaler.fit_transform(df_clean)

    def calculate_elbow(self, max_k: int = 8) -> list:
        """Calcula la inercia (Suma de cuadrados internos) para valores de K."""
        inercias = []
        ks = range(1, max_k + 1)
        for k in ks:
            kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
            kmeans.fit(self.features_scaled)
            inercias.append(kmeans.inertia_)
        return inercias

    def train_kmeans(self, n_clusters: int):
        """Entrena el modelo definitivo con el número de clústeres óptimo."""
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=self.seed, n_init=10)
        self.data_clean['CLUSTER'] = self.kmeans_model.fit_predict(self.features_scaled)
        return self.data_clean

    def get_profiles(self) -> pd.DataFrame:
        """Calcula las estadísticas promedio de las variables originales por clúster."""
        # Agrupamos por clúster y promediamos para ver el porcentaje de presencia de la enfermedad
        perfiles = self.data_clean.groupby('CLUSTER').mean()
        perfiles['Tamaño_Grupo'] = self.data_clean.groupby('CLUSTER').size()
        return perfiles