import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

class ModelTrainer:
    def __init__(self, file_path: str, random_state: int = 42):
        """Clase encargada del flujo de preprocesamiento, entrenamiento y persistencia."""
        self.file_path = file_path
        self.seed = random_state
        self.df = None
        self.X_train, self.X_test = None, None
        self.y_train, self.y_test = None, None
        self.model = None

    def load_and_preprocess(self):
        """Carga los datos y realiza el preprocesamiento específico."""
        # Leer el dataset de forma segura
        self.df = pd.read_csv(self.file_path, encoding='latin-1')
        
        # 1. Definición de variables de entrada (Features) y Objetivo (Target)
        # Filtramos para usar las comorbilidades y datos demográficos clave
        features = ['EDAD', 'SEXO', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']
        target = 'TIPO_PACIENTE'
        
        # 2. Limpieza de nulos ocultos (códigos 97, 98, 99, 999) en este bloque
        df_clean = self.df[features + [target]].copy()
        for col in features:
            if col != 'EDAD':
                df_clean[col] = df_clean[col].replace([97, 98, 99], np.nan)
        df_clean['EDAD'] = df_clean['EDAD'].replace(999, np.nan)
        df_clean[target] = df_clean[target].replace([97, 98, 99], np.nan)
        
        # Imputación simple por la moda/mediana y eliminación de nulos en el target
        df_clean = df_clean.dropna(subset=[target])
        df_clean['EDAD'] = df_clean['EDAD'].fillna(df_clean['EDAD'].median())
        for col in ['SEXO', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
        # 3. Transformación: Mapear target a binario estandarizado (0: Ambulatorio, 1: Hospitalizado)
        # Original: 1 es Ambulatorio, 2 es Hospitalizado
        df_clean[target] = df_clean[target].map({1: 0, 2: 1})
        
        self.X = df_clean[features]
        self.y = df_clean[target]

    def split_data(self, test_size: float = 0.3):
        """Divide los datos en conjuntos de entrenamiento y prueba."""
        # División obligatoria con semilla fija para reproducibilidad
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.seed, stratify=self.y
        )
        print(f"Datos divididos: {self.X_train.shape[0]} entrenamiento, {self.X_test.shape[0]} prueba.")

    def train_random_forest(self, n_estimators: int = 100, max_depth: int = 10):
        """Entrena el algoritmo Random Forest ajustando hiperparámetros básicos."""
        print("Entrenando el modelo Random Forest Classifier...")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=self.seed,
            n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        print("¡Entrenamiento completado exitosamente!")

    def evaluate_model(self) -> dict:
        """Calcula las métricas obligatorias requeridas en la rúbrica."""
        if self.model is None:
            raise ValueError("El modelo no ha sido entrenado.")
            
        preds = self.model.predict(self.X_test)
        
        # Cálculo de la línea base trivial (Clasificador mayoritario)
        clase_mayoritaria = self.y_train.mode()[0]
        preds_linea_base = np.full_like(self.y_test, fill_value=clase_mayoritaria)
        accuracy_base = accuracy_score(self.y_test, preds_linea_base)
        
        # Métricas del modelo
        acc = accuracy_score(self.y_test, preds)
        cm = confusion_matrix(self.y_test, preds)
        report = classification_report(self.y_test, preds, output_dict=True)
        
        return {
            'Accuracy_Modelo': acc,
            'Accuracy_Linea_Base': accuracy_base,
            'Matriz_Confusion': cm,
            'F1_Macro': report['macro avg']['f1-score'],
            'Precision_Macro': report['macro avg']['precision'],
            'Recall_Macro': report['macro avg']['recall']
        }

    def save_model(self, output_dir: str = "../models"):
        """Persiste el modelo entrenado en el disco usando joblib."""
        if self.model is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        ruta_guardado = os.path.join(output_dir, "random_forest_model.joblib")
        joblib.dump(self.model, ruta_guardado)
        print(f"Modelo serializado y guardado exitosamente en: {ruta_guardado}")