import numpy as np
import pandas as pd

class EdaAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Carga el dataset """
        try:
            # Usamos latin-1 por las eñes y acentos
            self.df = pd.read_csv(self.file_path, encoding='latin-1')
            print("¡Dataset cargado exitosamente!")
            return self.df
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return None

    def get_general_info(self):
        """Regresa las dimensiones y los tipos de datos por variable."""
        if self.df is None:
            return "El dataset no ha sido cargado."
        
        num_filas, num_columnas = self.df.shape
        tipos_datos = self.df.dtypes
        
        return {
            "filas": num_filas,
            "columnas": num_columnas,
            "tipos_datos": tipos_datos
        }

    def analyze_data_quality(self) -> pd.DataFrame:
        """2. Detecta y calcula el porcentaje de valores faltantes reales y duplicados."""
        if self.df is None:
            raise ValueError("El dataset no ha sido cargado.")

        # Columnas categóricas donde la SSa oculta nulos en los códigos 97, 98 y 99
        columnas_con_codigos = [
            'SEXO', 'TIPO_PACIENTE', 'PNEUMONIA', 'DIABETES', 'EPOC', 
            'ASMA', 'INMUNOSUPRESION', 'HIPERTENSION', 'OTRA_COM', 
            'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 
            'OTRO_CASO', 'RESULTADO_LAB', 'CLASIFICACION_FINAL', 'INTUBADO', 'UCI'
        ]
        
        df_mapeado = self.df.copy()
        for col in columnas_con_codigos:
            if col in df_mapeado.columns:
                df_mapeado[col] = df_mapeado[col].replace([97, 98, 99], np.nan)
        
        if 'EDAD' in df_mapeado.columns:
            df_mapeado['EDAD'] = df_mapeado['EDAD'].replace(999, np.nan)

        # Contar nulos transformados
        valores_faltantes = df_mapeado.isnull().sum()
        porcentajes = (valores_faltantes / len(df_mapeado)) * 100
        
        # Conteo de duplicados
        num_duplicados = self.df.duplicated().sum()
        print(f"\n[Calidad de Datos] Registros completamente idénticos detectados: {num_duplicados}")

        reporte_calidad = pd.DataFrame({
            'Valores Faltantes (NaN)': valores_faltantes,
            'Porcentaje (%)': porcentajes
        })
        
        return reporte_calidad.sort_values(by='Porcentaje (%)', ascending=False)
    
    def analyze_numeric_stats(self, column_name: str) -> pd.DataFrame:
        """Calcula media, mediana, desviación estándar, cuartiles y rango para una variable numérica."""
        if self.df is None or column_name not in self.df.columns:
            return None
        
        # Reemplazamos nulo de edad (999) por si acaso para no sesgar
        serie_num = self.df[column_name].replace(999, np.nan).dropna()
        
        desc = serie_num.describe()
        rango = desc['max'] - desc['min']
        
        reporte_num = pd.DataFrame({
            'Métrica': ['Media', 'Mediana', 'Desviación Estándar', 'Mínimo', 'Cuartil 1 (25%)', 'Cuartil 3 (75%)', 'Máximo', 'Rango'],
            'Valor': [desc['mean'], serie_num.median(), desc['std'], desc['min'], desc['25%'], desc['75%'], desc['max'], rango]
        })
        return reporte_num

    def analyze_categorical_stats(self, columns_list: list) -> pd.DataFrame:
        """Calcula las frecuencias absolutas, porcentajes y la moda para variables categóricas limpias."""
        if self.df is None:
            return None
            
        resultados = []
        for col in columns_list:
            if col in self.df.columns:
                # Quitamos nulos ocultos para calcular modas y frecuencias reales
                serie_cat = self.df[col].replace([97, 98, 99], np.nan).dropna()
                
                if not serie_cat.empty:
                    moda = serie_cat.mode()[0]
                    total_validos = len(serie_cat)
                    
                    # Obtenemos la frecuencia del valor más común
                    frecuencia_moda = serie_cat.value_counts().iloc[0]
                    pct_moda = (frecuencia_moda / total_validos) * 100
                    
                    resultados.append({
                        'Variable': col,
                        'Moda (Código)': moda,
                        'Frecuencia Absoluta Moda': frecuencia_moda,
                        'Porcentaje de la Moda (%)': round(pct_moda, 2)
                    })
                    
        return pd.DataFrame(resultados)
    

    #Graficas
    def plot_age_distribution(self):
        """Genera el histograma y boxplot de la columna EDAD."""
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        if self.df is None or 'EDAD' not in self.df.columns:
            print("Error: Dataset no cargado o columna EDAD ausente.")
            return

        # Configuración estética
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Limpieza interna
        edad_limpia = self.df['EDAD'].replace(999, np.nan).dropna()

        # Gráfica 1: Histograma
        sns.histplot(edad_limpia, bins=30, kde=True, color="skyblue", ax=axes[0])
        axes[0].set_title("Distribución General de la Edad", fontsize=14, pad=10)
        axes[0].set_xlabel("Edad (Años)", fontsize=12)
        axes[0].set_ylabel("Cantidad de Registros", fontsize=12)

        # Gráfica 2: Boxplot
        sns.boxplot(x=edad_limpia, color="lightcoral", ax=axes[1])
        axes[1].set_title("Detección de Valores Atípicos en la Edad", fontsize=14, pad=10)
        axes[1].set_xlabel("Edad (Años)", fontsize=12)

        plt.tight_layout()
        return fig # Retornamos la figura para que el notebook la renderice

    def calculate_age_outliers_iqr(self) -> dict:
        """Calcula matemáticamente los valores atípicos de la EDAD usando la regla IQR."""
        if self.df is None or 'EDAD' not in self.df.columns:
            return None
            
        # Limpieza básica inicial
        edad_serie = self.df['EDAD'].replace(999, np.nan).dropna()
        
        # Calculamos los cuartiles 1 y 3
        q1 = edad_serie.quantile(0.25)
        q3 = edad_serie.quantile(0.75)
        iqr = q3 - q1
        
        # Regla IQR: Límite superior es Q3 + 1.5 * IQR
        limite_superior = q3 + (1.5 * iqr)
        limite_inferior = q1 - (1.5 * iqr)
        
        # Pacientes que superan el límite superior
        atipicos = edad_serie[edad_serie > limite_superior]
        
        return {
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Limite_Superior_IQR': limite_superior,
            'Cantidad_Atipicos': len(atipicos),
            'Porcentaje_Atipicos': (len(atipicos) / len(edad_serie)) * 100
        }
    
    def plot_correlation_matrix(self):
        """Genera una matriz de correlación de Spearman y la muestra mediante un Heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if self.df is None:
            print("Error: Dataset no cargado.")
            return

        # Seleccionamos las columnas más importantes para analizar sus relaciones
        columnas_interes = [
            'EDAD', 'SEXO', 'TIPO_PACIENTE', 'DIABETES', 
            'HIPERTENSION', 'OBESIDAD', 'INTUBADO', 'UCI'
        ]
        
        # Filtramos para quedarnos solo con las columnas que existan en tu archivo
        cols_validas = [c for c in columnas_interes if c in self.df.columns]
        
        # Creamos una copia y limpiamos los nulos ocultos (97, 98, 99) para que no alteren la matemática
        df_corr = self.df[cols_validas].copy()
        for col in cols_validas:
            if col != 'EDAD':
                df_corr[col] = df_corr[col].replace([97, 98, 99], np.nan)
        
        # Calculamos la matriz usando Spearman (ideal para variables mixtas y categóricas ordenadas)
        matriz_corr = df_corr.corr(method='spearman')

        # Configuración de la gráfica
        sns.set_theme(style="white")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Dibujamos el Heatmap
        sns.heatmap(
            matriz_corr, 
            annot=True,          # Pone los números dentro de cada cuadro
            fmt=".2f",           # Limita a 2 decimales
            cmap="coolwarm",     # Color azul para negativo, rojo para positivo
            vmin=-1, vmax=1,     # Límites de la correlación
            square=True,         # Hace los cuadros perfectamente cuadrados
            linewidths=0.5,      # Línea delgada de separación
            cbar_kws={"shrink": .8},
            ax=ax
        )
        
        ax.set_title("Matriz de Correlación entre Factores Clínicos y Demográficos", fontsize=14, pad=15)
        plt.tight_layout()
        
        return fig
    
    def plot_categorical_distributions(self):
        """Genera gráficas de barras para evaluar las frecuencias y el balance de clases."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np

        if self.df is None:
            print("Error: Dataset no cargado.")
            return

        # Variables categóricas clave a graficar
        variables = ['SEXO', 'DIABETES', 'HIPERTENSION', 'TIPO_PACIENTE']
        
        # Configuramos una cuadrícula de 2 filas y 2 columnas para las gráficas
        sns.set_theme(style="whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten() # Aplana la matriz de ejes para iterar fácilmente

        # Diccionarios para cambiar los códigos numéricos por etiquetas claras en las gráficas
        labels_map = {
            'SEXO': {1: 'Mujer', 2: 'Hombre'},
            'DIABETES': {1: 'SÍ', 2: 'NO'},
            'HIPERTENSION': {1: 'SÍ', 2: 'NO'},
            'TIPO_PACIENTE': {1: 'Ambulatorio', 2: 'Hospitalizado'}
        }

        for i, col in enumerate(variables):
            if col in self.df.columns:
                # Limpiamos nulos ocultos temporalmente para la gráfica
                serie_clean = self.df[col].replace([97, 98, 99], np.nan).dropna()
                
                # Mapeamos los números a texto (ej. 1 -> Mujer)
                if col in labels_map:
                    serie_clean = serie_clean.map(labels_map[col])

                # Dibujamos la gráfica de barras de conteo (frecuencias)
                sns.countplot(
                    x=serie_clean, 
                    ax=axes[i], 
                    palette="Blues_r", 
                    hue=serie_clean, 
                    legend=False
                )
                
                # Añadimos títulos y etiquetas limpias
                axes[i].set_title(f"Distribución de la Variable: {col}", fontsize=12, pad=10)
                axes[i].set_xlabel("")
                axes[i].set_ylabel("Número de Casos")
                
                # Truco para poner el número de conteo encima de cada barra
                for container in axes[i].containers:
                    axes[i].bar_label(container, fmt='%d', label_type='edge', padding=3)

        plt.tight_layout()
        return fig
    
if __name__ == "__main__":
    ruta_dataset = "../data/COVID19MEXICO.csv" 
    
    analyzer = EdaAnalyzer(ruta_dataset)


    if analyzer.load_data() is not None:
        # # 1.Informacion general
        # info = analyzer.get_general_info()
        # print(f"\nDimensiones: {info['filas']} filas x {info['columnas']} columnas")
        # print("\nTipos de datos por columna:")
        # print(info['tipos_datos'])

        # # 2. Calidad de Datos
        # print(f"\n--- Top 10 columnas con más nulos reales ---")
        # quality_report = analyzer.analyze_data_quality()
        # print(quality_report.head(10))
        
        # 3. Estadisticas descriptivas
        print(f"\n--- Variable Numérica: EDAD ---")
        edad_stats = analyzer.analyze_numeric_stats('EDAD')
        print(edad_stats.to_string(index=False))

        print(f"\n--- Variables Categóricas Clave ---")
        # Analizamos una muestra de variables para comprobar las modas
        cat_cols = ['SEXO', 'TIPO_PACIENTE', 'DIABETES', 'HIPERTENSION', 'OBESIDAD']
        categorical_stats = analyzer.analyze_categorical_stats(cat_cols)
        print(categorical_stats.to_string(index=False))