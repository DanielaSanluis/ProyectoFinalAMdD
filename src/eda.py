import pandas as pd

class DatasetDescriptor:
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

if __name__ == "__main__":
    ruta_datos = "../data/COVID19MEXICO.csv" 
    
    descriptor = DatasetDescriptor(ruta_datos)
    if descriptor.load_data() is not None:
        info = descriptor.get_general_info()
        print(f"\nDimensiones: {info['filas']} filas x {info['columnas']} columnas")
        print("\nTipos de datos por columna:")
        print(info['tipos_datos'])