import numpy as np
import pickle
import os
import streamlit as st

# --- 1. CLASES DEL MODELO ORIGINAL ---
class ModelConfig:
    def __init__(self):
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        self.xgb_auc_target = 0.80
        self.recommendation_accuracy_target = 0.75
        self.regression_r2_target = 0.70

class HabitModel:
    def __init__(self):
        self.config = ModelConfig()
        self.pca = None
        self.scaler = None
        self.xgb_model = None       
        self.regression_model = None 
        self.trained = False

# --- 2. CARGA DEL CEREBRO ---
@st.cache_resource
def cargar_cerebro_completo():
    # Busca en la carpeta raíz y en 'models'
    ruta_app = os.path.dirname(os.path.abspath(__file__))
    rutas = [
        os.path.join(ruta_app, "habit_model.pkl"), 
        os.path.join(ruta_app, "models", "habit_model.pkl")
    ]
    
    for ruta in rutas:
        if os.path.exists(ruta):
            try:
                with open(ruta, "rb") as f:
                    return pickle.load(f)
            except: continue
    return None

# --- 3. LOGICA DE TRADUCCION (La magia de Flask) ---
def procesar_cuestionario_inteligente(respuestas):
    """
    Convierte el diccionario de respuestas humanas en el vector de 50 numeros.
    """
    # 1. Vector base
    features = np.zeros((1, 50))
    
    # 2. Extraer valores (0.0 a 1.0)
    energia = respuestas.get("energia", 0.5)
    sueño = respuestas.get("sueño", 0.5)
    estres = respuestas.get("estres", 0.5)
    ejercicio = respuestas.get("ejercicio", 0.0)
    animo = respuestas.get("animo", 0.5)
    disciplina = respuestas.get("disciplina", 0.5)
    
    # 3. Asignar a las posiciones principales (Feature Engineering)
    features[0, 0] = energia
    features[0, 1] = sueño
    features[0, 2] = estres
    features[0, 3] = ejercicio
    features[0, 4] = animo
    features[0, 5] = disciplina
    
    # 4. Relleno Inteligente (Simulación de comportamiento)
    promedio_general = (energia + sueño + (1-estres) + ejercicio + animo) / 5
    
    # Rellenamos bloques latentes con lógica difusa
    features[0, 10:20] = disciplina * 0.8 + np.random.normal(0, 0.05, 10) # Consistencia
    features[0, 20:30] = (sueño + (1-estres))/2 + np.random.normal(0, 0.05, 10) # Bienestar
    features[0, 40:50] = promedio_general # Tendencia general
    
    return features
