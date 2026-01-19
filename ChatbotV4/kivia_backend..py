import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime

# --- LÓGICA ORIGINAL DE MODELO_V2.PY (Necesaria para cargar el pickle) ---
class ModelConfig:
    def __init__(self):
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        self.xgb_auc_target = 0.80
        self.recommendation_accuracy_target = 0.75
        self.regression_r2_target = 0.70

# --- LÓGICA DE SERVICIO.PY ADAPTADA ---
def cargar_modelo_entrenado():
    """Carga el modelo desde la carpeta models/"""
    ruta_modelo = os.path.join("models", "habit_model.pkl")
    
    if not os.path.exists(ruta_modelo):
        return None, "No se encuentra el archivo models/habit_model.pkl"
    
    try:
        with open(ruta_modelo, "rb") as f:
            modelo = pickle.load(f)
        return modelo, "OK"
    except Exception as e:
        return None, str(e)

def analizar_perfil_usuario(features_list):
    """
    Recibe la lista de 50 números (features) y devuelve el dict con resultados.
    Esta es la misma lógica que tenías en tu endpoint /api/analyze
    """
    modelo, mensaje = cargar_modelo_entrenado()
    
    if modelo is None:
        return {"error": mensaje}

    try:
        # 1. Preprocesamiento (Igual que en servicio.py)
        # Convertir a numpy array y asegurar forma (1, 50)
        features = np.array(features_list).reshape(1, -1)
        
        # 2. Pipeline de transformación
        if hasattr(modelo, 'scaler'):
            features_scaled = modelo.scaler.transform(features)
        else:
            features_scaled = features
            
        if hasattr(modelo, 'pca'):
            features_pca = modelo.pca.transform(features_scaled)
        else:
            features_pca = features_scaled
            
        # 3. Predicciones
        probabilidad = modelo.xgb_model.predict_proba(features_pca)[0, 1]
        score = modelo.regression_model.predict(features_pca)[0]
        
        # Limitar score a 0-100
        score = max(0, min(100, float(score)))
        
        # 4. Generar Interpretación (Tu lógica original)
        recomendacion = "Mantener rutina actual"
        if score < 50: recomendacion = "Necesita cambios urgentes en sueño y actividad."
        elif score < 80: recomendacion = "Vas bien, ajusta pequeños hábitos."
        
        return {
            "kivia_score": round(score, 1),
            "probabilidad_adopcion": round(float(probabilidad), 2),
            "recomendacion": recomendacion,
            "energia_predicha": "Alta" if features_list[0] > 0.6 else "Baja" # Simplificación
        }
            
    except Exception as e:
        return {"error": f"Error en cálculo matemático: {e}"}
