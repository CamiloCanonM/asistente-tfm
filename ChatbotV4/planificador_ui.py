import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier

# Configuración de logging para evitar errores en Streamlit
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ==========================================
# 1. DEFINICIÓN DE CLASES (IDENTICAS AL MODELO ORIGINAL)
# ==========================================
# Estas clases son necesarias para que 'pickle' reconozca la estructura del archivo guardado.

class ModelConfig:
    """Configuración del modelo - serializable"""
    def __init__(self):
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        self.xgb_auc_target = 0.80
        self.recommendation_accuracy_target = 0.75
        self.regression_r2_target = 0.70
    
    def to_dict(self):
        return {
            'n_components': self.n_components,
            'pca_variance_threshold': self.pca_variance_threshold,
            'xgb_auc_target': self.xgb_auc_target,
            'recommendation_accuracy_target': self.recommendation_accuracy_target,
            'regression_r2_target': self.regression_r2_target
        }
    
    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

class HabitModel:
    """Modelo completo para el sistema de hábitos"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.pca = None
        self.scaler = None
        self.xgb_model = None
        self.regression_model = None
        self.expert_system_rules = None
        self.metrics = {}
        self.trained = False
        self.last_trained = None
        
    # Incluimos métodos mínimos necesarios para que el objeto no se rompa al cargarse
    def feature_engineering(self, X): pass
    def train_xgboost_classifier(self, X, y): pass
    def train_recommendation_system(self, X, u, c, h): pass
    def train_kivia_regression(self, X, y): pass
    def save_model(self, path): pass
    def load_model(self, path): pass

# ==========================================
# 2. FUNCIÓN DE CARGA ROBUSTA (VERSIÓN DEFINITIVA)
# ==========================================
@st.cache_resource
def cargar_modelo_entrenado():
    # 1. Obtener la ruta exacta donde está este archivo script (planificador_ui.py)
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Construir las rutas basándonos en la ubicación del script
    ruta_models = os.path.join(directorio_script, "models", "habit_model.pkl")
    ruta_raiz = os.path.join(directorio_script, "habit_model.pkl")
    
    # Debug visual: Muestra en la app dónde estamos buscando (puedes borrar esto luego)
    st.info(f"📂 Buscando modelo en:\n1. {ruta_models}\n2. {ruta_raiz}")

    ruta_final = None
    if os.path.exists(ruta_models):
        ruta_final = ruta_models
    elif os.path.exists(ruta_raiz):
        ruta_final = ruta_raiz
            
    if not ruta_final:
        st.error(f"❌ ARCHIVO NO ENCONTRADO. Python está ejecutándose desde: {os.getcwd()}")
        st.error("Por favor, mueve el archivo 'habit_model.pkl' a la misma carpeta donde está 'planificador_ui.py'.")
        return None

    try:
        # Truco para que pickle encuentre las clases
        sys.modules['__main__'] = sys.modules[__name__]
        # Por si el modelo se entrenó en un archivo llamado 'modelo.py'
        sys.modules['modelo'] = sys.modules[__name__] 
        
        with open(ruta_final, 'rb') as f:
            modelo_cargado = pickle.load(f)
            st.success(f"✅ Modelo cargado exitosamente desde: {ruta_final}")
            return modelo_cargado
            
    except Exception as e:
        st.error(f"⚠️ Error leyendo el archivo (puede estar corrupto): {e}")
        return None

# ==========================================
# 3. LÓGICA DE PREDICCIÓN (INPUT -> 50 FEATURES)
# ==========================================
def preparar_vector_entrada(respuestas):
    """
    Convierte las 6 respuestas del usuario en el vector de 50 características
    que espera tu modelo (simulando las otras 44).
    """
    # 1. Crear vector de ceros (1 fila, 50 columnas)
    features = np.zeros((1, 50))
    
    # 2. Mapeos de texto a número (0.0 a 1.0)
    map_energia = {"Muy bajo": 0.1, "Bajo": 0.3, "Moderado": 0.5, "Alto": 0.8, "Muy alto": 1.0}
    map_ejer = {"Sedentario": 0.0, "1-2 días": 0.3, "3-4 días": 0.7, "Atleta": 1.0}
    map_estres = {"Zen": 0.0, "Bajo": 0.2, "Moderado": 0.5, "Alto": 0.8, "Crítico": 1.0}
    map_disci = {"Baja": 0.2, "Variable": 0.5, "Alta": 0.8, "Férrea": 1.0}
    
    val_energia = map_energia.get(respuestas['energia'], 0.5)
    val_ejercicio = map_ejer.get(respuestas['ejercicio'], 0.0)
    val_sueño = respuestas['sueño'] / 100.0
    val_estres = map_estres.get(respuestas['estres'], 0.5)
    val_animo = respuestas['animo'] / 100.0
    val_disciplina = map_disci.get(respuestas['disciplina'], 0.5)
    
    # 3. Llenar las primeras posiciones (Suponiendo que estas son las más importantes)
    features[0, 0] = val_energia
    features[0, 1] = val_ejercicio
    features[0, 2] = val_sueño
    features[0, 3] = val_estres
    features[0, 4] = val_animo
    features[0, 5] = val_disciplina
    
    # 4. Relleno Inteligente (Simulación de correlaciones)
    # Tu modelo fue entrenado con datos aleatorios correlacionados.
    # Debemos imitar ese patrón para que la predicción tenga sentido.
    
    # Las primeras 10 variables en tu entrenamiento definen el Score Kivia
    # Así que llenamos hasta el índice 10 con variaciones de los datos ingresados
    promedio_salud = (val_energia + val_ejercicio + val_sueño + val_animo) / 4
    features[0, 6] = promedio_salud
    features[0, 7] = (1.0 - val_estres) # Inverso del estrés
    features[0, 8] = val_disciplina
    features[0, 9] = promedio_salud * val_disciplina
    
    # Rellenamos el resto (índices 10 a 49) con ruido aleatorio leve basado en el promedio
    # para no afectar drásticamente al PCA
    features[0, 10:] = np.random.normal(promedio_salud, 0.1, 40)
    
    return features

# ==========================================
# 4. UI PRINCIPAL
# ==========================================
def renderizar_planificador():
    st.header("📊 Diagnóstico Predictivo (Modelo Entrenado)")
    
    cerebro = cargar_modelo_entrenado()
    
    if cerebro and cerebro.trained:
        st.success(f"✅ Modelo cargado. Último entrenamiento: {cerebro.last_trained}")
    else:
        st.warning("⚠️ El modelo no está marcado como 'entrenado'. Los resultados pueden ser imprecisos.")

    with st.form("form_prediccion"):
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Fisiología")
            energia = st.select_slider("⚡ Nivel de Energía", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"], value="Moderado")
            sueño = st.slider("💤 Calidad de Sueño (0-100)", 0, 100, 70)
            ejercicio = st.select_slider("🏃 Actividad Física", ["Sedentario", "1-2 días", "3-4 días", "Atleta"], value="1-2 días")

        with c2:
            st.subheader("Psicología")
            estres = st.select_slider("🧠 Nivel de Estrés", ["Zen", "Bajo", "Moderado", "Alto", "Crítico"], value="Moderado")
            animo = st.slider("🎭 Estado de Ánimo (0-100)", 0, 100, 60)
            disciplina = st.select_slider("🛡️ Autodisciplina", ["Baja", "Variable", "Alta", "Férrea"], value="Variable")
            
        submitted = st.form_submit_button("🚀 Analizar Probabilidad", type="primary")
        
    if submitted and cerebro:
        # 1. Preparar datos
        inputs = {
            "energia": energia, "sueño": sueño, "ejercicio": ejercicio,
            "estres": estres, "animo": animo, "disciplina": disciplina
        }
        
        try:
            # 2. Crear vector de 50 características
            vector_raw = preparar_vector_entrada(inputs)
            
            # 3. Pipeline de Predicción (Scaler -> PCA -> Modelos)
            
            # A. Escalar
            if cerebro.scaler:
                vector_scaled = cerebro.scaler.transform(vector_raw)
            else:
                vector_scaled = vector_raw
                
            # B. PCA
            if cerebro.pca:
                vector_pca = cerebro.pca.transform(vector_scaled)
            else:
                vector_pca = vector_scaled
                
            # C. Predicción de Probabilidad (XGBoost)
            prob_exito = 0.5
            if cerebro.xgb_model:
                prob_exito = cerebro.xgb_model.predict_proba(vector_pca)[0, 1]
                
            # D. Predicción de Score (Regresión)
            score_kivia = 50
            if cerebro.regression_model:
                score_raw = cerebro.regression_model.predict(vector_pca)[0]
                score_kivia = int(max(0, min(100, score_raw)))
            
            # 4. Guardar en sesión para el chatbot
            st.session_state['kivia_data'] = {
                "score": score_kivia,
                "prob": prob_exito,
                "perfil": inputs
            }
            
            # 5. Visualización
            st.divider()
            colA, colB = st.columns([1, 2])
            
            with colA:
                st.metric("Score Kivia", f"{score_kivia}/100")
                st.progress(score_kivia/100)
                
            with colB:
                st.subheader(f"Probabilidad de Éxito: {prob_exito:.1%}")
                if prob_exito > 0.75:
                    st.success("🌟 Tu perfil es altamente compatible con la creación de nuevos hábitos.")
                elif prob_exito > 0.45:
                    st.warning("⚖️ Tienes una base sólida, pero el estrés o la falta de sueño podrían frenarte.")
                else:
                    st.error("🛡️ Se detectan barreras importantes. Recomendamos empezar con micro-hábitos muy pequeños.")
                    
        except Exception as e:
            st.error(f"Error durante el análisis: {e}")
            st.write("Detalles para depuración:", e)
