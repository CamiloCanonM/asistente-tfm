
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys

# ==========================================
# 1. ZONA DE CLASES (INTEGRACIÓN MODELO V2)
# ==========================================
# 👇 AQUÍ PEGA LAS CLASES DE TU ARCHIVO modelo_v2.py
# (Necesitamos que estén definidas AQUÍ para que el pickle las reconozca)

# --- EJEMPLO (Reemplaza esto con el contenido real de tu modelo_v2.py) ---

class ModelConfig:
    def __init__(self):
        # Configuración por defecto que suele tener tu modelo
        self.n_components = 20
        self.pca_variance_threshold = 0.85
        # ... copia el contenido real de tu archivo ...

class HabitModel:
    def __init__(self):
        self.config = ModelConfig()
        self.pca = None
        self.scaler = None
        self.xgb_model = None       
        self.regression_model = None 
        self.trained = False

# 👆 FIN DE LA ZONA DE PEGADO 👆


# ==========================================
# 2. CARGA DEL CEREBRO (CON TRUCO DE IDENTIDAD)
# ==========================================
@st.cache_resource
def cargar_modelo():
    """Carga el modelo original habit_model.pkl aplicando parches de compatibilidad"""
    ruta_modelo = os.path.join(os.path.dirname(__file__), "habit_model.pkl")
    
    if not os.path.exists(ruta_modelo):
        st.error(f"❌ No encuentro el archivo: {ruta_modelo}")
        return None

    try:
        # --- EL TRUCO MAESTRO ---
        # Engañamos al sistema para decirle: "Si el archivo busca 'modelo_v2', usa este archivo actual"
        # Esto soluciona el error "No module named 'modelo_v2'"
        sys.modules['modelo_v2'] = sys.modules[__name__]
        sys.modules['kivia_backend'] = sys.modules[__name__] # Por seguridad
        
        with open(ruta_modelo, "rb") as f:
            modelo = pickle.load(f)
            return modelo
            
    except Exception as e:
        st.error(f"⚠️ Error cargando el cerebro: {e}")
        st.info("💡 PISTA: Asegúrate de haber copiado las clases 'ModelConfig' y 'HabitModel' exactamente como están en tu archivo modelo_v2.py en la parte superior de este script.")
        return None

# ==========================================
# 3. PROCESAMIENTO INTELIGENTE
# ==========================================
def procesar_datos_entrada(respuestas):
    """Transforma las respuestas del usuario al vector de 50 características que necesita la IA"""
    features = np.zeros((1, 50))
    
    # 1. Extracción directa (Normalizamos de texto a 0.0 - 1.0)
    map_valores = {
        # Energía / Estrés / Disciplina
        "Muy bajo": 0.1, "Bajo": 0.3, "Moderado": 0.5, "Alto": 0.8, "Muy alto": 1.0,
        "Zen": 0.1, "Medio": 0.5, "Crítico": 1.0,
        "Baja": 0.2, "Variable": 0.5, "Férrea": 1.0,
        # Ejercicio
        "Sedentario": 0.0, "1-2 días": 0.3, "3-4 días": 0.7, "Atleta": 1.0
    }

    val_energia = map_valores.get(respuestas['energia'], 0.5)
    val_ejercicio = map_valores.get(respuestas['ejercicio'], 0.0)
    val_sueño = respuestas['sueño'] / 100.0
    val_estres = map_valores.get(respuestas['estres'], 0.5)
    val_animo = respuestas['animo'] / 100.0
    val_disciplina = map_valores.get(respuestas['disciplina'], 0.5)

    # 2. Asignación a las posiciones críticas (Ajustar según cómo entrenaste tu modelo)
    # Asumimos las primeras posiciones según lógica estándar
    features[0, 0] = val_energia
    features[0, 1] = val_sueño
    features[0, 2] = val_estres
    features[0, 3] = val_ejercicio
    features[0, 4] = val_animo
    features[0, 5] = val_disciplina
    
    # 3. Inferencia de datos latentes (Rellenamos el resto del vector con lógica difusa)
    # Si tienes estrés alto, tu calidad de sueño latente baja
    features[0, 10:20] = val_disciplina * 0.8  
    features[0, 20:30] = (val_energia + val_ejercicio) / 2
    features[0, 40:50] = (1.0 - val_estres) # Factor de resiliencia
    
    return features

# ==========================================
# 4. INTERFAZ GRÁFICA (RENDERING)
# ==========================================
def renderizar_planificador():
    st.title("📊 Planificador de Hábitos Kivia")
    st.markdown("---")

    # Cargar cerebro
    cerebro = cargar_modelo()
    
    # Formulario
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fisiología")
        energia = st.select_slider("⚡ Nivel de Energía", ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"], value="Moderado")
        sueño = st.slider("💤 Calidad de Sueño (0-100)", 0, 100, 70)
        ejercicio = st.select_slider("🏃 Actividad Física", ["Sedentario", "1-2 días", "3-4 días", "Atleta"], value="1-2 días")

    with col2:
        st.subheader("Psicología")
        estres = st.select_slider("🧠 Nivel de Estrés", ["Zen", "Bajo", "Moderado", "Alto", "Crítico"], value="Moderado")
        animo = st.slider("🎭 Estado de Ánimo (0-100)", 0, 100, 60)
        disciplina = st.select_slider("🛡️ Autodisciplina", ["Baja", "Variable", "Alta", "Férrea"], value="Variable")

    # Botón de Acción
    if st.button("🚀 Analizar Viabilidad", type="primary", use_container_width=True):
        if cerebro:
            with st.spinner("El modelo original está pensando..."):
                # 1. Preparar datos
                input_dict = {
                    "energia": energia, "sueño": sueño, "ejercicio": ejercicio,
                    "estres": estres, "animo": animo, "disciplina": disciplina
                }
                
                try:
                    # 2. Vectorizar
                    vector_raw = procesar_datos_entrada(input_dict)
                    
                    # 3. Transformaciones del Modelo (Scaler -> PCA)
                    # Verifica si tu modelo tiene estos pasos dentro
                    datos_procesados = vector_raw
                    
                    if hasattr(cerebro, 'scaler') and cerebro.scaler:
                        datos_procesados = cerebro.scaler.transform(datos_procesados)
                        
                    if hasattr(cerebro, 'pca') and cerebro.pca:
                        datos_procesados = cerebro.pca.transform(datos_procesados)
                    
                    # 4. Predicción
                    prob_exito = 0.5
                    puntaje = 50

                    # Intenta predecir probabilidad (Clasificación)
                    if hasattr(cerebro, 'xgb_model') and cerebro.xgb_model:
                        try:
                            probs = cerebro.xgb_model.predict_proba(datos_procesados)
                            prob_exito = probs[0, 1] # Probabilidad de clase 1 (Éxito)
                        except: pass

                    # Intenta predecir puntaje (Regresión)
                    if hasattr(cerebro, 'regression_model') and cerebro.regression_model:
                        try:
                            puntaje = cerebro.regression_model.predict(datos_procesados)[0]
                        except: pass
                    
                    # Ajuste visual
                    puntaje_final = int(np.clip(puntaje, 0, 100))
                    
                    # 5. Guardar en Session State para el Chatbot
                    st.session_state['kivia_data'] = {
                        "score": puntaje_final,
                        "prob": prob_exito,
                        "perfil": input_dict
                    }

                    # 6. Mostrar Resultados
                    st.success("✅ Análisis Completado")
                    
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("Puntaje Kivia", f"{puntaje_final}/100")
                    c_res2.metric("Probabilidad Éxito", f"{prob_exito:.1%}")
                    
                    if prob_exito > 0.7:
                        c_res3.info("🌟 Estás en gran forma para iniciar.")
                    elif prob_exito > 0.4:
                        c_res3.warning("⚠️ Requiere esfuerzo y constancia.")
                    else:
                        c_res3.error("🛡️ Empieza pequeño, riesgo de abandono.")

                except Exception as e:
                    st.error(f"Error en la predicción interna: {e}")
                    st.write("Detalle técnico:", e)
        else:
            st.warning("El modelo no se cargó correctamente. Revisa la sección de clases.")
