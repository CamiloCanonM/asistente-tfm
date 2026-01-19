import streamlit as st
import random
# Importamos la lógica matemática
from kivia_backend import analizar_perfil_usuario

# --- TU CUESTIONARIO ORIGINAL ---
CUESTIONARIO = [
    {
        "id": "energia",
        "pregunta": "¿Cómo calificarías tu nivel de energía?",
        "opciones": ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"]
    },
    {
        "id": "ejercicio",
        "pregunta": "¿Frecuencia de actividad física?",
        "opciones": ["Raramente", "1-2 veces/sem", "3-4 veces/sem", "5+ veces/sem"]
    },
    {
        "id": "sueño",
        "pregunta": "¿Calidad de sueño?",
        "opciones": ["Muy mala", "Mala", "Regular", "Buena", "Excelente"]
    },
    {
        "id": "estres",
        "pregunta": "¿Manejo del estrés?",
        "opciones": ["Muy mal", "Mal", "Regular", "Bien", "Muy bien"]
    },
    {
        "id": "cronotipo",
        "pregunta": "¿Eres matutino o nocturno?",
        "opciones": ["Muy Nocturno", "Nocturno", "Neutral", "Matutino", "Muy Matutino"]
    }
]

def procesar_inputs(respuestas):
    """Convierte texto a vector de 50 números (Lógica Frontend)"""
    vector = [0.5] * 50
    # Mapeo simplificado para el ejemplo (Ajustar según tu lógica exacta)
    indices = {
        "energia": [0,1,2], "ejercicio": [3,4,5], 
        "sueño": [6,7,8], "estres": [9,10,11], "cronotipo": [12,13,14]
    }
    
    for key, val in respuestas.items():
        pregunta = next((p for p in CUESTIONARIO if p["id"] == key), None)
        if pregunta:
            idx = pregunta["opciones"].index(val)
            norm = idx / (len(pregunta["opciones"]) - 1)
            
            if key in indices:
                for i in indices[key]:
                    vector[i] = max(0, min(1, norm + random.uniform(-0.05, 0.05)))
    return vector

def renderizar_planificador():
    """Esta función DIBUJA toda la pantalla del planificador"""
    st.header("📊 Planificador KIVIA")
    st.markdown("Responde para activar la inteligencia del Chatbot.")
    
    with st.container(border=True):
        respuestas = {}
        col1, col2 = st.columns(2)
        
        for i, p in enumerate(CUESTIONARIO):
            with (col1 if i % 2 == 0 else col2):
                respuestas[p["id"]] = st.select_slider(p["pregunta"], options=p["opciones"])
        
        st.write("")
        if st.button("🚀 Analizar Perfil", type="primary", use_container_width=True):
            
            with st.spinner("Procesando con Modelo V2..."):
                # 1. Convertir visual -> vector
                vector = procesar_inputs(respuestas)
                
                # 2. Calcular vector -> predicción (Llamada al Backend)
                resultado = analizar_perfil_usuario(vector)
                
                if "error" in resultado:
                    st.error(resultado["error"])
                else:
                    # 3. GUARDAR EN MEMORIA COMPARTIDA (Para el Chatbot)
                    st.session_state['kivia_data'] = resultado
                    
                    st.success("¡Datos procesados!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Score", resultado['kivia_score'])
                    m2.metric("Probabilidad", f"{resultado['probabilidad_adopcion']*100:.0f}%")
                    m3.metric("Energía", resultado['energia_predicha'])
                    
                    st.info(f"💡 Consejo: {resultado['recomendacion']}")
