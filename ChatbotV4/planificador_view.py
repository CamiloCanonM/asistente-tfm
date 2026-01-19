import streamlit as st
import requests
import random
import time

# Configuración de la API (Debe coincidir con donde corre servicio.py)
API_URL = "http://localhost:8001"

# ==========================================
# 1. DEFINICIÓN DEL CUESTIONARIO
# ==========================================
# Replicamos la estructura de app_flask.py adaptada a Streamlit
CUESTIONARIO = [
    {
        "id": "energia",
        "pregunta": "¿Cómo calificarías tu nivel de energía general?",
        "opciones": ["Muy bajo", "Bajo", "Moderado", "Alto", "Muy alto"]
    },
    {
        "id": "ejercicio",
        "pregunta": "¿Con qué frecuencia realizas actividad física?",
        "opciones": ["Raramente", "1-2 veces/sem", "3-4 veces/sem", "5+ veces/sem"]
    },
    {
        "id": "sueño",
        "pregunta": "¿Cómo es tu calidad de sueño?",
        "opciones": ["Muy mala", "Mala", "Regular", "Buena", "Excelente"]
    },
    {
        "id": "estres",
        "pregunta": "¿Cómo manejas el estrés actualmente?",
        "opciones": ["Muy mal", "Mal", "Regular", "Bien", "Muy bien"]
    },
    {
        "id": "cronotipo",
        "pregunta": "¿Eres una persona matutina o nocturna?",
        "opciones": ["Muy Nocturno", "Nocturno", "Neutral", "Matutino", "Muy Matutino"]
    }
]

# ==========================================
# 2. LÓGICA DE CONVERSIÓN (FRONTEND)
# ==========================================
def convertir_respuestas_a_vector(respuestas):
    """
    Transforma las respuestas de texto en el vector numérico de 50 características
    que espera modelo_v2.py.
    """
    # Inicializamos vector base de 50 características con valor neutro (0.5)
    features = [0.5] * 50
    
    # Mapeo simple: Qué índices del vector de 50 afecta cada pregunta
    # (Simulación de la lógica de app_flask.py)
    mapa_indices = {
        "energia": [0, 1, 2, 3, 4],
        "ejercicio": [5, 6, 7, 8, 9],
        "sueño": [10, 11, 12, 13, 14],
        "estres": [15, 16, 17, 18, 19],
        "cronotipo": [20, 21, 22, 23, 24]
    }
    
    for key, valor_texto in respuestas.items():
        # Encontrar la pregunta para saber sus opciones
        pregunta_obj = next((p for p in CUESTIONARIO if p["id"] == key), None)
        
        if pregunta_obj:
            opciones = pregunta_obj["opciones"]
            # Convertir texto a número (0.0 a 1.0)
            indice = opciones.index(valor_texto)
            valor_normalizado = indice / (len(opciones) - 1)
            
            # Asignar este valor a los índices correspondientes del vector
            indices_afectados = mapa_indices.get(key, [])
            for i in indices_afectados:
                # Añadimos ligera variación para que no sea plano (como en el modelo original)
                variacion = random.uniform(-0.05, 0.05)
                features[i] = max(0.0, min(1.0, valor_normalizado + variacion))
                
    return features

# ==========================================
# 3. INTERFAZ GRÁFICA (COMPONENT)
# ==========================================
def renderizar_planificador():
    st.header("🧠 Planificador de Hábitos IA")
    st.markdown("Este módulo utiliza el **Modelo V2** para analizar tus patrones y recomendar mejoras.")
    
    # VERIFICACIÓN DE SALUD DE LA API
    try:
        check = requests.get(f"{API_URL}/api/health", timeout=2)
        if check.status_code == 200:
            st.success("🟢 Cerebro IA Conectado")
        else:
            st.warning("🟠 La API responde pero con errores.")
    except:
        st.error("🔴 No se detecta el servicio IA.")
        st.info("⚠️ Asegúrate de ejecutar `python servicio.py` en tu terminal.")
        return # Salimos si no hay API

    st.divider()

    # FORMULARIO
    with st.form("form_habitos"):
        col1, col2 = st.columns(2)
        respuestas_usuario = {}
        
        for i, item in enumerate(CUESTIONARIO):
            # Alternar columnas para diseño bonito
            donde = col1 if i % 2 == 0 else col2
            with donde:
                val = st.select_slider(
                    label=item["pregunta"],
                    options=item["opciones"],
                    key=f"slider_{item['id']}"
                )
                respuestas_usuario[item["id"]] = val
        
        st.write("")
        enviar = st.form_submit_button("🚀 Analizar mis Hábitos")

    # PROCESAMIENTO
    if enviar:
        with st.spinner("Conectando con modelo_v2..."):
            # 1. Convertir a números
            vector_features = convertir_respuestas_a_vector(respuestas_usuario)
            
            # 2. Preparar payload JSON
            payload = {
                "features": vector_features,
                "user_id": st.session_state.get("usuario_nombre", "invitado")
            }
            
            # 3. Enviar a servicio.py
            try:
                response = requests.post(f"{API_URL}/api/analizar_rapido", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 4. MOSTRAR RESULTADOS
                    st.toast("Análisis completado", icon="✅")
                    
                    # Score Kivia
                    score = data.get("kivia_score", 0)
                    st.metric(label="🏆 KIVIA Score", value=f"{score:.1f}/100")
                    
                    # Probabilidad de Adopción
                    prob = data.get("probabilidad_adopcion", 0)
                    st.progress(prob, text=f"Probabilidad de éxito del hábito: {prob*100:.1f}%")
                    
                    # Recomendación del Modelo
                    if "recomendacion" in data:
                        st.info(f"💡 Consejo de la IA: {data['recomendacion']}")
                        
                    # Detalles técnicos (Opcional, debugging)
                    with st.expander("Ver detalles del modelo JSON"):
                        st.json(data)
                        
                else:
                    st.error(f"Error del servidor: {response.text}")
                    
            except Exception as e:
                st.error(f"Error de conexión: {e}")
