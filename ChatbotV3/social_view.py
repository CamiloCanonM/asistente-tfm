import streamlit as st
import pandas as pd
import googlemaps
import time
from langchain_core.messages import AIMessage

# --- 1. CONEXIÓN ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. BÚSQUEDA ---
def buscar_lugares_google(ciudad, palabra_clave):
    gmaps = obtener_cliente_google()
    
    # Si no hay API Key, error silencioso (o modo demo)
    if not gmaps:
        return None, "Error: No detecto la API Key de Google."

    try:
        # A. Geocodificar
        geocode_result = gmaps.geocode(ciudad)
        if not geocode_result: return None, f"No encontré la ciudad: {ciudad}"
        
        loc = geocode_result[0]['geometry']['location']
        lat_centro, lon_centro = loc['lat'], loc['lng']

        # B. Buscar Lugares
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000
        )
        
        lugares = places_result.get('results', [])
        if not lugares: return None, f"No vi {palabra_clave} en {ciudad}."

        # C. Crear DataFrame (Latitude/Longitude es OBLIGATORIO para st.map)
        data = []
        for l in lugares:
            data.append({
                'latitude': float(l['geometry']['location']['lat']),
                'longitude': float(l['geometry']['location']['lng'])
            })
            
        return pd.DataFrame(data), f"✅ Encontré {len(data)} {palabra_clave} en {ciudad}."

    except Exception as e:
        return None, f"Error Google: {str(e)}"

# --- 3. RENDERIZADO (LA CLAVE ESTÁ AQUÍ) ---
def renderizar_sidebar():
    st.subheader("📍 Mapa en Vivo")
    
    # Memoria del mapa
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    ciudad = st.text_input("Ciudad:", "Bogota")
    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios"])
    
    # --- LÓGICA DEL BOTÓN ---
    if st.button("🔍 VER EN MAPA", use_container_width=True):
        with st.spinner("Conectando..."):
            df, msg = buscar_lugares_google(ciudad, tipo)
            
            if df is not None:
                # 1. Guardamos el mapa
                st.session_state.mapa_data = df
                # 2. Guardamos el mensaje en el historial del chat DIRECTAMENTE AQUÍ
                st.session_state.chat_history.append(AIMessage(content=f"[MAPA]: {msg}"))
                st.success(msg)
            else:
                st.error(msg)
    
    # --- PINTAR EL MAPA (FUERA DEL BOTÓN) ---
    # Si hay datos en memoria, SE PINTA. No importa si la app se recarga.
    if st.session_state.mapa_data is not None:
        st.write("---")
        st.map(st.session_state.mapa_data, size=20, color='#FF0000') # Puntos rojos grandes
        
        if st.button("Limpiar"):
            st.session_state.mapa_data = None
            st.rerun()
