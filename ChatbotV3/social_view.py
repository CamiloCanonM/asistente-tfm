import streamlit as st
import pandas as pd
import googlemaps
import time
import numpy as np

# --- 1. CONFIGURACIÓN DEL CLIENTE GOOGLE ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. FUNCIÓN DE BÚSQUEDA ---
def buscar_lugares_google(ciudad, palabra_clave):
    gmaps = obtener_cliente_google()
    
    # MODO SIMULACIÓN (Si falla la API)
    if not gmaps:
        time.sleep(1)
        lat_base, lon_base = 40.4168, -3.7038
        return pd.DataFrame(
            np.random.randn(5, 2) / 100 + [lat_base, lon_base],
            columns=['lat', 'lon']
        ), f"Simulación: 5 {palabra_clave} en {ciudad}."

    # MODO REAL (Google Maps)
    try:
        geocode_result = gmaps.geocode(ciudad)
        if not geocode_result: return None, f"No encontré la ciudad: {ciudad}"
            
        location = geocode_result[0]['geometry']['location']
        lat_centro = location['lat']
        lon_centro = location['lng']

        # Buscar lugares (Radio 2km)
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000
        )

        lugares = places_result.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave} en {ciudad}."

        data_mapa = []
        for lugar in lugares:
            lat = lugar['geometry']['location']['lat']
            lng = lugar['geometry']['location']['lng']
            data_mapa.append([lat, lng])
            
        # NOMBRES DE COLUMNAS CORRECTOS PARA STREAMLIT (lat, lon)
        df = pd.DataFrame(data_mapa, columns=['lat', 'lon'])
        
        return df, f"✅ He encontrado {len(df)} {palabra_clave} en {ciudad}."

    except Exception as e:
        return None, f"Error Google: {str(e)}"

# --- 3. RENDERIZADO CON MEMORIA ---
def renderizar_sidebar():
    st.subheader("📍 Social View (Google)")
    
    # Inicializar memoria del mapa si no existe
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    ciudad = st.text_input("Ciudad:", "Bogota")
    tipo = st.selectbox("Buscar:", ["Farmacias", "Parques", "Gimnasios", "Hospitales", "Cafeterías"])
    
    # BOTÓN DE BÚSQUEDA
    if st.button("🔍 Buscar"):
        with st.spinner("Buscando en el mapa..."):
            df_resultados, mensaje = buscar_lugares_google(ciudad, tipo)
            
            if df_resultados is not None and not df_resultados.empty:
                # 1. GUARDAMOS EN MEMORIA
                st.session_state.mapa_data = df_resultados
                st.success(mensaje)
                return mensaje # Devolvemos el mensaje para el chat
            else:
                st.error(mensaje)
                st.session_state.mapa_data = None # Limpiamos si hay error
                return None

    # PINTAR EL MAPA (Se ejecuta SIEMPRE si hay datos en memoria)
    if st.session_state.mapa_data is not None:
        st.map(st.session_state.mapa_data)
        
        # Botón para limpiar el mapa
        if st.button("🗑️ Limpiar Mapa"):
            st.session_state.mapa_data = None
            st.rerun()

    return None
