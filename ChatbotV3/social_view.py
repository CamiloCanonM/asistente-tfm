import streamlit as st
import pandas as pd
import googlemaps
import time
import numpy as np

# Intentamos conectar con Google
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        return googlemaps.Client(key=api_key)
    except:
        return None

def buscar_lugares_google(ciudad, palabra_clave):
    """
    1. Geocodifica la ciudad.
    2. Busca lugares cercanos (ej: Farmacias).
    3. Devuelve un DataFrame listo para st.map().
    """
    gmaps = obtener_cliente_google()
    
    # --- MODO SIMULACIÓN (Si falla Google o no hay clave) ---
    if not gmaps:
        st.toast("⚠️ Usando modo simulación (Falta API Key)", icon="ℹ️")
        time.sleep(1) # Simular carga
        # Coordenadas base (Madrid por defecto)
        lat_base, lon_base = 40.4168, -3.7038
        return pd.DataFrame(
            np.random.randn(5, 2) / 100 + [lat_base, lon_base],
            columns=['lat', 'lon']
        ), f"Mostrando 5 {palabra_clave} simulados en {ciudad}."

    # --- MODO REAL (Google Maps) ---
    try:
        # 1. Obtener coordenadas de la ciudad
        geocode_result = gmaps.geocode(ciudad)
        
        if not geocode_result:
            return None, f"No encontré la ciudad: {ciudad}"
            
        location = geocode_result[0]['geometry']['location']
        lat_centro = location['lat']
        lon_centro = location['lng']

        # 2. Buscar lugares cercanos (Radius 2000 metros)
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000
        )

        lugares = places_result.get('results', [])
        
        if not lugares:
            return None, f"No encontré {palabra_clave} en {ciudad}."

        # 3. Formatear datos para el mapa
        data_mapa = []
        nombres = []
        for lugar in lugares:
            lat = lugar['geometry']['location']['lat']
            lng = lugar['geometry']['location']['lng']
            nombre = lugar.get('name', 'Lugar')
            data_mapa.append([lat, lng])
            nombres.append(nombre)
            
        df = pd.DataFrame(data_mapa, columns=['lat', 'lon'])
        
        return df, f"✅ He encontrado {len(df)} {palabra_clave} en {ciudad} reales."

    except Exception as e:
        return None, f"Error de Google Maps: {str(e)}"

def renderizar_sidebar():
    """
    Función principal que pinta la barra lateral
    """
    st.subheader("📍 Social View (Google)")
    
    ciudad = st.text_input("Ciudad:", "Madrid")
    tipo = st.selectbox("Buscar:", ["Farmacias", "Parques", "Gimnasios", "Centros de Salud", "Cafeterías"])
    
    if st.button("🔍 Buscar"):
        with st.spinner("Consultando satélites..."):
            df_resultados, mensaje = buscar_lugares_google(ciudad, tipo)
            
            if df_resultados is not None and not df_resultados.empty:
                st.success(mensaje)
                st.map(df_resultados)
                return mensaje # Devolvemos texto para que el chat lo sepa
            else:
                st.error(mensaje)
                return None
    return None
