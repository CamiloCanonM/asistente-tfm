import streamlit as st
import pandas as pd
import googlemaps
import time

# --- 1. CONFIGURACIÓN DEL CLIENTE ---
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

# --- 2. FUNCIÓN DE BÚSQUEDA ---
def buscar_lugares_google(ciudad, palabra_clave):
    gmaps = obtener_cliente_google()
    
    # MODO REAL
    try:
        # 1. Geocoding
        geocode_result = gmaps.geocode(ciudad)
        if not geocode_result: 
            return None, f"No encontré la ciudad: {ciudad}"
            
        location = geocode_result[0]['geometry']['location']
        lat_centro = location['lat']
        lon_centro = location['lng']

        # 2. Places Search
        places_result = gmaps.places_nearby(
            location=(lat_centro, lon_centro),
            keyword=palabra_clave,
            radius=2000
        )

        lugares = places_result.get('results', [])
        if not lugares: 
            return None, f"No encontré {palabra_clave} en {ciudad}."

        # 3. Construir DataFrame limpio
        data_mapa = []
        for lugar in lugares:
            lat = float(lugar['geometry']['location']['lat']) # Forzar decimal
            lng = float(lugar['geometry']['location']['lng']) # Forzar decimal
            data_mapa.append([lat, lng])
            
        # Usamos nombres explícitos para que st.map no falle
        df = pd.DataFrame(data_mapa, columns=['latitude', 'longitude'])
        
        return df, f"✅ He encontrado {len(df)} {palabra_clave} en {ciudad}."

    except Exception as e:
        return None, f"Error: {str(e)}"

# --- 3. RENDERIZADO (LA PARTE IMPORTANTE) ---
def renderizar_sidebar():
    st.subheader("📍 Mapa en vivo")
    
    # Inicializar memoria
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None

    ciudad = st.text_input("Ciudad:", "Bogota")
    tipo = st.selectbox("Buscar:", ["Farmacias", "Hospitales", "Parques", "Gimnasios"])
    
    # BOTÓN
    if st.button("🔍 Buscar ahora"):
        with st.spinner("Conectando satélite..."):
            df_resultados, mensaje = buscar_lugares_google(ciudad, tipo)
            
            if df_resultados is not None:
                st.session_state.mapa_data = df_resultados # Guardar en memoria
                st.success(mensaje)
                return mensaje
            else:
                st.error(mensaje)
    
    # --- PINTAR EL MAPA SIEMPRE QUE HAYA DATOS ---
    # Esto está fuera del botón para que no desaparezca al recargar
    if st.session_state.mapa_data is not None:
        st.divider()
        st.caption("Resultados en el mapa:")
        # Forzamos pintar en la sidebar explícitamente
        st.sidebar.map(st.session_state.mapa_data, zoom=13)
        
        if st.button("Limpiar Mapa"):
            st.session_state.mapa_data = None
            st.rerun()
            
    return None
