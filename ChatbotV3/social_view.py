import streamlit as st
import pandas as pd
import googlemaps
import math
from streamlit_js_eval import get_geolocation
from langchain_core.messages import AIMessage

# ==========================================
# 1. MATEMÁTICAS: CALCULAR DISTANCIA (Haversine)
# ==========================================
def calcular_distancia(lat1, lon1, lat2, lon2):
    # Radio de la tierra en km
    R = 6371  
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distancia = R * c
    return round(distancia, 2) # Retorna km con 2 decimales

# ==========================================
# 2. CONEXIÓN Y BÚSQUEDA GOOGLE
# ==========================================
def obtener_cliente_google():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key: return None
        return googlemaps.Client(key=api_key)
    except: return None

def buscar_lugares_google(ciudad, palabra_clave, lat_gps=None, lon_gps=None):
    gmaps = obtener_cliente_google()
    
    if not gmaps: return None, "Error: No se detectó la API Key de Google."

    try:
        # A. DEFINIR PUNTO DE PARTIDA (REFERENCIA)
        if lat_gps is not None and lon_gps is not None:
            lat_ref, lon_ref = lat_gps, lon_gps
            origen_texto = "tu ubicación actual"
        else:
            geocode_result = gmaps.geocode(ciudad)
            if not geocode_result: return None, f"No encontré la ciudad: {ciudad}"
            loc = geocode_result[0]['geometry']['location']
            lat_ref, lon_ref = loc['lat'], loc['lng']
            origen_texto = ciudad

        # B. CONSULTAR API DE LUGARES
        places_result = gmaps.places_nearby(
            location=(lat_ref, lon_ref),
            keyword=palabra_clave,
            radius=2000, # 2km a la redonda
            open_now=False
        )

        lugares = places_result.get('results', [])
        if not lugares: return None, f"No encontré {palabra_clave} cerca de {origen_texto}."

        # C. PROCESAR DATOS
        data_completa = []

        # 1. USUARIO (Distancia 0)
        if lat_gps is not None and lon_gps is not None:
            data_completa.append({
                'latitude': float(lat_ref),
                'longitude': float(lon_ref),
                'name': "🔵 TU UBICACIÓN",
                'address': "Estás aquí",
                'rating': "YO",
                'place_id': "USER_LOC",
                'color': '#0000FF',     # Azul
                'size': 40,             # Tamaño mediano
                'distancia': 0.0
            })

        # 2. LUGARES ENCONTRADOS
        for l in lugares:
            lat_lugar = l['geometry']['location']['lat']
            lon_lugar = l['geometry']['location']['lng']
            
            # Calcular distancia real
            km = calcular_distancia(lat_ref, lon_ref, lat_lugar, lon_lugar)

            data_completa.append({
                'latitude': float(lat_lugar),
                'longitude': float(lon_lugar),
                'name': l.get('name', 'Sin nombre'),
                'address': l.get('vicinity', ''),
                'rating': l.get('rating', 'N/A'),
                'place_id': l.get('place_id'),
                'color': '#FF0000',     # Rojo
                'size': 20,             # Tamaño normal
                'distancia': km
            })
            
        # D. ORDENAR POR CERCANÍA
        df = pd.DataFrame(data_completa)
        df = df.sort_values(by='distancia', ascending=True)
        
        return df, f"✅ Encontré {len(lugares)} sitios cercanos."

    except Exception as e:
        return None, f"Error técnico: {str(e)}"

# ==========================================
# 3. RENDERIZADO VISUAL (SIDEBAR)
# ==========================================
def renderizar_sidebar():
    st.subheader("📍 Mapa Interactivo")
    
    # Inicializar estado
    if "mapa_data" not in st.session_state:
        st.session_state.mapa_data = None
    if "lugar_seleccionado_id" not in st.session_state:
        st.session_state.lugar_seleccionado_id = None

    # --- CONTROLES ---
    tipo = st.selectbox("
