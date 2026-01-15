import streamlit as st
from streamlit_js_eval import get_geolocation
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
from langchain_core.messages import AIMessage

# ==========================================
# 1. LÓGICA DE BÚSQUEDA (OVERPASS API - GRATIS)
# ==========================================
@st.cache_data(show_spinner=False)
def buscar_lugares_osm(lat, lon, radio=2000, tipo="pharmacy"):
    """
    Busca puntos de interés en OpenStreetMap usando la API Overpass.
    Gratis y sin API Keys.
    """
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # Mapeo de términos comunes a etiquetas OSM
    tags_osm = {
        "droguerias": '["amenity"="pharmacy"]',
        "hospital": '["amenity"="hospital"]',
        "parques": '["leisure"="park"]',
        "restaurantes saludables": '["amenity"="restaurant"]',
        "cafes": '["amenity"="cafe"]'
    }
    
    # Seleccionamos la etiqueta adecuada o usamos una genérica
    tag_query = tags_osm.get(tipo.lower(), f'["amenity"="{tipo.lower()}"]')
    if tipo.lower() not in tags_osm:
        # Si no coincide, intentamos búsqueda genérica por nombre (más lento, mejor usar tags)
        pass 

    overpass_query = f"""
    [out:json];
    (
      node{tag_query}(around:{radio},{lat},{lon});
      way{tag_query}(around:{radio},{lat},{lon});
      relation{tag_query}(around:{radio},{lat},{lon});
    );
    out center;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=10)
        data = response.json()
        
        resultados = []
        for element in data['elements']:
            lat_el = element.get('lat') or element.get('center', {}).get('lat')
            lon_el = element.get('lon') or element.get('center', {}).get('lon')
            nombre = element.get('tags', {}).get('name', 'Sin nombre')
            
            if lat_el and lon_el:
                resultados.append({
                    "name": nombre,
                    "lat": lat_el,
                    "lon": lon_el,
                    "tipo": tipo
                })
        return pd.DataFrame(resultados)
    except Exception as e:
        return None

# ==========================================
# 2. RENDERIZADO BARRA LATERAL
# ==========================================
def renderizar_sidebar():
    st.subheader("🌍 Explorador Local")
    
    # 1. Obtener ubicación
    if "user_location" not in st.session_state:
        st.session_state.user_location = {'lat': 4.6097, 'lon': -74.0817} # Bogotá por defecto

    usar_gps = st.checkbox("📍 Usar mi ubicación real")
    if usar_gps:
        loc = get_geolocation()
        if loc:
            st.session_state.user_location = {
                'lat': loc['coords']['latitude'],
                'lon': loc['coords']['longitude']
            }
            st.caption(f"GPS Detectado: {st.session_state.user_location['lat']:.4f}, {st.session_state.user_location['lon']:.4f}")

    # 2. Controles de búsqueda
    tipo_lugar = st.selectbox("¿Qué buscas?", ["Droguerias", "Hospital", "Parques", "Restaurantes Saludables", "Cafe", "Gimnasios"])
    
    if st.button("🔍 Buscar en el Mapa"):
        lat = st.session_state.user_location['lat']
        lon = st.session_state.user_location['lon']
        
        with st.spinner(f"Buscando {tipo_lugar}..."):
            df_lugares = buscar_lugares_osm(lat, lon, tipo=tipo_lugar)
            
            if df_lugares is not None and not df_lugares.empty:
                st.session_state.mapa_osm_data = df_lugares
                
                # --- PUENTE CON LA IA ---
                texto_ia = f"He encontrado estos {tipo_lugar}s cerca:\n"
                for idx, row in df_lugares.head(5).iterrows():
                    texto_ia += f"- {row['name']}\n"
                st.session_state.geo_contexto = texto_ia
                st.toast("✅ IA Actualizada con el mapa")
                # ----------------------
            else:
                st.warning("No se encontraron lugares cerca.")
                st.session_state.geo_contexto = "No se encontraron lugares cercanos."

# ==========================================
# 3. RENDERIZADO CENTRAL (MAPA FOLIUM)
# ==========================================
def mostrar_mapa_central():
    st.divider()
    st.subheader("🗺️ Mapa de Bienestar")
    
    # Coordenadas base
    lat_center = st.session_state.get("user_location", {}).get('lat', 4.6097)
    lon_center = st.session_state.get("user_location", {}).get('lon', -74.0817)

    # Crear mapa base
    m = folium.Map(location=[lat_center, lon_center], zoom_start=15)
    
    # Marcador Usuario
    folium.Marker(
        [lat_center, lon_center], 
        popup="Tú estás aquí", 
        icon=folium.Icon(color="blue", icon="user")
    ).add_to(m)

    # Marcadores de Resultados (Si hay)
    if "mapa_osm_data" in st.session_state and st.session_state.mapa_osm_data is not None:
        for idx, row in st.session_state.mapa_osm_data.iterrows():
            folium.Marker(
                [row['lat'], row['lon']],
                popup=row['name'],
                icon=folium.Icon(color="green", icon="info-sign")
            ).add_to(m)

    # Renderizar en Streamlit
    st_data = st_folium(m, width="100%", height=400)
