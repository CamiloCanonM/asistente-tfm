import streamlit as st
from streamlit_js_eval import get_geolocation
import folium
from streamlit_folium import st_folium
import pandas as pd
import googlemaps
import math

# ==========================================
# 1. LÓGICA DE BÚSQUEDA (GOOGLE PLACES API)
# ==========================================
def calcular_distancia(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)

@st.cache_data(show_spinner=False)
def buscar_lugares_google(lat_ref, lon_ref, keyword, radio=2000):
    """
    Busca usando Google Places pero devuelve formato limpio para el mapa.
    """
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ Falta la GOOGLE_API_KEY en secrets.toml")
            return None

        gmaps = googlemaps.Client(key=api_key)
        
        # Búsqueda en Google
        res = gmaps.places_nearby(location=(lat_ref, lon_ref), keyword=keyword, radius=radio)
        lugares = res.get('results', [])
        
        if not lugares:
            return pd.DataFrame() # Vacío

        data = []
        for l in lugares:
            lat = l['geometry']['location']['lat']
            lng = l['geometry']['location']['lng']
            nombre = l.get('name', 'Sin nombre')
            rating = l.get('rating', 'N/A')
            direccion = l.get('vicinity', '')
            km = calcular_distancia(lat_ref, lon_ref, lat, lng)
            
            data.append({
                "name": nombre,
                "lat": lat,
                "lon": lng,
                "rating": rating,
                "address": direccion,
                "distancia": km
            })
            
        return pd.DataFrame(data).sort_values(by='distancia')
    
    except Exception as e:
        st.error(f"Error de Google: {str(e)}")
        return None

# ==========================================
# 2. RENDERIZADO BARRA LATERAL
# ==========================================
def renderizar_sidebar():
    st.subheader("🌍 Explorador (Google Data)")
    
    # 1. Ubicación (Por defecto Bogotá si falla GPS)
    if "user_location" not in st.session_state:
        st.session_state.user_location = {'lat': 4.6097, 'lon': -74.0817}

    usar_gps = st.checkbox("📍 Usar mi ubicación real", value=True)
    
    if usar_gps:
        try:
            loc = get_geolocation()
            if loc:
                st.session_state.user_location = {
                    'lat': loc['coords']['latitude'],
                    'lon': loc['coords']['longitude']
                }
                st.caption(f"GPS OK: {st.session_state.user_location['lat']:.4f}")
        except:
            st.warning("GPS no disponible en este navegador.")

    # 2. Buscador Libre
    query = st.text_input("¿Qué buscas hoy?", placeholder="Ej: Farmacia 24h, Pizza, Cine...")
    
    if st.button("🔍 Buscar en Google"):
        if not query:
            st.warning("Escribe algo para buscar.")
        else:
            lat = st.session_state.user_location['lat']
            lon = st.session_state.user_location['lon']
            
            with st.spinner(f"Consultando a Google sobre '{query}'..."):
                df_lugares = buscar_lugares_google(lat, lon, keyword=query)
                
                if df_lugares is not None and not df_lugares.empty:
                    st.session_state.mapa_data = df_lugares
                    
                    # --- PUENTE CON LA IA (GEO CONTEXTO) ---
                    texto_ia = f"Resultados de Google Maps para '{query}':\n"
                    for idx, row in df_lugares.head(5).iterrows():
                        texto_ia += f"- {row['name']} ({row['rating']}⭐) a {row['distancia']}km. Dirección: {row['address']}\n"
                    
                    st.session_state.geo_contexto = texto_ia
                    st.toast(f"✅ Google encontró {len(df_lugares)} sitios. IA Actualizada.")
                    # ---------------------------------------
                    
                    # Forzamos recarga para mostrar mapa
                    st.rerun()
                else:
                    st.warning("Google no encontró resultados cercanos.")
                    st.session_state.geo_contexto = "No se encontraron resultados."

# ==========================================
# 3. RENDERIZADO CENTRAL (MAPA FOLIUM)
# ==========================================
def mostrar_mapa_central():
    st.divider()
    st.subheader("🗺️ Mapa de Resultados")
    
    # Recuperamos coordenadas centro
    lat_center = st.session_state.get("user_location", {}).get('lat', 4.6097)
    lon_center = st.session_state.get("user_location", {}).get('lon', -74.0817)

    # Creamos mapa base
    m = folium.Map(location=[lat_center, lon_center], zoom_start=14)
    
    # Marcador: TÚ
    folium.Marker(
        [lat_center, lon_center], 
        popup="<b>TU UBICACIÓN</b>", 
        icon=folium.Icon(color="blue", icon="user", prefix="fa")
    ).add_to(m)

    # Marcadores: RESULTADOS
    if "mapa_data" in st.session_state and st.session_state.mapa_data is not None:
        df = st.session_state.mapa_data
        
        for idx, row in df.iterrows():
            # HTML para el popup (Nombre + Estrellas)
            html_popup = f"<b>{row['name']}</b><br>⭐ {row['rating']}<br>📍 {row['distancia']} km"
            
            folium.Marker(
                [row['lat'], row['lon']],
                popup=html_popup,
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

    # Mostrar mapa
    st_folium(m, width="100%", height=500)
    
    # Tabla de resultados debajo
    if "mapa_data" in st.session_state and st.session_state.mapa_data is not None:
        st.markdown("### 📋 Detalles")
        st.dataframe(
            st.session_state.mapa_data[['name', 'rating', 'distancia', 'address']],
            hide_index=True
        )
