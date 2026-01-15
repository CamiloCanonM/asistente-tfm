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
# 2. RENDERIZADO BARRA LATERAL (CON LISTA DESPLEGABLE)
# ==========================================
def renderizar_sidebar():
    st.subheader("🌍 Modulo de Georeferenciación")
    
    # 1. Ubicación
    if "user_location" not in st.session_state:
        st.session_state.user_location = {'lat': 4.6097, 'lon': -74.0817} # Bogotá por defecto

    usar_gps = st.checkbox("📍 Usar mi ubicación real", value=True)
    
    if usar_gps:
        try:
            loc = get_geolocation()
            if loc:
                st.session_state.user_location = {
                    'lat': loc['coords']['latitude'],
                    'lon': loc['coords']['longitude']
                }
                st.caption(f"GPS Activo: {st.session_state.user_location['lat']:.4f}")
        except:
            st.warning("GPS no disponible.")

    # 2. SELECTOR DE BÚSQUEDA (NUEVO)
    st.write("¿Qué necesitas hoy?")
    opcion = st.selectbox(
        "Selecciona una categoría:",
        ["Drogueria", "Hospital", "Restaurante Saludable", "Parques", "Supermercado", "Tienda Naturista", "Gimnasio", "Tienda deportiva"],
        label_visibility="collapsed"
    )

    # Si elige "Otro...", mostramos el campo de texto
    query = opcion
    if opcion == "Otro...":
        query = st.text_input("Escribe qué buscas:", placeholder="Ej: Cine, Zapatería...")
    
    # Botón de Buscar
    if st.button("🔍 Buscar en el Mapa", use_container_width=True):
        if not query:
            st.warning("Por favor escribe qué buscas.")
        else:
            lat = st.session_state.user_location['lat']
            lon = st.session_state.user_location['lon']
            
            with st.spinner(f"Buscando {query}..."):
                df_lugares = buscar_lugares_google(lat, lon, keyword=query)
                
                if df_lugares is not None and not df_lugares.empty:
                    st.session_state.mapa_data = df_lugares
                    
                    # --- PUENTE CON LA IA ---
                    texto_ia = f"Resultados del mapa para '{query}':\n"
                    for idx, row in df_lugares.head(5).iterrows():
                        texto_ia += f"- {row['name']} ({row['rating']}⭐) a {row['distancia']}km.\n"
                    
                    st.session_state.geo_contexto = texto_ia
                    st.toast(f"✅ Encontrados {len(df_lugares)} lugares.")
                    st.rerun()
                else:
                    st.warning("No se encontraron resultados cercanos.")
                    st.session_state.mapa_data = None

# ==========================================
# 3. RENDERIZADO CENTRAL (CON BOTÓN OCULTAR)
# ==========================================
def mostrar_mapa_central():
    # Solo mostramos si hay datos cargados
    if "mapa_data" in st.session_state and st.session_state.mapa_data is not None:
        
        st.divider()
        
        # Cabecera con Botón de Cerrar
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("🗺️ Mapa de Resultados")
        with c2:
            # BOTÓN PARA OCULTAR EL MAPA
            if st.button("❌ Ocultar", use_container_width=True):
                st.session_state.mapa_data = None
                st.rerun()

        # Recuperamos coordenadas
        lat_center = st.session_state.get("user_location", {}).get('lat', 4.6097)
        lon_center = st.session_state.get("user_location", {}).get('lon', -74.0817)

        # Mapa Folium
        m = folium.Map(location=[lat_center, lon_center], zoom_start=14)
        
        # Marcador Usuario
        folium.Marker(
            [lat_center, lon_center], 
            popup="<b>TÚ</b>", 
            icon=folium.Icon(color="blue", icon="user", prefix="fa")
        ).add_to(m)

        # Marcadores Resultados
        df = st.session_state.mapa_data
        for idx, row in df.iterrows():
            html_popup = f"<b>{row['name']}</b><br>⭐ {row['rating']}<br>📍 {row['distancia']} km"
            folium.Marker(
                [row['lat'], row['lon']],
                popup=html_popup,
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)

        # Renderizar
        st_folium(m, width="100%", height=500)
        
        # Tabla simple
        st.caption("Detalles de los lugares encontrados:")
        st.dataframe(
            df[['name', 'rating', 'distancia', 'address']],
            hide_index=True,
            use_container_width=True
        )
